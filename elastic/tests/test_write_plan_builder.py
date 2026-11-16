"""The writeback (WriteRouter) takes an event's requests as they are: one
NVMe engine item per entry, in order, from the slot's K and V cache blocks
(KV) or its activation buffer (ACT). It derives no offset, merges nothing, cuts
nothing.

And the writeback sends a request only once its own ticket's mark has passed
on the producer's stream, and `stop()` — a stop ticket marked behind the last
licensed write — lands everything it owes before the writeback exits. These
run a real WriteRouter over a blob in `tmp_path` (an NVMe engine of the
test's own, the writeback thread, and a slot pool that is bookkeeping only),
with the marks riding a real CUDA stream."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout, _uma_byte_view
from elastic.loader.nvme_loader import NvmeEngine
from elastic.runtime.labels import block_is_act
from elastic.vllm_runtime.layer_pool import LayerSlot, LayerSlotPool
from elastic.vllm_runtime.nvme_write_router import WriteRouter

N_SEQ, L, NBLK, BS, H, D = 4, 2, 8, 4, 2, 32  # kb = 512 B = the O_DIRECT grain

# `stop()` marks its ticket on a CUDA stream, so every test that builds a
# router needs one.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the stop ticket rides a CUDA stream")

# GPU busy-work long enough (a few hundred ms) that a mark queued behind it
# is still unpassed when the test looks.
_SLEEP_CYCLES = 200_000_000

_BLOB = "blob.bin"        # the router's blob, under the test's `tmp_path`


def _pool() -> LayerSlotPool:
    """The slot pool's bookkeeping only. Nothing here acquires a visit, so
    the writeback's release always finds no visit live."""
    p = object.__new__(LayerSlotPool)
    p._slots = [LayerSlot(idx=i, raw_buf=None, t_buf=None)
                for i in range(2)]
    p.n_slots = 2
    p._n_layers = L
    p._head_visit = p._tail_visit = 0
    p._bind = list(range(L))
    p._on_bind = lambda layer, slot: None
    p._cv = threading.Condition()
    p.n_acquired = p.n_use = p.n_released = p.n_moved = 0
    p.t_use_wait = 0.0
    p.release_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    return p


def _aligned_cache(shape):
    """A KV cache tensor whose data pointer is page-aligned, like the UMA
    allocation the runtime hands the router."""
    n = int(np.prod(shape)) * 2
    raw = torch.empty(n + 4096, dtype=torch.uint8)
    off = (-raw.data_ptr()) % 4096
    t = raw[off:off + n].view(torch.float16).view(*shape)
    t.normal_()
    return t


def _stop(router, guard_s: float = 60.0) -> None:
    """`router.stop()`, which has no timeout of its own, on a helper thread —
    so a writeback that never exits fails the test instead of hanging the
    run. Re-raises whatever `stop()` raised.

    The helper first binds the CUDA context, as the forward thread that calls
    `stop()` in a run has it bound: with no mark made yet the stop ticket
    rides the caller's current (default) stream, and the driver refuses a
    callback on the default stream from a thread with no current context
    (CUresult 201). `query()` binds it without waiting for queued work."""
    out = {}

    def run():
        try:
            torch.cuda.current_stream().query()
            router.stop()
        except BaseException as e:           # handed back to the test
            out["e"] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(guard_s)
    assert not th.is_alive(), "stop() did not return: the tail never exited"
    if "e" in out:
        raise out["e"]


# One router per test, stopped before the next is built: the stream signal's
# binding is process-wide, and a second live router would rebind it out from
# under the first one's stop ticket.
@pytest.fixture
def router(tmp_path):
    layout = BlockKVLayout(n_seq=N_SEQ, num_layers=L, max_blocks_per_seq=NBLK,
                           block_size=BS, num_heads=H, head_dim=D,
                           dtype=torch.float16, alpha=[0.0, 0.5])
    p = tmp_path / _BLOB
    with open(p, "wb") as f:
        f.truncate(max(4 << 20, layout.total_bytes))
    caches = [_aligned_cache((2, N_SEQ * NBLK, BS, H, D)) for _ in range(L)]
    eng = NvmeEngine(str(p), read_ring=8, write_ring=256)
    r = WriteRouter(layout, caches, engine=eng, pool=_pool())
    r.layout.set_prompt_len(4 * BS + 1)     # n=4, rem=1: mutable prompt blocks
    yield r
    if r._feeder is not None:                 # the test did not stop it
        _stop(r)
    eng.close()


def _nb(v):
    return int(v.nbytes) if hasattr(v, "nbytes") else len(v)


def _addr(v):
    return int(v.ctypes.data) if hasattr(v, "ctypes") else \
        int(np.frombuffer(v, dtype=np.uint8).ctypes.data)


def _bytes(v):
    return bytes(np.frombuffer(v, dtype=np.uint8))


def _read(tmp_path, off, n):
    with open(tmp_path / _BLOB, "rb") as f:
        f.seek(off)
        return f.read(n)


def _register(router, layer, entries, t):
    """`enqueue_writes` for a KV event licensed by ticket `t`, returning the
    descriptor it registered. Called before `t` is marked and while no
    earlier mark has passed, so the writeback has not been woken to take it
    off the inbox."""
    router.enqueue_writes(layer, "kv_batch", entries, gate_seq=t, path="kv")
    return router._inbox[-1]


def _kv_block(router, layer, k):
    """(file offset, K view, V offset, V view) of KV block `k`."""
    lay = router.layout
    kv, vv = router.kv_block_views(layer, lay.mem_block(layer, k))
    off = lay.k_offset(layer, k)
    return off, kv, off + lay.v_delta(layer), vv


# ---- requests as they are ----------------------------------------------------

@needs_cuda
def test_kv_block_views_are_the_cache_blocks(router):
    """A KV view is the cache block's own bytes (the DMA source), K from
    plane 0 and V from plane 1, one block long."""
    layout = router.layout
    kb = layout.block_bytes
    cache = router.kv_caches[0]
    for cb in (1, 5, N_SEQ * NBLK - 1):
        kv, vv = router.kv_block_views(0, cb)
        assert _nb(kv) == _nb(vv) == kb
        assert _addr(kv) == _addr(_uma_byte_view(cache[0, cb]))
        assert _addr(vv) == _addr(_uma_byte_view(cache[1, cb]))


@needs_cuda
@pytest.mark.parametrize("layer", [0, 1])
def test_last_n_seq_blocks_one_request_per_block_k_then_v(router, layer,
                                                          tmp_path):
    """A step's last N_seq blocks after the mutable prompt blocks: every
    sequence's block n+1, N_seq adjacent numbers k — one K request per KV block
    at `k_offset`, then one V request per block at `+ v_delta`, in order,
    handed to the NVMe engine as they are and landing where their offsets
    say."""
    layout = router.layout
    al = layout.alpha_of_layer(layer)
    n_immutable = layout.n_immutable
    ks = [int(layout.ks_of_tokens(s, (n_immutable + 1) * BS))
          for s in range(N_SEQ)]
    kvs = [k for k in ks if not block_is_act(k, al)]
    entries = []
    for k in kvs:
        kv, _v = router.kv_block_views(layer, layout.mem_block(layer, k))
        entries.append((layout.k_offset(layer, k), [kv]))
    for k in kvs:
        _k, vv = router.kv_block_views(layer, layout.mem_block(layer, k))
        entries.append((layout.k_offset(layer, k) + layout.v_delta(layer),
                        [vv]))
    s = torch.cuda.Stream()
    t = router.reserve()
    d = _register(router, layer, entries, t)
    router.mark(t, stream=s)
    _stop(router)
    assert d.completed
    w = d.writes
    assert [(o, [_addr(v) for v in b]) for o, b in w] == \
        [(o, [_addr(v) for v in b]) for o, b in entries]
    assert len(w) == 2 * len(kvs)
    assert router.engine_write_stats()[0] == len(w)
    # the requests are file-adjacent by the layout, not by any merge here
    offs = [o for o, _b in w]
    kb = layout.block_bytes
    n = len(kvs)
    assert offs[:n] == [offs[0] + i * kb for i in range(n)]
    assert offs[n:] == [offs[n] + i * kb for i in range(n)]
    for o, b in entries:
        assert _read(tmp_path, o, kb) == _bytes(b[0])


# ---- the licence, and stop ------------------------------------------------------

@needs_cuda
def test_the_writeback_hears_the_mark_and_nothing_else(router, tmp_path):
    """A registered descriptor sits until its own ticket's mark passes on
    the producer's stream: nothing polls it, registering wakes nobody, and a
    mark still queued behind GPU work has not passed. Once the GPU gets
    there the write goes out, and `stop()` lands it."""
    off, kv, _voff, _vv = _kv_block(router, 0, 0)
    kb = router.layout.block_bytes
    zero = bytes(kb)
    assert _read(tmp_path, off, kb) == zero
    s = torch.cuda.Stream()
    t = router.reserve()
    d = _register(router, 0, [(off, [kv])], t)
    threading.Event().wait(0.3)                 # registering wakes nobody
    assert d.n_items == 0 and not d.completed
    assert router.engine_write_stats()[0] == 0
    assert router._pub_cnt[0] < t
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        torch.cuda._sleep(_SLEEP_CYCLES)
    router.mark(t, stream=s)                    # queued behind the sleep
    assert router._pub_cnt[0] < t, "heard before the GPU got there"
    assert d.n_items == 0
    _stop(router)
    assert d.completed and d.n_items == 1
    assert router._pub_cnt[0] == router._stop_ticket > t
    assert _read(tmp_path, off, kb) == _bytes(kv)
    assert router.engine_write_stats()[0] == 1


@needs_cuda
def test_stop_lands_every_licensed_write_and_joins_the_writeback(router,
                                                                 tmp_path):
    """`stop()` is called while the GPU is still short of the writes' marks:
    its ticket rides the same stream behind them, so by the time the
    writeback hears it every licensed write has been sent, and the writeback
    exits only once they have landed. The join has no timeout."""
    off0, k0, voff0, v0 = _kv_block(router, 0, 0)
    off1, k1, voff1, v1 = _kv_block(router, 0, 1)
    kb = router.layout.block_bytes
    tail = router._feeder
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        torch.cuda._sleep(_SLEEP_CYCLES)
    ds = []
    for entries in ([(off0, [k0]), (off1, [k1])],
                    [(voff0, [v0]), (voff1, [v1])]):
        t = router.reserve()
        ds.append(_register(router, 0, entries, t))
        router.mark(t, stream=s)
    assert router._pub_cnt[0] < ds[0].gate_seq  # nothing licensed yet
    _stop(router)
    assert not tail.is_alive() and router._feeder is None
    assert all(d.completed for d in ds)
    for off, v in ((off0, k0), (off1, k1), (voff0, v0), (voff1, v1)):
        assert _read(tmp_path, off, kb) == _bytes(v)
    assert router.engine_write_stats()[0] == 4


@needs_cuda
def test_stop_with_nothing_owed_returns(router):
    """No mark was ever made, so the stop ticket rides the caller's current
    stream; the writeback hears it, owes nothing and exits."""
    tail = router._feeder
    assert router._mark_stream is None
    _stop(router)
    assert not tail.is_alive() and router._feeder is None
    assert router._pub_cnt[0] == router._stop_ticket
    assert router.engine_write_stats() == (0, 0)


def test_a_non_nhd_cache_layout_is_refused():
    """The blob's block byte order is the NHD cache block order, so importing
    the runtime under another cache layout fails before vLLM is touched."""
    env = {**os.environ, "VLLM_KV_CACHE_LAYOUT": "HND"}
    r = subprocess.run([sys.executable, "-c", "import elastic.vllm_runtime"],
                       cwd=Path(__file__).resolve().parents[2], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "requires VLLM_KV_CACHE_LAYOUT=NHD" in r.stderr
