"""The loader's completion accounting: what one read landing makes the
loader (ReadRouter) do.

A layer's plan is its weight segments, its ACT reads (grouped into kv_gen
launches) and its K,V reads (in pairs), handed to the NVMe engine in that
order. The io_uring result is the only proof bytes landed, so each
completion credits exactly one owner (`_reap_segment`):

  * a weight segment counts the layer's weight down; the last one to land —
    whichever it is — publishes the layer to the forward pass (`mark_ready`);
  * a K or V half counts its pair down; the pair's rank run goes to the
    forward pass's arrival feed once both halves are in;
  * an ACT read counts its kv_gen launch down; a launch is run through
    kv_gen and handed over once all its reads and the layer's weight are in,
    in launch order and never ahead of an earlier layer's launch
    (`_dispatch_ready`);
  * a failed or short read raises — crediting it would hand the forward pass
    bytes that never landed.

A layer's feed closes when every read has landed and every kv_gen launch has
been issued (`_plan_complete`), oldest plan first.

The loader here is built by its real constructor over a recording NVMe
engine port, a pool that lets it acquire a fixed number of visits, a real
`LayerArrival` and real CPU-resident KV caches, so `_fill`, `_plan_layer`,
`_build_plan`, `_direct_read_kv`, `_sub_segment`, `_reap_segment`,
`_take_in` and `_finish_plan` all run as in production. The one stand-in is
`_pipeline_layer_reads`, whose kv_gen launches run torch kernels on the
kv_gen stream; see `_act_reads_without_the_gpu`.
"""

import errno
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from elastic.loader import nvme_loader
from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.weight_table import LayerSpec
from elastic.vllm_runtime.layer_arrival import LayerArrival
from elastic.vllm_runtime.loader_thread import ReadRouter, act_segments_and_launches

MiB = 1 << 20
W_BYTES = 5 * MiB // 2        # one layer's weight: segments of 1, 1, 0.5 MiB
W_FIDX = 3                    # the file index the weight blob attaches as
N_SEQ, N, BS, H, D = 2, 160, 16, 4, 64
KB = BS * H * D * 2           # one block's K (or V, or ACT) bytes: 8 KiB
PAIR = MiB // KB              # blocks per K,V segment at 1 MiB segments


class _NoEvents:
    """Stands in for `EventRing` (CUDA events); nothing here records one."""

    def __init__(self, *a, **k):
        pass


class _Kvn:
    """The NVMe engine port as the loader sees it: submissions are
    recorded, and `try_wait_one` hands back whatever completions the test
    queued."""

    capacity = 1 << 16

    def __init__(self, fd):
        self._fd = fd
        self.subs = []
        self.cq = []

    def completion_fd(self):
        return self._fd

    def attach(self, path):
        return W_FIDX

    def submit_read(self, offset, size, view, user_data, file_idx=0):
        addr = view.ctypes.data if isinstance(view, np.ndarray) else None
        self.subs.append(SimpleNamespace(
            off=int(offset), size=int(size), ud=int(user_data),
            fidx=file_idx, addr=addr))

    def try_wait_one(self):
        return self.cq.pop(0) if self.cq else None


class _Pool:
    """Lets the loader acquire visits 0 .. n_open−1 and records the layers
    published ready."""

    def __init__(self, release_fd, n_open):
        self.release_fd = release_fd
        self.n_open = n_open
        self.ready = []

    def acquire(self, visit, layer):
        if visit >= self.n_open:
            return None
        return SimpleNamespace(raw_buf=np.zeros(W_BYTES, np.uint8),
                               layer=layer)

    def mark_ready(self, slot):
        self.ready.append(slot.layer)


def _act_reads_without_the_gpu(head, kvn, segment_blocks, launch_blocks):
    """Stands in for `_pipeline_layer_reads`: the same segment and kv_gen
    launch shapes — segments and launches cut by the real
    `act_segments_and_launches` over the layer's ACT block region, each
    launch `{"rank0", "nblk", "remaining"}` owned by the segments it
    overlaps — and a dispatch that hands the launch over through the
    loader's own `_arrival_push`. Left out is only the kv_gen the real
    dispatch launches before that push."""

    def _reads(layer_idx, slot, n_act, block_size, L, act_loader,
               weight_table, act_pool):
        dcs, wvs = act_segments_and_launches(n_act, L.region_bases(layer_idx)[0],
                                  L.block_bytes, segment_blocks, launch_blocks)
        sts = [{"rank0": w0, "nblk": n, "remaining": len(ids)}
               for (w0, n, ids) in wvs]
        owners = [[] for _ in dcs]
        for st, (_w0, _n, ids) in zip(sts, wvs):
            for d in ids:
                owners[d].append(st)
        segments = [(off, 0, (off, n * L.block_bytes, None), own)
                 for (off, n, _r0), own in zip(dcs, owners)]

        def _dispatch(st):
            head._arrival_push(layer_idx, 0, st["rank0"], st["nblk"], None)

        def _sub(pay, ud):
            kvn.submit_read(pay[0], pay[1], pay[2], user_data=ud)

        return {"sts": sts, "segments": segments, "dispatch": _dispatch,
                "sub": _sub}

    return _reads


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(nvme_loader, "EventRing", _NoEvents)
    monkeypatch.delenv("ELASTIC_LOADER_TIMING", raising=False)
    monkeypatch.setenv("ELASTIC_UNIFIED_WEIGHT_MB", "1")
    monkeypatch.setenv("ELASTIC_READ_SPAN_MB", "1")
    fds = []

    def _make(*, alpha=0.0, per_seq=0, n_layers=1, n_open=1, run_cap=16,
              launch_blocks=32, fifo=1 << 16):
        """A loader over `n_layers` layers. With `per_seq`, the loader is at
        decode step 1 of a batch whose prompt is one token short of
        `per_seq` blocks, so the new token fills the last of them and it
        reloads `per_seq` blocks of each of the N_seq sequences; without it the
        loader is in a prefill step and reads only the weights."""
        fds.extend(os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
                   for _ in range(2))
        layout = BlockKVLayout(n_seq=N_SEQ, num_layers=n_layers,
                               max_blocks_per_seq=N, block_size=BS,
                               num_heads=H, head_dim=D, dtype=torch.float16,
                               alpha=[alpha] * n_layers)
        nb = layout.mem_blocks_needed
        caches = [(torch.zeros(nb, BS, H, D, dtype=torch.float16),
                   torch.zeros(nb, BS, H, D, dtype=torch.float16))
                  for _ in range(n_layers)]
        kvn = _Kvn(fds[-2])
        kvn.capacity = fifo
        pool = _Pool(fds[-1], n_open)
        arr = LayerArrival(n_layers)
        bls = {"arrival": arr, "storage_layout": layout,
               "act_pool": None, "act_loader": SimpleNamespace(),
               "kv_caches": caches}
        specs = [LayerSpec(file_start=l * W_BYTES, bytes_padded=W_BYTES,
                           params=())
                 for l in range(n_layers)]
        head = ReadRouter(pool, kvn, "weights.bin",
                          SimpleNamespace(layers=specs), bls)
        head._pipeline_layer_reads = _act_reads_without_the_gpu(
            head, kvn, run_cap, launch_blocks)
        if per_seq:
            # The batch's prefill step is behind the loader: it takes the
            # declaration as it leaves that step.
            layout.declare_batch(per_seq * BS - 1, gen=2)
            head._next_step()
        return SimpleNamespace(head=head, kvn=kvn, pool=pool, arr=arr,
                               layout=layout, caches=caches)

    yield _make
    for fd in fds:
        os.close(fd)


def _fill(r):
    r.head._fill(r.head._plans)


def _block_region(r, sub, layer=0):
    """W (weight), A (ACT), K or V — by file and by file offset."""
    if sub.fidx == W_FIDX:
        return "W"
    _a, k_base, v_delta = r.layout.region_bases(layer)
    if sub.off < k_base:
        return "A"
    return "K" if sub.off < k_base + v_delta else "V"


def _of(r, kind, layer=0):
    return [s for s in r.kvn.subs if _block_region(r, s, layer) == kind]


def _land(r, *subs, res=None):
    """What the NVMe engine posts for `subs` — the full size unless `res`
    says otherwise — taken in by the loader's real `_take_in`."""
    r.kvn.cq.extend((s.ud, s.size if res is None else res) for s in subs)
    r.head._take_in(r.head._plans)


def _items(r, layer=0):
    return [(it.kind, it.rank0, it.n) for it in r.arr._q[layer]]


# ---- the plan as it goes out -------------------------------------------

def test_a_layers_segments_go_out_in_load_order(rig):
    """Weight, then the whole ACT block region, then K and V alternating per
    segment — the layer's load order, which is also the file's within the
    layer. Never the whole K block region before the whole V block region: a
    KV block arrives only once both its K and V have been loaded."""
    r = rig(alpha=0.5, per_seq=150)
    _fill(r)
    kinds = [_block_region(r, s) for s in r.kvn.subs]
    collapsed = [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k]
    assert collapsed == ["W", "A", "K", "V", "K", "V"]
    w = _of(r, "W")
    assert [(s.off, s.size, s.fidx) for s in w] == [
        (0, MiB, W_FIDX), (MiB, MiB, W_FIDX), (2 * MiB, MiB // 2, W_FIDX)]


def test_kv_reads_land_at_the_closed_form_offsets_and_buffer_blocks(rig):
    """A K,V pair is two numbers, (rank0, n): its file bytes are
    `k_base + rank0·kb` (V one `v_delta` further on) and they land in buffer
    blocks `mem_kv_base + rank0 ..` of the layer's K and V caches — one
    contiguous range each, no per-block table."""
    r = rig(alpha=0.5, per_seq=150)
    _fill(r)
    _a, k_base, v_delta = r.layout.region_bases(0)
    mkv = r.layout.mem_kv_base(0)
    kc, vc = r.caches[0]
    n_kv = 2 * 150 - 150                   # K = N_seq·c blocks, ⌈K·α⌉ are ACT
    pairs = [(0, PAIR), (PAIR, n_kv - PAIR)]
    assert len(_of(r, "K")) == len(_of(r, "V")) == len(pairs)
    for (r0, n), k, v in zip(pairs, _of(r, "K"), _of(r, "V")):
        assert (k.off, k.size) == (k_base + r0 * KB, n * KB)
        assert (v.off, v.size) == (k_base + r0 * KB + v_delta, n * KB)
        assert k.addr == kc.data_ptr() + (mkv + r0) * KB
        assert v.addr == vc.data_ptr() + (mkv + r0) * KB


def test_what_the_fifo_cannot_hold_goes_out_as_completions_come_back(rig):
    """The NVMe engine holds `capacity` segments outstanding. The fill hands
    over what fits, in plan order, and each completion taken makes room for
    the next — the NVMe engine sees the same stream, layer after layer, that
    it would with room for all of it."""
    whole = rig(alpha=0.5, per_seq=80, n_layers=2, n_open=2)
    _fill(whole)
    want = [(s.off, s.fidx) for s in whole.kvn.subs]
    r = rig(alpha=0.5, per_seq=80, n_layers=2, n_open=2, fifo=3)
    _fill(r)
    landed = 0
    while landed < len(r.kvn.subs):
        assert len(r.kvn.subs) - landed <= 3
        _land(r, r.kvn.subs[landed])
        landed += 1
        _fill(r)
    assert [(s.off, s.fidx) for s in r.kvn.subs] == want
    assert not r.head._plans


# ---- the step each loader pass belongs to --------------------------------

@pytest.mark.parametrize("gen", [1, 2, 5])
def test_a_batch_is_its_prefill_step_then_decode_steps_1_to_gen_minus_1(
        rig, gen):
    """The prefill step samples the batch's first token, so the decode steps
    that follow it are t = 1 .. gen−1 — none at all for gen 1 — and the step
    after the last one is the next batch's prefill, whose own declaration
    the loader reads on leaving it."""
    r = rig()
    r.layout.declare_batch(40 * BS, gen=gen)
    assert r.head._step is None, "the head starts on a prefill sweep"
    steps = []
    for _ in range(gen):
        r.head._next_step()
        steps.append(r.head._step)
    assert steps == list(range(1, gen)) + [None]
    r.layout.declare_batch(40 * BS, gen=3)
    r.head._next_step()
    assert (r.head._step, r.head._gen) == (1, 3)


def test_a_gen_1_batch_is_not_given_the_next_batchs_declaration(rig):
    """A gen-1 batch does not hold its forward thread until the loader has
    left its prefill step, so the next batch's prefill step can declare
    first. Each batch is still the one it declared: the loader takes the
    declarations in order, one per batch."""
    r = rig()
    r.layout.declare_batch(40 * BS, gen=1)
    r.layout.declare_batch(20 * BS, gen=16)
    r.head._next_step()
    assert (r.head._step, r.head._gen, r.head._prompt_len) == (
        None, 1, 40 * BS)
    r.head._next_step()
    assert (r.head._step, r.head._gen, r.head._prompt_len) == (
        1, 16, 20 * BS)


@pytest.mark.parametrize("S", [40 * BS - 1, 40 * BS])
def test_the_first_decode_step_reloads_every_block_up_to_the_new_token(rig, S):
    """Through real visits: the prefill step (visits 0 and 1 of a two-layer
    model) reads weights only; the step entered after it is decode step 1,
    whose new token sits at position S, so it reloads ⌈(S+1)/bs⌉ blocks of
    every sequence — the whole prompt and the current block (the mutable
    block that receives the new token)."""
    r = rig(n_layers=2, n_open=3)
    r.layout.declare_batch(S, gen=3)
    _fill(r)
    assert [p["layer"] for p in r.head._plans] == [0, 1]
    assert {s.fidx for s in r.kvn.subs} == {W_FIDX}
    _land(r, *[s for s in r.kvn.subs if s.off < W_BYTES])
    _fill(r)
    assert r.head._step == 1
    p = r.head._plans[-1]
    c = -(-(S + 1) // BS)
    assert p["layer"] == 0 and not p["sts"]     # α = 0: no ACT block region
    k = [s for s in r.kvn.subs if s.fidx != W_FIDX and _block_region(r, s) == "K"]
    assert sum(s.size for s in k) == N_SEQ * c * KB


# ---- weight ---------------------------------------------------------------

def test_the_layer_is_ready_only_when_its_last_weight_segment_lands(rig):
    """The weight is counted, not ordered: segments land in any order, and
    the one that brings the count to zero publishes the layer — never
    before, since the forward pass would read a slot the drive is still
    writing."""
    r = rig()
    _fill(r)
    w = _of(r, "W")
    plan = r.head._plans[0]
    for s in (w[2], w[0]):
        r.head._reap_segment(s.ud, s.size)
        assert r.pool.ready == [] and plan["ready"] == [False]
    r.head._reap_segment(w[1].ud, w[1].size)
    assert r.pool.ready == [0] and plan["ready"] == [True]
    assert ReadRouter._plan_complete(plan), (
        "nothing on disk yet: the weight is the whole plan")


@pytest.mark.parametrize("how", ["failed", "short"])
def test_a_failed_or_short_weight_read_raises(rig, how):
    """A segment that did not land in full is not counted, so the rest
    landing cannot publish the layer either."""
    r = rig()
    _fill(r)
    w = _of(r, "W")
    res = -errno.EBADF if how == "failed" else w[2].size - 4096
    with pytest.raises(OSError) as ei:
        r.head._reap_segment(w[2].ud, res)
    assert ei.value.errno == (errno.EBADF if how == "failed" else errno.EIO)
    r.head._reap_segment(w[0].ud, w[0].size)
    r.head._reap_segment(w[1].ud, w[1].size)
    assert r.pool.ready == []


# ---- K,V --------------------------------------------------------------------

def test_a_kv_pair_is_handed_over_only_when_both_halves_have_landed(rig):
    """The pair's rank run goes to the arrival feed after its second half's
    completion is taken — whichever half that is — and pairs go out in the
    order they complete, not the order they were submitted. A KV item
    carries no event: the drive made those bytes, and taking the CQE first
    is the whole ordering."""
    r = rig(per_seq=80)                      # α = 0: 160 KV blocks, two pairs
    _fill(r)
    (k0, k1), (v0, v1) = _of(r, "K"), _of(r, "V")
    r.head._reap_segment(v1.ud, v1.size)
    r.head._reap_segment(k0.ud, k0.size)
    assert _items(r) == []
    r.head._reap_segment(k1.ud, k1.size)
    assert _items(r) == [(1, PAIR, 160 - PAIR)]
    r.head._reap_segment(v0.ud, v0.size)
    assert _items(r) == [(1, PAIR, 160 - PAIR), (1, 0, PAIR)]
    assert all(it.event is None for it in r.arr._q[0])


# ---- kv_gen launches -------------------------------------------------------

def test_a_kv_gen_launch_goes_out_once_its_reads_and_the_weight_are_in(rig):
    """80 activation blocks at 16 blocks a read and 32 a kv_gen launch:
    launches [0,32) and [32,64) of two reads each, then [64,80). A launch
    waits for all its reads, for the layer's weight (kv_gen slices W_k/W_v
    out of the weight buffer), and for the launch before it."""
    r = rig(alpha=1.0, per_seq=40, run_cap=16, launch_blocks=32)
    _fill(r)
    a = _of(r, "A")
    assert [s.size // KB for s in a] == [16, 16, 16, 16, 16]
    _land(r, a[2], a[3], a[0])
    assert _items(r) == [], "wave 1 is in, but wave 0 is not"
    _land(r, a[1])
    assert _items(r) == [], "waves 0 and 1 are in, but the weight is not"
    _land(r, *_of(r, "W"))
    assert r.pool.ready == [0]
    assert _items(r) == [(0, 0, 32), (0, 32, 32)]
    assert r.arr._open[0], "a wave is still out"
    _land(r, a[4])
    assert _items(r) == [(0, 0, 32), (0, 32, 32), (0, 64, 16)]
    assert not r.arr._open[0] and not r.head._plans


@pytest.mark.parametrize("how", ["failed", "short"])
@pytest.mark.parametrize("block_region", ["A", "K"])
def test_a_failed_or_short_reload_read_raises(rig, block_region, how):
    """An ACT read that did not land would feed kv_gen stale activation
    buffer bytes, and a K half that did not land would declare blocks
    arrived whose K was never loaded — so neither is credited: the
    kv_gen launch keeps its count and the pair pushes nothing."""
    r = rig(alpha=0.5, per_seq=40)
    _fill(r)
    s = _of(r, block_region)[0]
    res = -errno.EBADF if how == "failed" else s.size - 4096
    with pytest.raises(OSError) as ei:
        r.head._reap_segment(s.ud, res)
    assert ei.value.errno == (errno.EBADF if how == "failed" else errno.EIO)
    plan = r.head._plans[0]
    if block_region == "A":
        st0 = plan["sts"][0]
        assert st0["remaining"] == sum(
            1 for d in plan["segments"] if d[1] == 0 and st0 in d[3])
    else:
        v = _of(r, "V")[0]
        r.head._reap_segment(v.ud, v.size)
        assert _items(r) == []


# ---- retirement -------------------------------------------------------------

def test_the_layer_feed_closes_once_every_read_has_landed(rig):
    """The feed's close is what ends the forward pass's partial attention
    calls for the layer, so it closes after the layer's last item and not
    before; the retired plan's tag is dropped."""
    r = rig(per_seq=80)
    _fill(r)
    subs = list(r.kvn.subs)
    _land(r, *subs[:-1])
    assert r.arr._open[0] and len(r.head._plans) == 1
    _land(r, subs[-1])
    assert not r.arr._open[0]
    assert not r.head._plans and not r.head._plan_by_pid
    assert _items(r) == [(1, 0, PAIR), (1, PAIR, 160 - PAIR)]


def test_plans_retire_in_layer_order(rig):
    """Layer 1's reads landing first publish layer 1's weight, but its feed
    stays open until layer 0's plan has retired: plans leave the queue from
    the front."""
    r = rig(n_layers=2, n_open=2)
    _fill(r)
    assert [p["layer"] for p in r.head._plans] == [0, 1]
    closed = []
    _close = r.arr.close
    r.arr.close = lambda l: (closed.append(l), _close(l))[1]
    w1 = [s for s in r.kvn.subs if s.off >= W_BYTES]
    w0 = [s for s in r.kvn.subs if s.off < W_BYTES]
    _land(r, *w1)
    assert r.pool.ready == [1]
    assert closed == [] and len(r.head._plans) == 2
    _land(r, *w0)
    assert r.pool.ready == [1, 0]
    assert closed == [0, 1] and not r.head._plans


# ---- the dispatch and completion rules on bare plans ----------------------

_BARE = object.__new__(ReadRouter)


def _launches(layer, landed, ready, log):
    """A plan carrying only what `_dispatch_ready` reads."""
    sts = [{"remaining": 0 if ok else 1, "i": i}
           for i, ok in enumerate(landed)]
    return {"layer": layer, "sts": sts, "di": 0, "ready": [ready],
            "dispatch": lambda st: log.append((layer, st["i"]))}


def test_a_kv_gen_launch_waits_for_its_layers_weight_and_the_one_before_it():
    """All of a kv_gen launch's reads in is necessary, not sufficient: the
    layer's weight must be in too, and the launch before it must have been
    issued."""
    log = []
    p = _launches(0, [True, True], False, log)
    _BARE._dispatch_ready([p])
    assert log == [], "the reproject reads W_k/W_v out of the weight slot"
    p = _launches(0, [False, True], True, log)
    _BARE._dispatch_ready([p])
    assert log == [], "waves launch in wave order"
    p["sts"][0]["remaining"] = 0
    _BARE._dispatch_ready([p])
    assert log == [(0, 0), (0, 1)]


def test_a_later_layer_never_jumps_an_earlier_layers_kv_gen_launch():
    """The kv_gen stream and the arrival feed see the layers in sequence: a
    later layer whose kv_gen launches are all in waits behind an earlier
    layer's launch that is not."""
    log = []
    p0 = _launches(0, [True, False], True, log)
    p1 = _launches(1, [True], True, log)
    _BARE._dispatch_ready([p0, p1])
    assert log == [(0, 0)]
    p0["sts"][1]["remaining"] = 0
    _BARE._dispatch_ready([p0, p1])
    assert log == [(0, 0), (0, 1), (1, 0)]
    _BARE._dispatch_ready([p0, p1])
    assert log == [(0, 0), (0, 1), (1, 0)], "each wave launches once"


def test_a_layer_without_kv_gen_launches_does_not_hold_the_next():
    """Only an undispatched kv_gen launch holds the layers behind it — a
    layer with no ACT block region, weight in or not, has nothing to hold
    them with."""
    log = []
    _BARE._dispatch_ready([_launches(0, [], False, log),
                           _launches(1, [True], True, log)])
    assert log == [(1, 0)]


def test_a_plan_is_complete_only_when_every_read_and_kv_gen_launch_is_done():
    """Every read in is not enough: a kv_gen launch not yet issued still has
    an item to push, and closing the feed first would lose it."""
    plan = {"segments": [None] * 3, "done": 3, "sts": [{}], "di": 0}
    assert not ReadRouter._plan_complete(plan)
    plan["di"] = 1
    assert ReadRouter._plan_complete(plan)
    plan["done"] = 2
    assert not ReadRouter._plan_complete(plan)
