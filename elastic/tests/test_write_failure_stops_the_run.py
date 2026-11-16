"""A save that did not land must stop the run.

The reload reads back every block the step's count says is on disk — there
is no registry that could skip an unwritten one — so a write that fails has
to stop the run before that block is served. In the writeback that is one
rule: whatever goes wrong inside a turn raises out of `_turn` and ends the
writeback's loop (`_feed_main`) with it. The slot the failed write was
reading from is never released, so the visit n_slots later is never acquired.

These tests drive the writeback's real methods (`_feed_main`, `_turn`,
`_reap`, `_send`, `_release`, `_submit`, `_complete`) on a light harness: a
scripted NVMe engine port and a pool whose slots are bookkeeping only.
"""

import errno
import os
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from elastic.vllm_runtime.layer_pool import LayerSlot, LayerSlotPool
from elastic.vllm_runtime.nvme_write_router import WriteRouter


class _FakeRing:
    """Just enough NVMe engine write port. `submit_writev` records the
    request and posts its completion fd (the NVMe engine would, once the
    write lands); `try_wait_one` replays one scripted result per submitted
    request, in submission order (None = full length). `refuse` maps a
    submit call's 1-based index to the errno it is refused with, once."""

    def __init__(self, results, refuse=None):
        self.results = list(results)
        self.refuse = dict(refuse or {})
        self.submitted = []            # (off, nbytes, ud), every accepted one
        self._pending = []
        self.n_calls = 0
        self.drains = 0
        self.cq_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    def submit_writev(self, off, bufs, user_data=0):
        self.n_calls += 1
        e = self.refuse.pop(self.n_calls, None)
        if e is not None:
            raise OSError(e, os.strerror(e))
        rec = (off, sum(b.nbytes for b in bufs), user_data)
        self.submitted.append(rec)
        self._pending.append(rec)
        os.eventfd_write(self.cq_fd, 1)
        return object()                      # keepalive handle

    def drain(self) -> None:
        self.drains += 1                     # admit what is queued

    def try_wait_one(self):
        if not self._pending or not self.results:
            return None
        _off, want, ud = self._pending.pop(0)
        res = self.results.pop(0)
        return (ud, want if res is None else res)

    def completion_fd(self) -> int:
        return self.cq_fd


class _Harness:
    """The writeback without its thread or a device. Its layout is the one
    number the accounting reads: a block is one of the fixture's 4 KiB
    sources."""
    _feed_main = WriteRouter._feed_main
    _turn = WriteRouter._turn
    _reap = WriteRouter._reap
    _send = WriteRouter._send
    _release = WriteRouter._release
    _submit = WriteRouter._submit
    _account_built = WriteRouter._account_built
    _complete = WriteRouter._complete
    enqueue_writes = WriteRouter.enqueue_writes
    reserve = WriteRouter.reserve

    def __init__(self, results, pool=None, refuse=None):
        self.ring = _FakeRing(results, refuse)
        self.layout = SimpleNamespace(block_bytes=4096)
        self._pool = pool if pool is not None else _pool(2, 4)
        self._inbox = deque()
        self._fifo = deque()
        self._sub = 0
        self._inflight = {}
        self._next_ud = 1
        self._next_did = 0
        self._audit_on = False
        self._timing = True
        self._pub_cnt = np.zeros(1, dtype=np.int64)
        self._pub_ticket = 0
        self._sig_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._stop_ticket = None
        self._wait_n = 0
        self.n_kv_saves = self.n_act_saves = self.n_items = 0
        self.n_stop_end = self.n_stop_save = 0
        for _f in ("st_e2s", "st_e2ev", "st_ev2s", "st_s2c"):
            setattr(self, _f, 0.0)
        for _f in ("st_ndesc", "st_bytes", "st_src_n", "st_src_bad",
                   "st_ph_n", "st_ph_bad"):
            setattr(self, _f, 0)
        self.st_ph_by = {}
        self.st_src_off = {}

    def register(self, entries, kind="act_batch", layer=3):
        """reserve → enqueue_writes, as a producer does (the mark is the
        test's `passes`). Returns the descriptor it registered."""
        t = self.reserve()
        self.enqueue_writes(layer, kind, entries, gate_seq=t)
        return self._inbox[-1]

    def passes(self, ticket):
        """What a mark's callback does when the GPU gets there."""
        self._pub_cnt[0] = max(int(self._pub_cnt[0]), int(ticket))


def _pool(n_slots: int, n_layers: int) -> LayerSlotPool:
    """The slot pool's bookkeeping only — the real constructor registers a slab
    per slot, and no byte of a slot is touched here."""
    p = object.__new__(LayerSlotPool)
    p._slots = [LayerSlot(idx=i, raw_buf=None, t_buf=None)
                for i in range(n_slots)]
    p.n_slots = n_slots
    p._n_layers = n_layers
    p._head_visit = p._tail_visit = 0
    p._bind = list(range(n_layers))
    p._on_bind = lambda layer, slot: None
    p._cv = threading.Condition()
    p.n_acquired = p.n_use = p.n_released = p.n_moved = 0
    p.t_use_wait = 0.0
    p.release_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    return p


def _heard(fd) -> bool:
    try:
        return os.read(fd, 8) != b""
    except BlockingIOError:
        return False


@pytest.fixture
def bufs():
    """Four 4 KiB write sources, 512-aligned like the O_DIRECT grain."""
    raw = np.zeros(4 * 4096 + 512, dtype=np.uint8)
    a0 = (-raw.ctypes.data) % 512
    return [raw[a0 + i * 4096:a0 + (i + 1) * 4096] for i in range(4)]


# ---- a write that did not land ---------------------------------------------

def test_a_hard_write_error_raises_out_of_the_turn(bufs):
    """No bytes landed (-EIO): the next turn that reaps it raises, so the
    reload never serves the file's previous content as saved state."""
    h = _Harness(results=[-5])
    d = h.register([(0, [bufs[0]])])
    h.passes(d.gate_seq)
    assert h._turn() is False                      # sent
    assert len(h.ring.submitted) == 1
    with pytest.raises(RuntimeError, match="write failed"):
        h._turn()                                  # reaped: -EIO
    assert not d.completed


def test_a_short_write_raises(bufs):
    """The NVMe engine owns short-write remainders, so a short count
    reaching the writeback is an EOF the bounds checks make unreachable — it
    raises, it is not counted and let go."""
    h = _Harness(results=[512])                    # 512 of 4096
    d = h.register([(0, [bufs[0]])])
    h.passes(d.gate_seq)
    h._turn()
    with pytest.raises(RuntimeError, match="short write"):
        h._turn()
    assert not d.completed


def test_a_full_write_completes_and_leaves_the_fifo(bufs):
    """The control: without it the two tests above would pass on a
    writeback that fails everything. A full-length completion completes the
    descriptor, and the next release drops it off the FIFO front."""
    h = _Harness(results=[None])                   # None == full length
    d = h.register([(0, [bufs[0]])])
    h.passes(d.gate_seq)
    h._turn()
    assert not d.completed and h._sub == 1
    h._turn()
    assert d.completed
    assert not h._fifo and h._sub == 0 and not h._inflight
    assert h.n_items == 1 and h.n_act_saves == 1
    assert h.st_ndesc == 1 and h.st_bytes == 4096


def test_a_refused_submit_raises(bufs):
    """An NVMe engine refusal other than a full FIFO is a request the blob will
    not hold: it raises out of the turn instead of being skipped."""
    h = _Harness(results=[None, None], refuse={2: errno.EINVAL})
    d = h.register([(0, [bufs[0]]), (4096, [bufs[1]])])
    h.passes(d.gate_seq)
    with pytest.raises(RuntimeError, match="write submit failed"):
        h._turn()
    assert not d.completed


# ---- the licence and the full FIFO ------------------------------------------

def test_a_descriptor_waits_for_its_own_ticket(bufs):
    """Its licence is its own ticket and nothing else: a counter that has
    passed some earlier point (the previous visit's layer end, say) does not
    let it through."""
    h = _Harness(results=[None])
    h.reserve()
    h.reserve()                                     # earlier points: 1, 2
    d = h.register([(0, [bufs[0]])], kind="kv_batch")
    assert d.gate_seq == 3
    h.passes(2)                                     # an earlier point
    h._turn()
    assert h.ring.submitted == [] and h._sub == 0 and list(h._fifo) == [d]
    h.passes(3)                                     # its own point
    h._turn()
    h._turn()
    assert d.completed and h.n_kv_saves == 1 and not h._fifo


def test_a_full_engine_fifo_leaves_the_descriptor_resumable(bufs):
    """-EAGAIN is not a failure: the descriptor keeps its place, what it
    already handed over is not handed over again, a descriptor behind it
    does not jump ahead, and the turn a landed write wakes finishes it. It
    completes only once every one of its requests has landed — the first
    landing is not the descriptor's."""
    h = _Harness(results=[None] * 4, refuse={2: errno.EAGAIN})
    d = h.register([(0, [bufs[0]]), (4096, [bufs[1]]), (8192, [bufs[2]])])
    d2 = h.register([(12288, [bufs[3]])])
    h.passes(d2.gate_seq)
    h._turn()                                       # #1 in, #2 refused
    assert [s[0] for s in h.ring.submitted] == [0]
    assert d.next_w == 1 and h._sub == 0 and h.ring.drains == 1
    h._turn()                                       # #1 lands; resume
    assert not d.completed
    assert [s[0] for s in h.ring.submitted] == [0, 4096, 8192, 12288]
    h._turn()                                       # the rest land
    assert d.completed and d2.completed and not h._fifo
    assert d.n_items == 3 and h.n_items == 4


# ---- the slot is released only on a landed write ----------------------------

def _visit_with_a_write(p, h, bufs):
    """Visit 0 of layer 0: the loader acquires and publishes it, the forward
    pass registers the layer's KV write, stamps the layer end and marks
    both."""
    s = p.acquire(0, 0)
    p.mark_ready(s)
    d = h.register([(0, [bufs[0]])], kind="kv_batch", layer=0)
    t_end = h.reserve()
    p.stamp_forward_done(s, t_end)
    return s, d, t_end


def test_a_landed_write_releases_the_slot(bufs):
    """The control for the next test: the visit's layer end has passed and
    its write landed, so the writeback releases it — the loader is woken and
    can acquire the slot for the visit n_slots later."""
    p = _pool(2, 4)
    h = _Harness(results=[None], pool=p)
    s, d, t_end = _visit_with_a_write(p, h, bufs)
    assert p.acquire(1, 1) is not None
    assert p.acquire(2, 2) is None                  # every slot live
    h._turn()                                       # layer end not heard yet
    assert h.ring.submitted == [] and h.n_stop_end == 1
    h.passes(t_end)
    h._turn()                                       # sent, still landing
    assert p.n_released == 0 and h.n_stop_save == 1
    assert not _heard(p.release_fd)
    h._turn()                                       # landed
    assert d.completed and p.n_released == 1 and s.released_visit == 0
    assert _heard(p.release_fd)
    assert p.acquire(2, 2) is s


def test_a_failed_write_ends_the_writeback_and_never_releases_the_slot(bufs):
    """The raise ends the writeback's loop: the slot whose write failed is
    never released, so the loader never acquires it for the visit n_slots
    later."""
    p = _pool(2, 4)
    h = _Harness(results=[-5], pool=p)
    s, d, t_end = _visit_with_a_write(p, h, bufs)
    assert p.acquire(1, 1) is not None
    h.passes(t_end)
    os.eventfd_write(h._sig_fd, 1)                  # the mark's fd write
    with pytest.raises(RuntimeError, match="write failed"):
        h._feed_main()
    assert p.n_released == 0 and s.released_visit == -1
    assert p.acquire(2, 2) is None
