"""The writeback releases a slot's visit on one condition, read by the
writeback (`WriteRouter._release`) and nowhere else:

    1 the forward pass is done with this visit: the slot's
      `forward_done_visit` is this visit (the forward pass stamps `done_seq`
      first, then `forward_done_visit`)
    2 its layer end passed: the compute stream's progress counter is at or
      past the slot's `done_seq` (raised by the mark's stream callback, which
      is also what wakes the writeback)
    3 every write licensed up to that layer end has landed: the oldest write
      still in the FIFO carries a later ticket, or there is none

Term 3 follows from the save being copy-free: it DMA-reads its source straight
out of this slot's K and V buffers and its activation buffer, so while a write
is outstanding the drive is reading those bytes.

Two properties of the shape:

  * the loader does not evaluate it. The loader's acquisition reads one fact the
    writeback wrote (`released_visit`), so the thread that owns NVMe reads
    never waits on a write or asks about one; a "not yet" is the
    writeback's, and it sleeps until a compute point passes or a write lands.
  * it is all-or-nothing and in visit order. The slot is held whole until its
    writes land, and a later visit finishing first moves nothing.

The pool here is the real class built with `object.__new__` over real
`LayerSlot`s with no buffers; the writeback is the real `WriteRouter` methods
bound onto a harness holding only the state they read.
"""

import os
import threading
import weakref
from collections import deque

import numpy as np
import pytest

from elastic.vllm_runtime.layer_pool import LayerSlot, LayerSlotPool
from elastic.vllm_runtime.nvme_write_router import WriteRouter, _Desc

_ALL = 1 << 40          # a progress counter past every ticket in a test


def _pool(n: int = 2, n_layers: int = 8) -> LayerSlotPool:
    """Only the bookkeeping, with the attributes `__init__` sets. The real
    constructor cudaHostRegisters a slab per slot, which bookkeeping does not
    need."""
    p = object.__new__(LayerSlotPool)
    p._slots = [LayerSlot(idx=i, raw_buf=None, t_buf=None)
                for i in range(n)]
    p.n_slots = n
    p._n_layers = n_layers
    p.n_acquired = 0
    p.t_use_wait = 0.0
    p.n_use = 0
    p._cv = threading.Condition()
    p._head_visit = 0
    p._tail_visit = 0
    p._bind = list(range(n_layers))
    p._on_bind = lambda layer, slot: None
    p.n_released = 0
    p.n_moved = 0
    p.release_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    weakref.finalize(p, os.close, p.release_fd)
    return p


class _Ring:
    """The NVMe engine's write port as the writeback's reap sees it:
    completions, scripted."""

    def __init__(self):
        self.cqes = deque()

    def try_wait_one(self):
        return self.cqes.popleft() if self.cqes else None


class _Tail:
    """The writeback's state and its real methods — no thread, no NVMe
    engine."""
    _release = WriteRouter._release
    _turn = WriteRouter._turn
    _reap = WriteRouter._reap
    _send = WriteRouter._send
    _complete = WriteRouter._complete

    def __init__(self, pool):
        self._pool = pool
        self._inbox = deque()
        self._fifo = deque()
        self._sub = 0
        self._inflight = {}
        self.ring = _Ring()
        self._pub_cnt = np.zeros(1, dtype=np.int64)
        self._stop_ticket = None
        self.n_stop_end = 0
        self.n_stop_save = 0
        self.st_s2c = 0.0
        self.st_bytes = 0
        self.st_ndesc = 0

    def run(self, passed: int = _ALL) -> int:
        """One `_release` with the progress counter at `passed`; returns how
        many visits it released."""
        n0 = self._pool.n_released
        self._release(passed)
        return self._pool.n_released - n0


def _desc(gate_seq: int, completed: bool = False) -> _Desc:
    d = _Desc(kind="kv_batch", layer=0, gate_seq=gate_seq)
    d.completed = completed
    return d


def _in_flight(tail: _Tail, d: _Desc, ud: int, nbytes: int = 4096) -> None:
    """`d` handed to the NVMe engine as one SQE (what `_submit` leaves
    behind)."""
    buf = np.zeros(nbytes, dtype=np.uint8)
    d.writes = [(0, [buf])]
    d.next_w = 1
    d.n_items = 1
    tail._inflight[ud] = (d, 0, nbytes)
    tail._fifo.append(d)
    tail._sub = len(tail._fifo)


def _held(p, visit: int, layer: int, done_seq: int):
    """The loader acquires the visit, the forward pass runs it and stamps its
    layer end."""
    s = p.acquire(visit, layer)
    assert s is not None
    p.mark_ready(s)
    p.stamp_forward_done(s, done_seq)
    return s


# ---- the condition, all three terms in one place -------------------------

def test_all_three_terms_and_which_one_said_no():
    """Each term alone holds the visit, and the stop is counted against the
    term that made it: the layer end (terms 1 and 2) or a save (term 3)."""
    p = _pool()
    tail = _Tail(p)
    s = p.acquire(0, 0)

    assert tail.run() == 0               # 1: the forward pass still holds it
    assert (tail.n_stop_end, tail.n_stop_save) == (1, 0)

    p.mark_ready(s)
    p.stamp_forward_done(s, 5)
    assert tail.run(passed=4) == 0       # 2: its layer end has not passed
    assert (tail.n_stop_end, tail.n_stop_save) == (2, 0)

    d = _desc(gate_seq=5)                # 3: a save still reads the slot
    tail._fifo.append(d)
    tail._sub = 1
    assert tail.run(passed=5) == 0
    assert (tail.n_stop_end, tail.n_stop_save) == (2, 1)

    d.completed = True
    assert tail.run(passed=5) == 1
    assert s.released_visit == 0 and p.acquire(2, 2) is s


def test_the_layer_end_must_be_this_visits_own_stamp():
    """Term 1 is what keeps term 2 honest: the slot still carries its previous
    tenancy's `done_seq`, long passed. Until the forward pass stamps this
    visit's `forward_done_visit`, that old ticket must not read as this
    visit's layer end."""
    p = _pool()
    tail = _Tail(p)
    _held(p, 0, 0, done_seq=1)
    _held(p, 1, 1, done_seq=2)
    assert tail.run() == 2
    s = p.acquire(2, 2)                  # slot 0 again: done_seq 1, passed
    p.mark_ready(s)
    assert s.done_seq == 1 and s.forward_done_visit == 0
    assert tail.run() == 0 and tail.n_stop_end == 1
    p.stamp_forward_done(s, 3)
    assert tail.run() == 1


def test_a_write_licensed_up_to_the_layer_end_holds_the_visit():
    """Any write whose ticket is at or before the layer end is this visit's
    (or an older one's) and may be reading the slot — whether the NVMe
    engine has taken it yet or not (a full NVMe engine FIFO leaves it
    unsubmitted)."""
    for gate, submitted in ((3, True), (3, False), (1, True)):
        p = _pool()
        tail = _Tail(p)
        _held(p, 0, 0, done_seq=3)
        tail._fifo.append(_desc(gate_seq=gate))
        tail._sub = 1 if submitted else 0
        assert tail.run() == 0, (gate, submitted)
        assert tail.n_stop_save == 1


def test_a_later_ticket_write_does_not_hold_the_visit():
    """A write licensed after the layer end belongs to a later visit: still
    landing, it holds that visit and not this one."""
    p = _pool()
    tail = _Tail(p)
    _held(p, 0, 0, done_seq=3)
    _held(p, 1, 1, done_seq=6)
    later = _desc(gate_seq=4)            # visit 1's, still landing
    tail._fifo.append(later)
    tail._sub = 1
    assert tail.run() == 1               # visit 0 goes ...
    assert p._slots[0].released_visit == 0
    assert tail.n_stop_save == 1         # ... visit 1 stops on its own write
    later.completed = True
    assert tail.run() == 1


def test_landed_writes_leave_the_fifo_from_the_front_only():
    """The FIFO is in ticket order and the release reads only its front, so
    every landed descriptor at the front is dropped (and the submitted-prefix
    count with it); one that landed behind a write still landing stays until
    the front clears."""
    p = _pool()
    tail = _Tail(p)
    a, b, c, d = (_desc(1, True), _desc(2, True), _desc(3), _desc(4, True))
    tail._fifo.extend([a, b, c, d])
    tail._sub = 4
    tail.run()
    assert list(tail._fifo) == [c, d] and tail._sub == 2
    c.completed = True
    tail.run()
    assert not tail._fifo and tail._sub == 0


def test_release_is_in_visit_order_only():
    """A later visit that is finished in every term does not pass an earlier
    one that is not: the writeback stands at the oldest visit and steps only
    when it is released, and then releases every finished visit behind it in
    order."""
    p = _pool(n=3)
    tail = _Tail(p)
    _held(p, 0, 0, done_seq=2)
    _held(p, 1, 1, done_seq=3)
    s2 = _held(p, 2, 2, done_seq=4)
    tail._fifo.append(_desc(gate_seq=1))  # visit 0's write, still landing
    tail._sub = 1
    assert tail.run() == 0
    assert all(s.released_visit == -1 for s in p._slots)
    tail._fifo[0].completed = True
    assert tail.run() == 3
    assert [s.released_visit for s in p._slots] == [0, 1, 2]
    assert p._tail_visit == 3 and s2.released_visit == 2


# ---- the loader's side of it --------------------------------------------

def test_first_tenant_of_a_slot_has_nothing_to_wait_for():
    """A slot nobody has held counts as released: its `released_visit` and
    `acquired_visit` both start at -1, so the first round needs no
    writeback."""
    p = _pool(n=3)
    for v in range(3):
        s = p.acquire(v, v)
        assert s is not None and s.acquired_visit == v
    assert p.live_visits() == 3


def test_a_refused_acquire_changes_nothing():
    """A refusal is "not yet", not a reservation: the slot, the loader's
    visit pointer, the layer binding and the bind hook are exactly as they
    were."""
    p = _pool()
    bound = []
    p.set_bind_hook(lambda layer, slot: bound.append((layer, slot)))
    for v in range(2):
        assert p.acquire(v, v) is not None
    before = ([vars(s).copy() for s in p._slots], p._head_visit, p.n_acquired,
              list(p._bind), list(bound))
    assert p.acquire(2, 2) is None
    after = ([vars(s).copy() for s in p._slots], p._head_visit, p.n_acquired,
             list(p._bind), list(bound))
    assert after == before


def test_the_bind_hook_hears_an_acquisition_before_any_byte_is_loaded():
    """The K and V buffers belong to the same slot, so their alias is
    re-pointed at the acquisition itself — with the slot the rotation gave the
    visit, not `layer % n_slots`."""
    L, n_slots = 5, 3
    p = _pool(n=n_slots, n_layers=L)
    tail = _Tail(p)
    bound = []
    p.set_bind_hook(lambda layer, slot: bound.append((layer, slot)))
    for v in range(2 * L):
        assert p.acquire(v, v % L) is not None
        assert bound[-1] == (v % L, v % n_slots)
        s = p._slots[v % n_slots]
        p.mark_ready(s)
        p.stamp_forward_done(s, v + 1)
        assert tail.run() == 1
    assert len(bound) == 2 * L


# ---- the writeback's turn around it -------------------------------------

def test_a_landed_write_releases_its_slot_in_the_same_turn():
    """The landing is what wakes the writeback, and the turn it wakes reaps
    the completion before it looks at the slots, so the slot is released on
    that wake and not on some later one."""
    p = _pool()
    tail = _Tail(p)
    _held(p, 0, 0, done_seq=1)
    d = _desc(gate_seq=1)
    _in_flight(tail, d, ud=7)
    tail._pub_cnt[0] = 1                  # the layer end has passed
    assert tail._turn() is False
    assert p._slots[0].released_visit == -1   # its write has not landed
    tail.ring.cqes.append((7, 4096))
    tail._turn()
    assert d.completed and not tail._fifo
    assert p._slots[0].released_visit == 0


def test_a_failed_write_never_releases_its_slot():
    """A write that did not land raises on the writeback before the release
    is looked at, so the slot the save was reading from is never handed
    on."""
    p = _pool()
    tail = _Tail(p)
    _held(p, 0, 0, done_seq=1)
    _in_flight(tail, _desc(gate_seq=1), ud=7)
    tail._pub_cnt[0] = 1
    tail.ring.cqes.append((7, -5))       # -EIO
    with pytest.raises(RuntimeError, match="write failed"):
        tail._turn()
    assert p._slots[0].released_visit == -1


def test_a_short_write_never_releases_its_slot():
    """Some bytes landed and some did not: the block on disk is neither the
    old nor the new one, so it is a failure like any other."""
    p = _pool()
    tail = _Tail(p)
    _held(p, 0, 0, done_seq=1)
    _in_flight(tail, _desc(gate_seq=1), ud=7)
    tail._pub_cnt[0] = 1
    tail.ring.cqes.append((7, 512))
    with pytest.raises(RuntimeError, match="short write"):
        tail._turn()
    assert p._slots[0].released_visit == -1
