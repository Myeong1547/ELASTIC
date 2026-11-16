"""The slot rotation is walked by two pointers.

The loader's visit pointer (`acquire`) moves when the loader acquires a visit's
slot; the writeback's moves from behind, over the oldest tenancy, once its
layer end has passed and its writes have landed — `WriteRouter._release`,
which steps the pool with `release_tail` and then says so with
`notify_released`. A tenancy in the middle finishing moves nothing. Both
pointers turn their own visit into a slot with the same function
(`slot_of_visit`), so the writeback neither searches nor sorts — its pointer
is the loader's with a different number.

A slot cannot be handed on until the writeback releases it, so the tenant of
`slot_of_visit(t)` is visit `t`, always. The properties under test:

  (a) every comparison is between monotone visit indices, which cannot wrap
      into each other the way a layer number reduced by `% L` does;
  (b) the writeback reaches its slot by the rotation rather than by a
      search, and it asks about a visit and a ticket, never a layer — a slot
      remembers only its most recent tenant, so with n_slots=6 and L=32 slot
      0 carries layers 0,6,...,30 in one step and a layer-keyed question
      gets answered about the wrong one;
  (c) the rotation is one modulus, `visit % n_slots`, so slot reuse distance
      is n_slots everywhere and a layer's slot moves by `L mod n_slots` every
      step;
  (d) every wait on the rotation is woken by the fact it waits for, and has
      no clock.

The pool here is the real class built with `object.__new__` over real
`LayerSlot`s with no buffers (the constructor cudaHostRegisters a slab per
slot, and none of these bytes are ever touched). The writeback is the real
`WriteRouter._release` bound onto a harness holding only the FIFO it reads.
"""

import dataclasses
import os
import select
import threading
import time
import weakref
from collections import deque

import pytest

from elastic.vllm_runtime.layer_pool import (LayerSlot, LayerSlotPool,
                                             published)
from elastic.vllm_runtime.nvme_write_router import WriteRouter, _Desc

_ALL = 1 << 40          # a progress counter past every ticket in a test


def _pool(n: int, n_layers: int = 0) -> LayerSlotPool:
    """Only the bookkeeping, with the attributes `__init__` sets."""
    p = object.__new__(LayerSlotPool)
    p._slots = [LayerSlot(idx=i, raw_buf=None, t_buf=None)
                for i in range(n)]
    p.n_slots = n
    p._n_layers = int(n_layers) or n
    p.n_acquired = 0
    p.t_use_wait = 0.0
    p.n_use = 0
    p._cv = threading.Condition()
    p._head_visit = 0
    p._tail_visit = 0
    p._bind = list(range(p._n_layers))
    p._on_bind = lambda layer, slot: None
    p.n_released = 0
    p.n_moved = 0
    p.release_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    weakref.finalize(p, os.close, p.release_fd)
    return p


class _Tail:
    """The writeback's release half — the real `WriteRouter._release` — over
    a FIFO the test fills by hand. No thread, no NVMe engine, no signal."""
    _release = WriteRouter._release

    def __init__(self, pool):
        self._pool = pool
        self._fifo = deque()
        self._sub = 0
        self.n_stop_end = 0
        self.n_stop_save = 0

    def run(self, passed: int = _ALL) -> int:
        """One release with the progress counter at `passed`; returns how
        many visits it released."""
        n0 = self._pool.n_released
        self._release(passed)
        return self._pool.n_released - n0


def _desc(gate_seq: int, layer: int = 0, completed: bool = False) -> _Desc:
    d = _Desc(kind="kv_batch", layer=layer, gate_seq=gate_seq)
    d.completed = completed
    return d


def _use(p, visit, done_seq=None):
    """The forward pass consumes the tenancy: the loader publishes it, the
    forward pass stamps its layer end with a ticket (default: visit + 1, so
    tickets rise with visits as they do on the forward pass's stream)."""
    s = p._slots[p.slot_of_visit(visit)]
    p.mark_ready(s)
    p.stamp_forward_done(s, visit + 1 if done_seq is None else done_seq)
    return s


def _readable(fd) -> bool:
    return bool(select.select([fd], (), (), 0)[0])


def _drain(fd) -> int:
    try:
        return int.from_bytes(os.read(fd, 8), "little")
    except BlockingIOError:
        return 0


def _waiter(fn, *args):
    """Run a blocking pool wait on its own thread. Returns (thread, box):
    box["ret"] is what it returned, box["exc"] what it raised."""
    box = {}

    def run():
        try:
            box["ret"] = fn(*args)
        except BaseException as e:          # noqa: BLE001 — reported to test
            box["exc"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, box


def _still_waiting(th, settle: float = 0.1) -> bool:
    th.join(settle)
    return th.is_alive()


# ---- the loader ----------------------------------------------------------

def test_the_acquisition_reads_only_the_writebacks_release():
    """The loader compares two integers of the slot's own state: has the
    writeback released my last tenant (`released_visit`) against who that
    tenant was (`acquired_visit`). If it looked at the forward pass's stamp,
    the layer-end ticket or the writes instead, a GPU that has not drained or
    a save still reading would be judged on the path that decides whether the
    next load starts — those are the writeback's to hear."""
    L, n_slots = 8, 4
    p = _pool(n_slots, n_layers=L)
    for v in range(n_slots):
        s = p.acquire(v, v)
        assert s is not None and s.acquired_visit == v
    _use(p, 0)                           # forward-done, layer end not passed
    read = []

    class _Watch:
        def __init__(self, s):
            object.__setattr__(self, "_s", s)

        def __getattr__(self, name):
            read.append(name)
            return getattr(object.__getattribute__(self, "_s"), name)

        def __setattr__(self, name, value):
            read.append("set:" + name)
            setattr(object.__getattribute__(self, "_s"), name, value)

    p._slots[0] = _Watch(p._slots[0])
    assert p.acquire(n_slots, n_slots) is None       # every slot live — and only that
    assert set(read) == {"released_visit", "acquired_visit"}, read


def test_rotation_holds_exactly_n_slots_visits():
    """n_slots visits can be held at once and not one more; one released by
    the writeback makes room for exactly one."""
    L, n_slots = 8, 3
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(n_slots):
        assert p.acquire(v, v) is not None
    assert p.acquire(n_slots, n_slots) is None
    _use(p, 0)
    assert tail.run() == 1
    assert p.acquire(n_slots, n_slots) is not None   # one out, one in
    assert p.acquire(n_slots + 1, n_slots + 1) is None


def test_a_landing_write_keeps_the_slot_held():
    """A save still DMA-reading the slot's buffers holds the release — the
    writeback stops on it — and the loader simply finds the slot still held.
    No question about writes is asked on the acquisition path, and none is
    needed."""
    L, n_slots = 8, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(n_slots):
        p.acquire(v, v)
    _use(p, 0, done_seq=3)
    d = _desc(gate_seq=3)                # licensed by visit 0's layer end
    tail._fifo.append(d)
    tail._sub = 1
    assert tail.run() == 0 and tail.n_stop_save == 1
    assert p.acquire(2, 2) is None       # ... so the slot stays held
    d.completed = True                   # the write landed
    assert tail.run() == 1
    assert p.acquire(2, 2) is not None


def test_publication_names_a_visit_not_a_layer():
    """Publication is `published_visit == acquired_visit` — it names the
    tenancy, not the layer. A visit is monotone, so the loader's next `acquire`
    retires the previous publication by itself; a layer-keyed stamp would
    need the consumer to clear it, and a missed clear leaves a reader
    comparing against a stamp that will never be its own."""
    L, n_slots = 2, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    s = p.acquire(0, 0)
    assert not published(s)              # an acquisition is not a publication
    p.mark_ready(s)
    assert published(s)
    p.acquire(1, 1)                      # another layer acquiring its slot
    assert published(s)                  # ... touches no publication
    p.stamp_forward_done(s, 1)
    assert not published(s)              # the forward-done stamp ended it
    assert tail.run() == 1
    assert p.acquire(L, 0) is s          # same slot, same layer, next step
    assert not published(s)              # ... and the new tenancy is unfilled
    p.mark_ready(s)
    assert published(s)                  # published again, as itself


def test_a_retired_tenancy_never_reads_as_published():
    """Nothing clears the publication: the loader's next `acquire` retires it
    by moving `acquired_visit`. But once the forward pass is done with a
    tenancy (`stamp_forward_done`), `_bind[l]` still names it until the
    loader acquires the layer's next visit, so a retired tenancy's
    `published_visit == acquired_visit` still holds there, and a forward pass
    calling `await_ready` in that interval would be handed a slot whose
    bytes are about to be replaced. The publication therefore ends with the
    forward-done stamp, not with the next acquisition."""
    p = _pool(2, n_layers=8)
    tail = _Tail(p)
    s = p.acquire(0, 0)
    p.mark_ready(s)
    p.stamp_forward_done(s, 1)
    assert tail.run() == 1
    assert not published(s), (
        "a released, passed tenancy still reads as published — the next "
        "acquire has an interval where the worker sees stale weights as ready")


def test_the_forward_pass_follows_the_moving_binding_not_l_mod_n_slots():
    """The forward pass reaches its slot through the loader's published
    binding, so it tracks the rotation as it moves. With n_slots ∤ L the
    naive `l % n_slots` is right for the first step and silently wrong for
    every one after it, because the slot it names holds another layer's
    weights."""
    L, n_slots = 5, 3
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(L * 3):
        s = p.acquire(v, v % L)
        assert s is not None, v
        p.mark_ready(s)
        got = p.await_ready(v % L)
        assert got is s and got.idx == v % n_slots, (v, got.idx)
        p.stamp_forward_done(s, v + 1)
        assert tail.run() == 1


def test_a_stale_binding_never_reads_as_published():
    """`_bind` holds the visit, not the slot number, and that is what makes a
    leftover entry safe: until the loader has acquired this step's tenancy
    the forward pass is comparing against a visit that slot no longer holds,
    so it waits instead of taking somebody else's bytes. Keyed by slot number
    it would have matched whatever tenancy happened to be in that slot."""
    L, n_slots = 5, 3
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(L):                   # one whole step, consumed
        s = p.acquire(v, v % L)
        _use(p, v)
        assert tail.run() == 1
    # Layer 3's binding still names visit 3 -> slot 0. In the new step slot 0
    # goes to visit 6, i.e. layer 1 — so a binding read as a slot number would
    # hand layer 3's consumer layer 1's weights the moment they are published.
    assert p.slot_of_layer(3) == p.slot_of_visit(L + 1)
    th, box = _waiter(p.await_ready, 3)
    for v in (L, L + 1, L + 2):          # layers 0, 1, 2 acquired and published
        s = p.acquire(v, v % L)
        p.mark_ready(s)
        assert _still_waiting(th), (
            f"layer 3 read as published off a stale binding after visit {v} "
            f"(layer {v % L}) took a slot")
    # Its own tenancy, published, is what ends the wait.
    _use(p, L)
    assert tail.run() == 1
    s3 = p.acquire(L + 3, 3)
    p.mark_ready(s3)
    th.join(5)
    assert not th.is_alive() and box.get("ret") is s3
    assert s3.acquired_visit == L + 3


# ---- the rotation both pointers use --------------------------------------

def test_one_rotation_reached_from_either_side():
    """The loader and the writeback differ in their visit pointer, not in how
    they turn it into a slot; the forward pass differs only in reaching it
    through the loader's published binding. If these disagreed the writeback
    would be interrogating some other tenancy."""
    L, n_slots = 32, 6
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(L * 3):
        assert p.acquire(v, v % L) is not None, v
        assert p.slot_of_layer(v % L) == p.slot_of_visit(v)
        t, s = p.tail_slot()
        assert t == v and s is p._slots[p.slot_of_layer(v % L)]
        _use(p, v)
        assert tail.run() == 1


def test_the_rotation_is_one_modulus_so_reuse_distance_is_always_n_slots():
    """The rotation is one modulus, `visit % n_slots`, so a slot's reuse
    distance is exactly n_slots everywhere, step boundaries included, and the
    loader's slot sequence is 0,1,...,n_slots-1,0,1,... for the whole run.

    Composing two moduli as `(v % L) % n_slots` breaks that whenever n_slots
    does not divide L, because the inner one restarts first: L=32, n_slots=6
    hands slot 0 to visit 30 and wants it again at visit 32. Since the
    loader cannot have a slot until the writeback has released its previous
    tenant, the loader's visit pointer minus the writeback's would fall to
    `L mod n_slots` at every step boundary, collapsing how many visits the
    loader is ahead of the writeback, and the read-ahead with it. Only the
    cases with L % n_slots != 0 can distinguish the two, hence the n_slots=5
    and n_slots=7 rows below."""
    for L, n_slots in ((32, 6), (12, 5), (12, 6), (40, 7)):
        p = _pool(n_slots, n_layers=L)
        seq = [p.slot_of_visit(v) for v in range(4 * L)]
        assert seq[:n_slots + 1] == list(range(n_slots)) + [0], (L, n_slots, seq[:n_slots + 1])
        prev, reuse = {}, []
        for v, s in enumerate(seq):
            if s in prev:
                reuse.append(v - prev[s])
            prev[s] = v
        assert set(reuse) == {n_slots}, (L, n_slots, sorted(set(reuse)))


def test_a_layers_slot_moves_by_l_mod_n_slots_every_step():
    """The consequence of one modulus: a layer is not pinned to
    `l % n_slots`. Its slot advances by `L mod n_slots` each step, so the
    layer→slot map has to be published by whoever acquires the slot rather
    than recomputed by every reader — which is what `_bind` is, and what the
    alias of the K and V buffers and the activation buffer are re-pointed
    from."""
    L, n_slots = 32, 6
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    seen = []
    for v in range(L * 3):
        assert p.acquire(v, v % L) is not None
        if v % L == 0:
            seen.append(p.slot_of_layer(0))
        _use(p, v)
        tail.run()
    assert seen == [0, L % n_slots, (2 * L) % n_slots], seen


# ---- the writeback -------------------------------------------------------

def test_the_writeback_keys_on_tickets_not_layers():
    """The writeback never derives a layer from a visit, and never asks about
    one: what holds a visit is its slot's own layer-end ticket (`done_seq`)
    against the FIFO's tickets. The loader may hand any layer to any visit
    and the writes of a different layer licensed before that layer end still
    hold it, while a later ticket of the same layer does not."""
    L, n_slots = 8, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    s = p.acquire(0, 7)                  # visit 0 carries layer 7
    p.mark_ready(s)
    p.stamp_forward_done(s, 3)
    early = _desc(gate_seq=2, layer=0)   # another layer, earlier ticket
    tail._fifo.append(early)
    assert tail.run() == 0 and tail.n_stop_save == 1
    early.completed = True
    tail._sub = 1
    tail._fifo.append(_desc(gate_seq=4, layer=7))   # layer 7, next ticket
    assert tail.run() == 1


def test_writeback_reaches_its_slot_without_looking_at_the_others():
    """The writeback indexes; it does not search. With every slot live and
    only the oldest tenancy finished, it must touch exactly two slots — the
    one it releases and the one it stops on — not all n_slots. A sort by
    `acquired_visit`, a `min`, or any filter over `_slots` reads every one."""
    L, n_slots = 32, 6
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(n_slots):
        p.acquire(v, v % L)
    _use(p, 0)

    seen = set()

    class _Watch:
        def __init__(self, s):
            object.__setattr__(self, "_s", s)

        def __getattr__(self, name):
            seen.add(object.__getattribute__(self, "_s").idx)
            return getattr(object.__getattribute__(self, "_s"), name)

        def __setattr__(self, name, value):
            seen.add(object.__getattribute__(self, "_s").idx)
            setattr(object.__getattribute__(self, "_s"), name, value)

    p._slots = [_Watch(s) for s in list(p._slots)]
    assert tail.run() == 1
    assert seen == {p.slot_of_visit(0), p.slot_of_visit(1)}, seen


def test_writeback_releases_the_oldest_only():
    """A middle tenancy finishing moves nothing; once the oldest finishes both
    go, oldest first, in one release."""
    L, n_slots = 8, 4
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(n_slots):
        p.acquire(v, v)
    _use(p, 1)                           # a middle tenancy finishes
    assert tail.run() == 0               # ... and moves nothing
    assert tail.n_stop_end == 1
    _use(p, 0)
    assert tail.run() == 2               # now both, oldest first
    assert [s.released_visit for s in p._slots[:2]] == [0, 1]


def test_writeback_survives_a_step_wrap_when_n_slots_does_not_divide_l():
    """L=32, n_slots=6: slot 0 takes layers 0,6,...,30 and then layer 0 of
    the next step — 32 visits during which its per-slot state is overwritten
    five times. A writeback that recomputed its slot from a layer number
    would be answered about layer 30 and stop forever."""
    L, n_slots = 32, 6
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    head = 0
    for _ in range(L * 3):               # three whole steps
        s = p.acquire(head, head % L)
        assert s is not None, f"rotation jammed at visit {head}"
        head += 1
        _use(p, head - 1)
        tail.run()
    assert p.live_visits() == 0         # every slot released again
    assert p.n_released == head            # ... exactly once each


def test_slot_is_never_handed_on_before_the_writeback_releases_its_tenant():
    """The invariant every other test rests on, checked directly: while a
    visit is not released, nothing else may be given its slot — which is
    also why the slot's `done_seq` still belongs to the tenancy the writeback
    is asking about."""
    L, n_slots = 32, 6
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    holder = {}
    head = 0
    for _ in range(L * 2):
        s = p.acquire(head, head % L)
        if s is None:
            # Every slot live: retire the oldest and try again.
            oldest = p._tail_visit        # the pointer, not a search
            _use(p, oldest)
            assert tail.run() >= 1
            holder.pop(p._slots[p.slot_of_visit(oldest)].idx, None)
            continue
        assert s.idx not in holder, (
            f"slot {s.idx} re-claimed for visit {head} while visit "
            f"{holder.get(s.idx)} still holds it")
        holder[s.idx] = head
        head += 1
    # Past the step boundary: slot 0 goes from layer 30 to layer 0 with only
    # two visits between them.
    assert head > L


def test_forward_done_stamps_the_visit_and_its_ticket_not_the_layer():
    """Layers repeat every step; visits do not. The forward pass's
    forward-done stamp records the visit it held and the ticket its layer
    end is marked with, and a stamp left from the previous step's tenancy of
    the same layer cannot read as this one's."""
    L, n_slots = 2, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    s = p.acquire(0, 0)
    p.mark_ready(s)
    p.stamp_forward_done(s, 5)
    assert (s.forward_done_visit, s.done_seq) == (0, 5)
    p.acquire(1, 1)
    assert tail.run() == 1
    s2 = p.acquire(L, 0)                 # same layer, next step
    assert s2 is s and s.forward_done_visit == 0 < L
    assert tail.run() == 0               # visit 1 is live: nothing released
    _use(p, 1, done_seq=6)
    assert tail.run() == 1               # visit 1 goes, visit L stays
    assert tail.run() == 0 and p._tail_visit == L
    assert tail.n_stop_end >= 2          # ... held by its own, unmade stamp


def test_publishing_one_slot_says_nothing_about_another():
    """Slots are independent: publishing layer 1 leaves layer 0 exactly as it
    was, so a waiter on one slot is unaffected by another slot's
    publication."""
    L, n_slots = 4, 2
    p = _pool(n_slots, n_layers=L)
    s0 = p.acquire(0, 0)
    s1 = p.acquire(1, 1)
    p.mark_ready(s1)
    assert published(s1) and not published(s0)
    p.mark_ready(s0)
    assert published(s0)


def test_the_handshake_is_integers_on_the_slot():
    """Every fact the rotation reads is one of the slot's own integers with one
    writer. An Event or a flag object would be a second copy of the fact that
    somebody has to remember to clear, and a missed clear strands a reader."""
    buffers = {"raw_buf", "t_buf"}
    facts = {f.name: f.default for f in dataclasses.fields(LayerSlot)
             if f.name not in buffers}
    assert facts.pop("idx") is dataclasses.MISSING
    assert set(facts) == {"done_seq", "published_visit",
                          "acquired_visit", "forward_done_visit", "released_visit"}
    assert all(type(v) is int for v in facts.values()), facts


# ---- who wakes whom -------------------------------------------------------

def test_forward_done_signals_nobody_the_released_slot_does():
    """The forward pass's forward-done stamp is a stamp, not a signal: the
    layer end's mark (fired by the GPU passing it) is what the writeback
    hears, and the loader hears only when the writeback actually releases a
    slot (`notify_released` writes `release_fd`)."""
    L, n_slots = 8, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    s = p.acquire(0, 0)
    p.mark_ready(s)
    p.stamp_forward_done(s, 1)
    assert not _readable(p.release_fd)   # a stamp, no signal
    assert tail.run(passed=0) == 0       # layer end not passed: no release
    assert not _readable(p.release_fd)   # ... and still nothing said
    assert tail.run(passed=1) == 1
    assert _drain(p.release_fd) == 1     # ... and now a slot is released


def test_one_release_turn_says_so_once():
    """Several visits released by one turn are announced with one
    `notify_released`: the loader and the forward pass re-read the facts,
    they do not count wakes."""
    L, n_slots = 8, 4
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(3):
        p.acquire(v, v)
        _use(p, v)
    assert tail.run() == 3
    assert _drain(p.release_fd) == 1


def test_forward_done_is_paired_with_its_own_ticket():
    """`done_seq` is written before `forward_done_visit`, and the writeback
    reads them in the other order: a forward-done stamp it can see never
    comes with the previous tenancy's (already passed) ticket, so the
    writeback holds the slot until this tenancy's layer end has passed."""
    L, n_slots = 4, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    s = p.acquire(0, 0)
    p.mark_ready(s)
    p.stamp_forward_done(s, 5)            # layer end = ticket 5
    assert tail.run(passed=0) == 0       # not passed yet: held
    assert tail.run(passed=4) == 0
    assert tail.run(passed=5) == 1       # the mark fired
    # The write order, on the slot the forward pass stamps.
    order = []

    class _Rec:
        def __init__(self, s):
            object.__setattr__(self, "_s", s)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_s"), name)

        def __setattr__(self, name, value):
            order.append(name)
            setattr(object.__getattribute__(self, "_s"), name, value)

    s1 = p.acquire(1, 1)
    p.stamp_forward_done(_Rec(s1), 7)
    assert order.index("done_seq") < order.index("forward_done_visit"), order


def test_writeback_release_needs_no_lock_against_a_concurrent_acquisition():
    """The loader's acquisition and the writeback's release touch disjoint
    facts — `acquired_visit` on one side, `released_visit` and the
    writeback's visit pointer on the other, each with one writer that only
    ever increases it — so an acquisition racing a release can at worst read
    a stale `released_visit` and refuse a slot it could have had. It can
    never be handed a live one."""
    L, n_slots = 32, 6
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    stop = []

    def run_tail():
        while not stop:
            tail._release(_ALL)

    th = threading.Thread(target=run_tail, daemon=True)
    th.start()
    head, attempts = 0, 0
    holder = {}
    try:
        while head < L * 4 and attempts < 10_000_000:
            attempts += 1
            s = p.acquire(head, head % L)
            if s is None:
                continue                 # every slot live: writeback decides
            assert s.idx not in holder, f"slot {s.idx} handed out twice"
            holder[s.idx] = head
            _use(p, head)                # forward pass consumes it immediately
            del holder[s.idx]
            head += 1
    finally:
        stop.append(1)
        th.join(timeout=2.0)
    assert head == L * 4, f"jammed at visit {head} after {attempts} attempts"


# ---- the x-wheel's binding ------------------------------------------------

def test_all_layers_resident_leaves_a_live_next_layer_alone():
    """Every layer resident (n_slots == L): the loader can be a whole step
    ahead, so when it acquires layer l's slot for the next step the forward
    pass may still be waiting on layer l+1 of this one. The early l+1 stamp
    would send that wait to the next step's visit, which cannot come first —
    so it is not made. The two visits are L apart, so with L slots they name
    the same slot anyway."""
    L = 4
    p = _pool(L, n_layers=L)
    tail = _Tail(p)
    for v in range(L):
        assert p.acquire(v, v) is not None
    _use(p, 0)
    assert tail.run() == 1
    assert p.acquire(L, 0) is not None   # layer 0, next step
    assert p._bind[1] == 1               # layer 1's live visit, kept
    assert p.slot_of_layer(1) == (L + 1) % L    # ... which is the same slot
    assert p._bind[0] == L


def test_rotating_slots_stamp_the_next_layer_early():
    """n_slots < L: the entry being replaced was released long ago, and the
    x-wheel needs the next visit's slot before the loader acquires it. Checked
    on the second step, where the initial table (`visit == layer`) no longer
    says it."""
    L, n_slots = 8, 3
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(L):                   # one whole step, consumed
        assert p.acquire(v, v) is not None
        _use(p, v)
        assert tail.run() == 1
    assert p._bind[1] == 1               # the first step's visit
    assert p.acquire(L, 0) is not None   # layer 0, next step
    assert p._bind[1] == L + 1           # layer 1, stamped before acquisition
    assert p.slot_of_layer(1) == p.slot_of_visit(L + 1)
    assert p._slots[p.slot_of_layer(1)].acquired_visit < L + 1   # not acquired yet


def test_one_slot_is_refused(monkeypatch):
    """With one slot, layer p's output would land in its own slot (the next
    visit's activation buffer is p's own). The slot count is a
    configuration, refused where it is derived (`_compute_n_slots`)."""
    import types

    import torch

    from elastic.vllm_runtime.streaming_loader import StreamingModelLoader
    monkeypatch.delenv("ELASTIC_SLOTS", raising=False)
    text = types.SimpleNamespace(num_attention_heads=4, hidden_size=64)
    model_config = types.SimpleNamespace(
        max_model_len=64, dtype=torch.float16,
        hf_config=types.SimpleNamespace(get_text_config=lambda: text))
    vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=2),
        cache_config=types.SimpleNamespace(block_size=16))
    per_layer = 2 * 64 * (2 * 4 * 16 * 2) + 4096   # max_batch·max_seq·kv + weight
    with pytest.raises(ValueError):
        StreamingModelLoader._compute_n_slots(
            slots_mem_bytes=per_layer * 3 // 2, model_config=model_config,
            vllm_config=vllm_config, num_layers=4, max_layer_bytes=4096)


# ---- the forward pass's waits ---------------------------------------------

def test_the_forward_pass_is_woken_by_the_publication_not_a_clock():
    """`await_ready` sleeps on the pool's condition and the loader's
    publication notifies it: the wait ends when the fact is stamped, and
    there is no sleep loop to wait out."""
    p = _pool(2, n_layers=4)
    s = p.acquire(0, 0)
    th, box = _waiter(p.await_ready, 0)
    assert _still_waiting(th)            # nothing published yet
    t0 = time.perf_counter()
    p.mark_ready(s)
    th.join(5)
    assert box.get("ret") is s
    assert time.perf_counter() - t0 < 0.05


def test_await_released_is_woken_by_the_writeback_release():
    """The x-wheel's wait: visit 2 lands in slot 0, still held by visit 0
    until the writeback releases it. The release — `release_tail`, then
    `notify_released` — ends the wait, and the same turn also tells the
    loader."""
    L, n_slots = 8, 2
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(n_slots):
        assert p.acquire(v, v) is not None
    _use(p, 0)
    th, box = _waiter(p.await_released, 2)
    assert _still_waiting(th)
    assert tail.run() == 1
    th.join(5)
    assert not th.is_alive() and "exc" not in box
    assert _readable(p.release_fd)


def test_await_released_asks_about_the_previous_tenant_of_that_visit():
    """`await_released(v)` is "has the writeback released visit v − n_slots".
    A visit of the first round has no previous tenant and never waits; a
    later one waits for exactly its own predecessor — not for the writeback
    to be anywhere else."""
    L, n_slots = 8, 3
    p = _pool(n_slots, n_layers=L)
    tail = _Tail(p)
    for v in range(n_slots):
        p.await_released(v)                  # first round: nobody before
        p.acquire(v, v)
    _use(p, 0)
    _use(p, 1)
    assert tail.run() == 2
    p.await_released(n_slots)                      # visit 0 released
    p.await_released(n_slots + 1)                  # visit 1 released
    th, box = _waiter(p.await_released, n_slots + 2)
    assert _still_waiting(th)            # visit 2 is not
    _use(p, 2)
    assert tail.run() == 1
    th.join(5)
    assert not th.is_alive() and "exc" not in box


def test_await_released_with_all_layers_resident_waits_for_the_live_tenant():
    """Every layer resident (n_slots == L) and the loader a step ahead: the
    loader has acquired layer 0's slot for the next step (visit L) and, by
    design, left `_bind[1]` naming this step's visit 1. The forward pass
    running visit L asks for the slot of visit L+1. Visit 1's writes are
    still landing, so the answer is "not yet" — a question keyed by the
    layer entry would have asked about visit 1 − L, which is long gone, and
    gone ahead onto bytes a save is reading."""
    L = 4
    p = _pool(L, n_layers=L)
    tail = _Tail(p)
    for v in range(L):
        p.acquire(v, v)
        _use(p, v)                       # forward pass ran the whole step
    landing = _desc(gate_seq=2)          # licensed by visit 1's layer end
    tail._fifo.append(landing)
    tail._sub = 1
    assert tail.run() == 1               # visit 0 released, visit 1 held
    assert p.acquire(L, 0) is not None   # layer 0, next step
    assert p._bind[1] == 1               # not stamped ahead: resident
    th, box = _waiter(p.await_released, L + 1)
    assert _still_waiting(th)
    landing.completed = True
    assert tail.run() == L - 1           # visits 1..3 released
    th.join(5)
    assert not th.is_alive() and "exc" not in box
