"""Unit tests for LayerArrival (no GPU needed).

No wait in LayerArrival has a timeout: the loader closes every feed it opens.
So the tests that put a thread in a wait run it on a daemon thread and bound
only the join — a regression hangs that thread, not the suite.
"""

import threading
import time

from elastic.vllm_runtime.layer_arrival import ArrivalItem, LayerArrival

_JOIN_S = 10.0


def _item(rank0, n=1, kind=1):
    """One block-region rank run; `rank0` doubles as the item's identity in
    the order checks below."""
    return ArrivalItem(kind=kind, rank0=rank0, n=n, event=None)


def _spawn(fn, *args, **kw):
    """Run `fn` on a daemon thread; the returned dict gets 'value' or
    'error' when it returns."""
    out = {}

    def run():
        try:
            out["value"] = fn(*args, **kw)
        except BaseException as e:      # noqa: BLE001 — reported to the test
            out["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def _joined(t):
    t.join(_JOIN_S)
    assert not t.is_alive(), "a wait that should have ended is still waiting"


def _waiting(cond, n=1):
    """Return once `n` threads sit in `cond.wait()`, so what the test does
    next reaches a thread that is really waiting rather than one that has not
    got there yet."""
    t_end = time.monotonic() + _JOIN_S
    while len(cond._waiters) < n:
        assert time.monotonic() < t_end, "the waiter never got to its wait"
        time.sleep(0.001)


def _consume_all(arr, layer, **kw):
    return [[it.rank0 for it in b] for b in arr.arrived_batches(layer, **kw)]


def test_arrival_order_and_close():
    """Items come out in push order, and close ends `arrived_batches`."""
    arr = LayerArrival(num_layers=2)
    arr.begin(0)
    for r in (3, 1, 2):
        arr.push(0, _item(r))
    arr.close(0)
    got = [it.rank0 for batch in arr.arrived_batches(0, min_blocks=1)
           for it in batch]
    assert got == [3, 1, 2], got


def test_streaming_batches():
    """Items pushed while the consumer takes arrived batches come out as
    separate batches; `arrived_batches` blocks on an open-but-empty queue
    and wakes on push/close."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    arr.push(0, _item(10))
    t, out = _spawn(_consume_all, arr, 0, min_blocks=1)
    _waiting(arr._cond[0])                 # took [10], now waiting
    arr.push(0, _item(11))
    arr.push(0, _item(12))
    _waiting(arr._cond[0])
    arr.push(0, _item(13))
    arr.close(0)
    _joined(t)
    assert "error" not in out, out
    seen_batches = out["value"]
    flat = [r for b in seen_batches for r in b]
    assert flat == [10, 11, 12, 13], seen_batches
    assert len(seen_batches) >= 2, seen_batches  # trickle, not one blob


def test_empty_close():
    """A feed opened and closed with nothing pushed yields no batch."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    arr.close(0)
    assert list(arr.arrived_batches(0, min_blocks=1)) == []


def test_reset_on_begin():
    """begin() starts a new visit: the previous one's items are gone."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    arr.push(0, _item(1))
    arr.close(0)
    arr.begin(0)  # next visit: prior items dropped
    arr.push(0, _item(2))
    arr.close(0)
    got = [it.rank0 for b in arr.arrived_batches(0, min_blocks=1) for it in b]
    assert got == [2], got


def test_min_blocks_threshold_and_close_flush():
    """min_blocks coarsens by block volume (an item's `n`), and close always
    flushes the remainder. Guards the cumulative pushed/yielded counters the
    threshold test runs on: an off-by-one there either yields early
    (threshold not honoured) or strands the remainder."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    # 3 blocks queued, threshold 4 → nothing may be yielded yet.
    arr.push(0, _item(1, n=2))
    arr.push(0, _item(2, n=1))
    gen = arr.arrived_batches(0, min_blocks=4)
    # Crossing the threshold releases exactly the queued items...
    arr.push(0, _item(3, n=2))
    first = next(gen)
    assert [it.rank0 for it in first] == [1, 2, 3], first
    # ...and the next 2 blocks are under the threshold, so only close
    # releases them (the counter must not carry the yielded 5 forward).
    arr.push(0, _item(4, n=2))
    arr.close(0)
    rest = [it.rank0 for batch in gen for it in batch]
    assert rest == [4], rest


def test_min_blocks_holds_an_open_feed_below_the_budget():
    """Below the block budget an open feed yields nothing —
    `arrived_batches` waits rather than handing out a short batch — and the
    push that crosses the budget wakes it."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    arr.push(0, _item(1, n=3))
    gen = arr.arrived_batches(0, min_blocks=4)
    t, out = _spawn(next, gen)
    _waiting(arr._cond[0])                 # 3 < 4: waiting, nothing yielded
    assert "value" not in out
    arr.push(0, _item(2, n=1))
    _joined(t)
    assert [it.rank0 for it in out["value"]] == [1, 2], out
    arr.close(0)
    assert list(gen) == []


def test_min_blocks_shortest_prefix():
    """A yield carries at most one budget's worth — the shortest prefix that
    reaches it — and never splits an item: a run longer than the budget goes
    out alone, the rest stays queued for the next yield."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    for r, n in ((1, 3), (2, 3), (3, 9), (4, 1)):
        arr.push(0, _item(r, n=n))
    arr.close(0)
    got = _consume_all(arr, 0, min_blocks=4)
    assert got == [[1, 2], [3], [4]], got


def test_min_blocks_resets_on_begin():
    """begin() starts a new visit — a previous visit's pushed block count
    must not pre-satisfy the next one's threshold."""
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    arr.push(0, _item(1, n=6))
    arr.close(0)
    assert [it.rank0 for b in arr.arrived_batches(0, min_blocks=4)
            for it in b] == [1]
    arr.begin(0)
    arr.push(0, _item(2, n=1))
    gen = arr.arrived_batches(0, min_blocks=4)
    # The feed is still open: a carried-over count would yield here.
    t, out = _spawn(next, gen)
    _waiting(arr._cond[0])
    assert "value" not in out and "error" not in out
    arr.close(0)          # only close may release the single under-threshold block
    _joined(t)
    assert [it.rank0 for it in out["value"]] == [2]
    assert list(gen) == []


def test_layers_do_not_share_a_lock():
    """Layers are independent, so their locks are: layer 0's queue held by
    somebody else must not stop layer 1 from being pushed. A single shared
    Condition would make every push wait behind every other layer's
    `arrived_batches` and wake every consumer to tell all but one of them
    nothing."""
    arr = LayerArrival(num_layers=4)
    arr.begin(0); arr.begin(1)
    held = threading.Event(); done = threading.Event()

    def hold_layer0():
        with arr._cond[0]:
            held.set()
            done.wait(5.0)

    t = threading.Thread(target=hold_layer0, daemon=True)
    t.start()
    assert held.wait(5.0)
    t1 = time.perf_counter()
    arr.push(1, _item(9))                     # must not wait on layer 0
    assert time.perf_counter() - t1 < 1.0, "push(L=1) blocked behind layer 0"
    done.set(); t.join(5.0)
    assert len(arr._q[1]) == 1 and len(arr._q[0]) == 0

