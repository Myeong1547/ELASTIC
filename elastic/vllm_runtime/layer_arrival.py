"""Per-layer arrival-order readiness feed: loader → forward pass, for
arrival-based decode attention.

The forward pass's attention consumes a layer's reloaded KV in the order its
blocks arrive. A block arrives when it becomes available for attention: a KV
block when both its K and V have been loaded (a KV run landed via io_uring),
an activation block when its kv_gen has completed (a kv_gen launch over
loaded activations was dispatched; the consumer's stream waits on its
event). The forward pass does a partial attention call + LSE merging per
arrived batch.

Producer (the loader, ReadRouter), once per layer load:
    arr.begin(layer)                # a new visit's feed: empty and open
    pool.mark_ready(slot)           # weight-only ready; forward pass starts here
    ... per completed bulk op:
        a KV run: after its CQE      arr.push(layer, ArrivalItem(1, r0, n))
        a kv_gen launch: after it is launched, with the event
        recorded behind it           arr.push(layer, ArrivalItem(0, r0, n, ev))
    arr.close(layer)                # feed terminator

Consumer (the forward pass, inside the layer's attention):
    for batch in arr.arrived_batches(layer, min_blocks): ...the stream waits
    the batch's last ACT event, partial attention call + LSE merging...
                                    # ends on close+exhausted

The queue carries readiness only — a block region's rank run, nothing else: no
sequence ids, no positions, no token counts (the forward pass turns a rank run
back into query rows with the layout's closed form and decides what and how
much it attends from seq_lens).

Only the loader begins, pushes and closes. A visit's feed is closed before the
forward pass finishes that visit (a decode visit consumes it to its close; a
prefill visit reloads nothing and its plan completes with its weights), and the
loader begins the layer's next visit only in a slot the writeback released
after that — so a begin never meets an open feed, and a push always lands in
the visit it belongs to.

Race-freeness of the handshake: the producer records the CUDA event before
push (record happens-before the locked append); the consumer reads the item
under the lock before its stream waits on the event — so an observed item
always carries its op's completion.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, List


@dataclass
class ArrivalItem:
    """One completed bulk op: `n` consecutive ranks of one block region of the
    layer whose bytes just became available for attention.

    kind   : 0 = the activation block region (K,V recomputed by kv_gen now
             sit in buffer blocks `1 + rank0 .. 1 + rank0 + n − 1`), 1 = the
             K/V block regions (buffer blocks `mem_kv_base + rank0 ..`). The
             buffer is filled in load order (= file offset order), so a rank
             run lands in one contiguous range and the item names nothing
             else.
    rank0  : first rank of the run in its block region.
    n      : run length in blocks (> 0).
    event  : a CUDA event handle (`EventRing`) recorded on the stream that
             produced the bytes, right after it (the run's kv_gen launch);
             the consumer's stream waits on it. None when nothing on a stream
             produced them (a KV DMA, handed over after its CQE).

    Which sequences the run touches, and where each block sits in its
    sequence, is the layout's closed form (`BlockKVLayout.rank_tables` →
    global numbers k, `seq_of_ks` → sequence, `j_of_ks` → position in the
    row) — the consumer evaluates it, the producer does not carry it.
    """
    kind: int
    rank0: int
    n: int
    event: Any = None


class LayerArrival:
    """Per-layer arrival-order readiness queue. One instance shared between
    the single loader thread (producer) and the forward thread (consumer)."""

    def __init__(self, num_layers: int) -> None:
        # One condition per layer. All the state below is per-layer, and a
        # shared lock would be a shared refusal: pushing layer l's items would
        # wait behind the consumer of layer m, and `notify_all` would wake
        # every layer's consumer to tell all but one of them nothing.
        self._cond: List[threading.Condition] = [
            threading.Condition() for _ in range(num_layers)]
        self._q: List[List[ArrivalItem]] = [[] for _ in range(num_layers)]
        self._open: List[bool] = [False] * num_layers
        # Running block count of everything pushed for the layer's current
        # visit, so the min_blocks yield test is O(1): unconsumed blocks =
        # pushed − consumed, never a re-sum over the unconsumed items.
        self._blocks_pushed: List[int] = [0] * num_layers
        # What the waiting consumer needs before it can go on: the pushed
        # block count at which its yield test turns true (0: nobody is
        # waiting). A push below it changes nothing the consumer can act on,
        # so it does not wake it; close always does.
        self._need: List[int] = [0] * num_layers

    # ---- producer (loader thread) ------------------------------------

    def begin(self, layer_idx: int) -> None:
        with self._cond[layer_idx]:
            self._q[layer_idx] = []
            self._blocks_pushed[layer_idx] = 0
            self._need[layer_idx] = 0
            self._open[layer_idx] = True

    def push(self, layer_idx: int, item: ArrivalItem) -> None:
        with self._cond[layer_idx]:
            self._q[layer_idx].append(item)
            self._blocks_pushed[layer_idx] += int(item.n)
            if self._blocks_pushed[layer_idx] >= self._need[layer_idx]:
                self._cond[layer_idx].notify_all()

    def close(self, layer_idx: int) -> None:
        with self._cond[layer_idx]:
            self._open[layer_idx] = False
            self._cond[layer_idx].notify_all()

    # ---- consumer (forward pass) -------------------------------------

    def arrived_batches(self, layer_idx: int, min_blocks: int):
        """Yield lists of items (>= 1 each) in arrival order. Blocks (CPU)
        while the queue is open and holds less than the budget; terminates
        once the layer is closed AND every item has been yielded.

        `min_blocks` (> 0) is a block budget, not an item count: an open
        queue is not yielded until the unconsumed items cover at least that
        many KV blocks, AND a yield carries at most one budget's worth — the
        shortest prefix that reaches it — leaving the rest queued for the
        next yield (close always flushes the remainder, budget by budget).
        Items are bulk ops of very different sizes (a K/V segment pair's
        run vs one kv_gen launch), so a block threshold tracks "KV tokens
        arrived" where an item count cannot (the caller derives it from
        ELASTIC_ARRIVAL_DRAIN_TOKENS / block_size). While the loader is
        still reading, each yield's partial attention call overlaps the
        blocks still loading.

        The upper half of that budget is part of what the knob means: one
        partial attention call is one varlen call plus its LSE merging, and
        attention cost is not linear in call width, so a call carrying a whole
        layer costs more per token than the same work split into budget-sized
        calls (NHD per-head reads leave the linear zone). Without the cap the
        call width would be set by how far ahead the loader happens to be — a
        loader that finished first would hand the consumer the entire layer in
        one call. Expressing the budget in tokens rather than rows keeps the
        granularity fixed as the context length changes.

        The wait has no deadline: the loader closes every feed it opens once
        the layer's reads have landed."""
        head = 0
        blocks_consumed = 0       # blocks in items already yielded
        while True:
            with self._cond[layer_idx]:
                q = self._q[layer_idx]
                # Unconsumed block volume without re-walking the items.
                unconsumed = self._blocks_pushed[layer_idx] - blocks_consumed
                if (head < len(q)
                        and (unconsumed >= min_blocks
                             or not self._open[layer_idx])):
                    # Shortest prefix that reaches the budget. A single
                    # item may exceed it on its own (a whole segment's
                    # rank run is one item) — it still goes out alone
                    # rather than being split, because an item is the
                    # unit the loader made available for attention.
                    _n = _blk = 0
                    for _it in q[head:]:
                        _n += 1
                        _blk += int(_it.n)
                        if _blk >= min_blocks:
                            break
                    batch = q[head:head + _n]
                    head += len(batch)
                    blocks_consumed += sum(int(it.n) for it in batch)
                elif not self._open[layer_idx] and head >= len(q):
                    return
                else:
                    # The queue is state: `push` appends under this lock and
                    # `head` indexes into it, so an item pushed while nobody
                    # was consuming is still here when the consumer arrives —
                    # a notify can never be missed. What this wait needs is
                    # recorded under the same lock: the yield test turns
                    # true once the unconsumed volume reaches the budget, so
                    # `push` wakes it only then.
                    self._need[layer_idx] = blocks_consumed + min_blocks
                    self._cond[layer_idx].wait()
                    continue
            yield batch  # off-lock: caller does GPU work per batch
