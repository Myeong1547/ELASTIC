"""Persistent layer-slot pool with n_slots slots.

n_slots is derived from the memory budget the caller provides:
`_compute_n_slots` (streaming_loader) takes the largest n with
`n·m_w_kv + _act_buffers_bytes(n) ≤ slots_mem_bytes`, one M_slot per slot
(weight buffer + K and V buffers + activation buffer).

Each slot is cudaHostRegister'd once at startup and reused for the whole
run; registration is expensive enough that doing it per layer per step
would dominate the hot path. The loader (ReadRouter) acquires the rotation's
next slot once the writeback (WriteRouter) has released it, has the layer's
weight bytes read into it and publishes the layer (`mark_ready`). The forward
pass reaches the same slot through the same rotation — `visit % n_slots`,
published per layer in `_bind`, so neither side searches — runs the layer,
and at the layer end stamps the slot with the ticket its layer end is marked
with (`done_seq`). The writeback hears that mark pass, sends the layer's
writes, and once they have landed releases the visit, which frees the slot
for its next visit.

The rotation is a single modulus over visits (`slot_of_visit`), not over
layer numbers: `l % n_slots` rewinds early at a step boundary whenever
n_slots does not divide the layer count L, which shrinks how far the
loader's visit pointer runs ahead of the writeback's to `L mod n_slots` once
per step.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# One eventfd count: what the writeback writes to wake the loader.
_EFD_ONE = (1).to_bytes(8, "little")


def published(s: "LayerSlot") -> bool:
    """Are this slot's bytes in, for the tenancy it is holding right now?

    Two monotone comparisons and no clearing anywhere:
      `published_visit == acquired_visit`  the loader said the bytes landed for
                                      this tenancy (a visit, never a layer
                                      number — a layer repeats every step, a
                                      visit does not)
      `acquired_visit > forward_done_visit`     and the consumer has not finished with it

    The second term is load-bearing: once the forward pass has finished with
    a tenancy (`stamp_forward_done`), `_bind[l]` still names it until the
    loader acquires the layer's next visit, and without the term `await_ready`
    would return the finished slot. That stamp is the fact that ends a
    publication, so the publication ends at that instant and nobody has to
    remember to clear anything.
    """
    return s.published_visit == s.acquired_visit > s.forward_done_visit


@dataclass
class LayerSlot:
    idx: int                          # slot index (0..n_slots-1), static
    raw_buf: np.ndarray               # uint8 numpy view, 4 KiB-aligned
    t_buf: torch.Tensor               # uint8 CUDA tensor over the same memory
    done_seq: int = 0                 # the ticket this tenancy's layer end is
                                      # marked with on the forward pass's
                                      # stream; the mark's callback raises the
                                      # progress counter past it when the GPU
                                      # gets there. 0 = never occupied.
    published_visit: int = -1         # the tenancy whose bytes are in. Read
                                      # through `published()` above, never on
                                      # its own — the forward-done stamp is
                                      # half of the answer.
                                      # A visit, not a layer number: visits are
                                      # monotone, so `published_visit ==
                                      # acquired_visit` names this tenancy and a
                                      # value left over from the previous one
                                      # can never match. Nothing has to clear
                                      # it — the loader's next `acquire` moves
                                      # `acquired_visit` and the comparison goes
                                      # false by itself, exactly as
                                      # `released_visit < acquired_visit` does for
                                      # whether the slot is released.
    # ---- the loader's and the writeback's visit pointers ---------------
    # Visits, not layers. A layer number repeats every step and a slot only
    # ever remembers its most recent tenant, so per-step numbering cannot
    # answer "is the oldest tenancy finished with" once the two pointers are
    # more than a step apart. A visit index is monotone for the whole run, so
    # every comparison below is between two integers that can never wrap into
    # each other.
    #
    # Both pointers turn the visit into a slot with the same function
    # (`LayerSlotPool.slot_of_visit`), so neither side stores the other's
    # answer: the loader is at one visit, the writeback at another, and that
    # is the only difference between them.
    acquired_visit: int = -1          # the visit this slot holds (the
                                      # loader's `acquire`)
    forward_done_visit: int = -1           # visit the forward pass is done with
    released_visit: int = -1            # visit the writeback has released.
                                      # The release lives on the slot, not in
                                      # a pool-wide watermark: the loader is at
                                      # the newest live visit and must not be
                                      # refused by a fact about the oldest
                                      # one. Free <=> == acquired_visit.


@dataclass
class LayerBuffer:
    """A registered UMA buffer (the activation-buffer slab, the arrival index
    buffer) or a view of one: the same bytes as a numpy array and a CUDA
    tensor."""
    raw_buf: "np.ndarray"
    t_buf: "torch.Tensor"


# Activation buffers and index spans are carved at page granularity, so every
# one starts page-aligned for O_DIRECT.
_ALIGN = 4096


class ActBufferPool:
    """The slot's activation buffer: one (slot.act) per slot, alongside the
    weight buffer and the K and V buffers.

    Layer l's activations go into the activation buffer with its slot's
    index, exactly as its weights live in that slot and its KV in that slot's
    K and V buffers — one coupled unit, one rotation, and the rotation is over
    visits (`LayerSlotPool.slot_of_visit`), so this asks the pool rather than
    recomputing anything (`slot_of`, given at construction). The activation
    buffer has the K and V buffers' layout with one unit per block instead of
    two (K,V) — an activation block is half of a KV block — so n_slots
    activation buffers are exactly the `0.5·kv·n_slots` the solver folds into
    M_slot (`solver.py` `_m_slot`: M_slot = weight + kv + 0.5·kv).

    Nothing recycles inside a layer. The loaded activations stay live from
    their NVMe landing through kv_gen and the forward pass, and are
    released by the same event that releases the weight buffer (the slot is
    one coupled unit, released by the writeback once its layer end has passed
    and its writes have landed — `LayerSlotPool.release_tail`), and the
    forward pass's stream waits for kv_gen(l) before it attends them.
    A kv_gen launch is therefore a submit grouping over the activation
    buffer, not an allocation unit.

    The slab is carved in space, never time-multiplexed:
    [activation buffer 0 .. n_slots-1]. In the plain prefill the capture
    LayerNorm writes the layer's own activation buffer (`hidden_span`); in the
    zig-zag prefill `prep_span` writes it (`act_buffer_rows`); in decode
    `arrival_decode` appends each row into its block in the activation
    buffer. An activation buffer is `T·d·dtype`, i.e. exactly one whole-batch
    hidden, which is what the zig-zag prefill needs and what the decode reload
    fills; prefill reloads nothing, so there the activation buffer is the
    hidden matrix.
    """

    def __init__(self, n_slots: int, act_buffer_bytes: int, alloc_aligned,
                 slot_of, index_bytes: int, launch_blocks: int):
        act_buffer_bytes = (((int(act_buffer_bytes) + _ALIGN - 1) // _ALIGN)
                        * _ALIGN)
        # Allocation is eager on purpose. Deferring it would let a low-alpha
        # configuration run well under the budget that reserved the pool while
        # a high-alpha one runs right at it — an alpha-dependent OOM.
        # Allocating what the budget already reserved keeps accounting ==
        # allocation, the invariant `_act_buffers_bytes` exists to hold.
        # One registered slab, permanently carved in space (not
        # time-multiplexed): [activation buffer 0 .. n_slots-1], one per slot.
        # Activation buffers never alias; they are views, and the slab owns
        # the registration.
        self._slab = alloc_layer_buffer(n_slots * act_buffer_bytes, alloc_aligned)
        self._act_buffers: list["LayerBuffer"] = [
            LayerBuffer(
                raw_buf=self._slab.raw_buf[i * act_buffer_bytes:
                                            (i + 1) * act_buffer_bytes],
                t_buf=self._slab.t_buf[i * act_buffer_bytes:
                                        (i + 1) * act_buffer_bytes],
            )
            for i in range(n_slots)]
        self.n_slots = n_slots
        self.act_buffer_bytes = act_buffer_bytes
        # Blocks per kv_gen launch (ELASTIC_LOAD_CHUNK_TOKENS): a submit and
        # launch grouping over the activation buffer, not an allocation unit.
        self.launch_blocks = int(launch_blocks)
        # The index operands of each partial attention call (a varlen call:
        # query rows, block table, seqused, ALiBi gap) and the K/V and ACT
        # appends' row selections, written straight into registered memory.
        # `torch.tensor(list, device=cuda)` would copy from pageable memory
        # and block the host on the stream, turning the arrival feed (queue a
        # wait and run ahead) into a synchronous request/response.
        #
        # Bump-allocated within a layer visit, never recycled inside one: a
        # layer's partial attention calls each keep their operands live until
        # their kernel retires, so handing call k+1 call k's bytes would be a
        # host write racing a live kernel. Reuse is gated by the slot rotation
        # instead — the same rule the activation buffer's own bytes follow.
        # One span per slot, sized once from the closed-form bound on a
        # visit's calls (`arrival_attention.arrival_index_bytes`), in one
        # registration.
        self.index_bytes = (((int(index_bytes) + _ALIGN - 1) // _ALIGN)
                            * _ALIGN)
        self._idx = alloc_layer_buffer(n_slots * self.index_bytes,
                                       alloc_aligned)
        _ib = self.index_bytes
        self._arr_idx: list = [
            (self._idx.raw_buf[i * _ib:(i + 1) * _ib].view(np.uint8),
             self._idx.t_buf[i * _ib:(i + 1) * _ib].view(torch.uint8))
            for i in range(n_slots)]
        self._arr_used: list = [0] * n_slots
        # Per-slot capture-LayerNorm row statistics: the out1/out2 targets
        # `native_layer_norm.out` needs. fp32 [rows, 1] pairs, allocated
        # lazily, GPU-only, write-only — nothing reads them back.
        self._stats: dict = {}
        # How a layer finds its activation buffer. The slot is one coupled
        # unit, so this is the weight buffer's rotation and nothing of its
        # own: `LayerSlotPool.slot_of_layer`, the single authority for it.
        self._slot_of = slot_of

    def act_buffer(self, layer_idx: int) -> "LayerBuffer":
        """Layer `layer_idx`'s activation buffer (slot.act) — in its slot,
        the same rotation (and the same slot number) its weights and KV
        follow.

        Reuse is gated by the slot itself. The slot is not released
        until the previous tenancy's forward pass has finished and its writes
        have landed, and the loader reads a layer's activations only after it
        has acquired that layer's slot, so the activation buffer cannot be
        overwritten while the previous tenant's activations are still being
        read."""
        return self._act_buffers[self._slot_of(layer_idx)]

    def hidden_span(self, layer_idx: int, nbytes: int) -> "torch.Tensor":
        """Where layer `layer_idx`'s capture LayerNorm writes in the plain
        prefill: the first `nbytes` of its own activation buffer, as a CUDA
        uint8 view.

        Prefill reloads nothing (fresh sequences have no past), so the
        activation buffer — sized `T·d·dtype`, one whole-batch hidden — is
        the hidden matrix.

        Decode does not call this: there a row's place is its own block's
        empty position in the part of the activation buffer the reload
        filled, and the row goes in once that block's activation load has
        completed (`arrival_decode`).

        The activation buffer is the layer's for as long as it holds the
        slot, and the writeback releases the tenancy only once its writes
        have landed."""
        r = self._act_buffers[self._slot_of(layer_idx)]
        return r.t_buf[:int(nbytes)]

    def arrival_index_reset(self, layer_idx: int) -> None:
        """Start a layer's visit: rewind its slot's bump pointer."""
        self._arr_used[self._slot_of(layer_idx)] = 0

    def arrival_index(self, layer_idx: int, nbytes: int):
        """Bump `nbytes` (256-B aligned) off this slot's attention-index
        scratch → (numpy uint8 view, CUDA uint8 view) over the same bytes,
        so the host write is the upload: no copy, no H2D node, no sync."""
        i = self._slot_of(layer_idx)
        n = (int(nbytes) + 255) & ~255
        off = self._arr_used[i]
        self._arr_used[i] = off + n
        np_v, t_v = self._arr_idx[i]
        return np_v[off:off + n], t_v[off:off + n]

    def stats(self, layer_idx: int, rows: int):
        """(mean, rstd) fp32 [>= rows, 1] buffers for a LayerNorm of `rows`
        rows into layer `layer_idx`'s activation buffer — allocated on first
        use and grown to the longest span seen. Write-only:
        `native_layer_norm.out` needs out1/out2 targets, and nothing reads
        them back.

        Sized by the LN's own row count, not the activation buffer's: the
        activation buffer holds the whole batch's context, while one call
        norms one span. The caller
        slices to exactly `rows` — aten resizes an oversized `out` in place,
        which would shrink this pair behind the growth check below."""
        i = self._slot_of(layer_idx)
        s = self._stats.get(i)
        if s is None or s[0].shape[0] < rows:
            s = (torch.empty((rows, 1), device="cuda", dtype=torch.float32),
                 torch.empty((rows, 1), device="cuda", dtype=torch.float32))
            self._stats[i] = s
        return s

    def act_buffer_rows(self, layer_idx: int, t0: int, t1: int,
                    row_bytes: int, dtype) -> "torch.Tensor":
        """Rows [t0, t1) of layer `layer_idx`'s activation buffer as a CUDA
        [n, d] view — the x-wheel's addressing (row i at byte i·row_bytes
        from the activation buffer's start; prefill has no reload placed in
        front)."""
        r = self._act_buffers[self._slot_of(layer_idx)]
        off = t0 * row_bytes
        nb = (t1 - t0) * row_bytes
        _elem = dtype.itemsize
        return (r.t_buf[off:off + nb].view(dtype)
                .view(t1 - t0, row_bytes // _elem))


def alloc_layer_buffer(nbytes: int, alloc_aligned) -> LayerBuffer:
    """One-shot UMA buffer: posix_memalign'd numpy via `alloc_aligned`
    (caller supplies; typically `nvme_loader.alloc_aligned` to satisfy
    O_DIRECT alignment), then cudaHostRegister to expose the same
    bytes as a CUDA tensor view. Registered once for the life of the
    process."""
    from vllm.utils.torch_utils import (
        get_accelerator_view_from_cpu_tensor,
    )
    raw = alloc_aligned(nbytes)
    host_t = torch.from_numpy(raw)
    err = torch.cuda.cudart().cudaHostRegister(
        host_t.data_ptr(), host_t.nbytes, 0)
    if err.value != 0:
        # The numeric code matters: 712 is HostMemoryAlreadyRegistered — an
        # earlier buffer's page-granular registration already covers a page
        # this one shares — and that is a different bug from running out of
        # pinnable memory. torch's enum repr prints `???` for codes it does
        # not map, so the number is printed alongside it.
        raise RuntimeError(
            f"cudaHostRegister failed (nbytes={nbytes}, "
            f"ptr%4096={host_t.data_ptr() % 4096}): code={int(err)} ({err})")
    t = get_accelerator_view_from_cpu_tensor(host_t)
    return LayerBuffer(raw_buf=raw, t_buf=t)


class LayerSlotPool:
    """n_slots persistent UMA-aligned slots, cudaHostRegister'd once, and the
    rotation over them.

    The rotation is walked by two pointers over visits, the loader's and the
    writeback's. The loader acquires the next visit's slot once it is released
    (`acquire`), loads it and publishes it (`mark_ready`); the forward pass
    waits for the publication (`await_ready`), runs the layer and stamps its
    layer end on the slot (`stamp_forward_done`); the writeback hears that
    layer end pass, lands the layer's writes and then releases the visit
    (`release_tail`), which releases the slot for its next visit. Every fact
    lives on the slot it is about and has one writer per transition, so the
    facts need no lock; the waits need a signal, and each waiter has one:

      * the loader sleeps on `release_fd` (and the drive's completion fd);
        the writeback writes it when it releases a slot.
      * the forward thread sleeps on `_cv`. Whoever stamps a fact the forward
        thread waits on — the loader's publication, the writeback's release —
        notifies it after the stamp, and the forward thread re-reads the facts
        under it, so a notify cannot be missed. It guards no state."""

    def __init__(self,
                  n_slots: int,
                  weight_buffer_bytes: int,
                  alloc_aligned,
                  n_layers: int):
        from vllm.utils.torch_utils import (
            get_accelerator_view_from_cpu_tensor,
        )
        self._slots: list[LayerSlot] = []
        for i in range(n_slots):
            raw = alloc_aligned(weight_buffer_bytes)
            host_t = torch.from_numpy(raw)
            err = torch.cuda.cudart().cudaHostRegister(
                host_t.data_ptr(), host_t.nbytes, 0)
            if err.value != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed for slot {i}: code={err}")
            t = get_accelerator_view_from_cpu_tensor(host_t)
            # done_seq starts at 0 and forward_done_visit at -1: the first tenant
            # of a slot finds nobody before it.
            self._slots.append(LayerSlot(idx=i, raw_buf=raw, t_buf=t))
        self.n_slots = n_slots
        # Handshake accounting (ELASTIC_LOADER_TIMING): visits the loader
        # acquired, and the forward pass's wait for the loader's publication.
        self.n_acquired = 0      # visits the loader acquired
        self.t_use_wait = 0.0    # forward pass: waiting for mark_ready
        self.n_use = 0
        self._cv = threading.Condition()
        # `_head_visit` is the next visit to be handed out, `_tail_visit` the
        # oldest not yet released; `[_tail_visit, _head_visit)` is exactly
        # what is live. Neither pointer looks at the other's slots and
        # neither searches: both turn their own visit into a slot with
        # `slot_of_visit`.
        self._n_layers = int(n_layers)        # length of the layer -> visit table
        self._head_visit = 0
        self._tail_visit = 0
        # Which visit each layer is living in, published by the loader at
        # `acquire`. The rotation is over visits (`slot_of_visit`), so the
        # layer -> slot map moves by `L mod n_slots` every step and cannot be
        # a formula any reader evaluates on its own; this is the one place it
        # is written and every reader goes through `slot_of_layer`.
        #
        # Entry l+1 is stamped at the same time as l, with the next visit —
        # the loader's visits are consecutive by construction, so that value
        # is exactly what the loader will stamp when it gets there. Writing it
        # early is idempotent, and it is what lets the prefill x-wheel land
        # layer p's output in layer p+1's activation buffer before the loader
        # has acquired p+1.
        #
        # Initialised to visit == layer, which is the first step exactly
        # (the loader starts at visit 0 on layer 0), so the K and V buffers'
        # alias vLLM built at `l mod n_slots` is correct until the loader
        # first rebinds it.
        self._bind = list(range(self._n_layers))
        # Told at every `acquire`: (layer_idx, slot_idx). The K and V buffers
        # are part of the same slot and vLLM binds them per layer, so somebody
        # has to re-point them when the rotation moves — the pool knows when,
        # the streaming loader knows what, so the pool calls and the streaming
        # loader acts.
        self._on_bind = None
        self.n_released = 0
        # Rebinds whose slot is not `layer % n_slots` (the bind hook counts).
        self.n_moved = 0
        # The loader's wake: written when the writeback releases a slot. An
        # eventfd, so a count written while nobody waits is still there when
        # the loader gets to its wait.
        self.release_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    # ---- shared ------------------------------------------------------

    def set_bind_hook(self, fn) -> None:
        """Install `fn(layer_idx, slot_idx)`, called from `acquire` the
        moment a layer's slot is settled and before any byte of it is
        loaded. The K and V buffers rotate with the weight buffer (one
        coupled slot), and vLLM reaches them per layer, so this is where that
        binding is re-pointed."""
        self._on_bind = fn

    def _notify(self) -> None:
        """A slot fact was just stamped; the forward thread re-reads its
        facts."""
        with self._cv:
            self._cv.notify_all()

    def _wake_head(self) -> None:
        os.write(self.release_fd, _EFD_ONE)

    def _bind_layer(self, layer_idx: int, visit: int, slot_idx: int) -> None:
        """Publish `layer_idx -> visit` (and, ahead of time, l+1 -> visit+1)
        and re-point whatever else the slot carries.

        The early l+1 entry is for the x-wheel, which lands layer l's output
        in the next visit's slot before the loader has acquired it. With every
        layer resident (n_slots == L) it is left alone: there the entry being
        replaced can still be the live tenancy the forward pass is about to
        wait on (the loader can be a whole step ahead), and replacing it would
        send that wait to next step's visit — which cannot come until this one
        is done. Nothing is lost by leaving it: the two visits are L apart,
        so with L slots they name the same slot."""
        self._bind[layer_idx] = visit
        _nxt = layer_idx + 1
        if _nxt < self._n_layers and len(self._slots) < self._n_layers:
            self._bind[_nxt] = visit + 1
        self._on_bind(layer_idx, slot_idx)

    def slot_of_visit(self, visit: int) -> int:
        """The rotation, and it is a single modulus: visit v holds slot
        `v mod n_slots`.

        Every pointer over the rotation — the loader that loads, the
        writeback that releases behind it, the forward pass that consumes —
        turns its own visit into a slot with this one function, and all three
        start at visit 0. So the slot sequence is 0, 1, ... n_slots-1, 0, 1,
        ... for the whole run and a slot's reuse distance is n_slots
        everywhere.

        A composed `(v mod L) mod n_slots` would let the inner modulus rewind
        first whenever n_slots does not divide L. With L=32, n_slots=6 the
        visits 26..37 would land on slots 2 3 4 5 0 1 | 0 1 2 3 4 5, so slot 0
        is handed out at visit 30 and wanted again at visit 32 — the loader
        would be only two visits ahead of the writeback at every step
        boundary, and that distance would collapse from 6 to 2 once per
        step."""
        return visit % len(self._slots)

    def slot_of_layer(self, layer_idx: int) -> int:
        """Which slot layer `layer_idx` is in right now.

        The rotation is over visits, so a layer's slot is not a property of
        the layer — it moves by `L mod n_slots` every step. The loader
        publishes the answer when it acquires the slot (`acquire` stamps
        `_bind`), and that is the single authority every other reader uses:
        the forward pass's `await_ready`, the activation buffer, the K and V
        buffers' alias. Nobody searches and nobody keeps a second copy.

        `_bind` holds the visit, not the slot number, so a stale entry cannot
        be mistaken for a live one: the reader compares it against the slot's
        own `acquired_visit` and a leftover from the previous step never
        matches."""
        return self._bind[layer_idx] % len(self._slots)

    def live_visits(self) -> int:
        """Visits live right now: how many visits the loader is ahead of the
        writeback, at most n_slots."""
        return self._head_visit - self._tail_visit

    # ---- loader ------------------------------------------------------

    def acquire(self, visit: int, layer_idx: int) -> "Optional[LayerSlot]":
        """The loader moves onto visit `visit`: if its slot is released — the
        writeback has released the visit before it in that slot — acquire it
        (Alg. 1 line 7): record that the slot now holds this visit, and
        return it; otherwise None, and the loader waits for the writeback's
        release (`release_fd`).

        One integer of the slot's own state is read (has the writeback
        released the previous visit here) and the new visit is written down;
        the forward pass and the writeback find the layer's slot through that
        record."""
        s = self._slots[self.slot_of_visit(visit)]
        if s.released_visit < s.acquired_visit:
            return None
        s.acquired_visit = visit
        self._head_visit = visit + 1
        # The layer's slot is settled — publish it before a single byte is
        # loaded, so the K and V buffers' alias the reload DMAs into is the
        # one this tenancy owns and not the one the previous step left behind.
        self._bind_layer(layer_idx, visit, s.idx)
        self.n_acquired += 1
        return s

    def mark_ready(self, slot: LayerSlot) -> None:
        """The layer's weight bytes have landed in `slot`: publish it on the
        slot's own `published_visit`, a field of its own, so nothing that
        later writes the slot can overwrite a publication."""
        # Order: the fact, then the signal. The forward thread re-reads the
        # fact under the signal's lock, so it either sees the fact or is
        # waiting when the notify comes.
        slot.published_visit = slot.acquired_visit
        self._notify()

    # ---- writeback ---------------------------------------------------

    def tail_slot(self) -> "Optional[tuple[int, LayerSlot]]":
        """The visit the writeback stands at and its slot, or None when the
        writeback has caught up with the loader (nothing live to release)."""
        t = self._tail_visit
        if t >= self._head_visit:
            return None
        return t, self._slots[self.slot_of_visit(t)]

    def release_tail(self) -> None:
        """Release the writeback's visit and step to the next one. The
        caller — the writeback — has heard that visit's layer end pass and
        its writes land. `notify_released` tells the waiters once the
        writeback is done stepping."""
        t = self._tail_visit
        s = self._slots[self.slot_of_visit(t)]
        s.released_visit = t            # the slot is released, and it says so
        self._tail_visit = t + 1
        self.n_released += 1

    def notify_released(self) -> None:
        """One or more visits were just released: the loader may acquire their
        slots, and a forward pass in the zig-zag prefill may be waiting to
        land into one."""
        self._wake_head()
        self._notify()

    # ---- forward pass ------------------------------------------------

    def await_released(self, visit: int) -> None:
        """Block until the slot `visit` will occupy is no longer its previous
        tenant's — the writeback has released visit `visit − n_slots`.

        The forward pass calls this before writing into an activation buffer
        that is not its own layer's: the prefill x-wheel lands layer p's
        output in the next visit's activation buffer. It asks about a visit,
        not a layer entry, so it does not depend on whether `_bind` has been
        stamped ahead for that layer — with every layer resident it is not,
        and a layer entry would name the previous step's tenant, which is the
        slot but not the question."""
        s = self._slots[self.slot_of_visit(visit)]
        prev = visit - len(self._slots)
        with self._cv:
            while s.released_visit < prev:
                self._cv.wait()

    def await_ready(self, layer_idx: int) -> LayerSlot:
        """Wait for the loader's `mark_ready` on layer `layer_idx`, then hand
        back its slot.

        The forward pass takes nothing and mutates nothing here: the tenancy
        and the publication are the loader's. The whole of this call is "has
        layer_idx been published for the visit it is living in" — two integer
        compares on the slot the rotation names.

        The rotation moves, so the slot is re-read every pass: `_bind` is the
        loader's own statement of which visit this layer is living in, and
        until the loader has acquired this step's visit the entry is still the
        previous step's. `acquired_visit == _bind[l]` is what makes a stale entry
        harmless — the slot it names has long since been handed on, so its
        `acquired_visit` is somebody else's number and can never match.

        The rest of what the forward pass consumes is the arrival feed — KV
        blocks as they land — which is `LayerArrival`, not this."""
        t0 = time.perf_counter()
        with self._cv:
            while True:
                _v = self._bind[layer_idx]
                s = self._slots[_v % len(self._slots)]
                if s.acquired_visit == _v and published(s):
                    break
                self._cv.wait()
        self.t_use_wait += time.perf_counter() - t0
        self.n_use += 1
        return s

    def stamp_forward_done(self, slot: LayerSlot, done_seq: int) -> None:
        """The forward pass is done with this layer. `done_seq` is the
        ticket the caller reserved for the layer end and marks on its stream
        right after this returns — the mark is how the writeback hears the
        layer end pass, so the slot is stamped before it can fire.

        Order: `done_seq`, then `forward_done_visit`. The writeback reads
        `forward_done_visit` first and the ticket second, so a forward-done
        stamp it can see always comes with this tenancy's ticket, never the
        previous one's. The forward pass never waits here: it stamps and moves
        on."""
        slot.done_seq = int(done_seq)
        # Nothing clears the publication here: it names a visit, and the
        # loader's next `acquire` moves `acquired_visit` past it, so the next
        # tenancy's arrival is what ends the previous one's publication.
        slot.forward_done_visit = slot.acquired_visit
