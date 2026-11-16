"""Per-(layer, block, seq) K/V + ACT storage on NVMe — production order.

The blob is a sequence of per-layer file spaces, and a layer's file space is
three block regions, one per kind of byte the layer produces:

    layer i : [ ACT_i ][ K_i ][ V_i ]

The unit is the block — everywhere, in every block region, for reads and for
writes alike. It is the unit the KV cache allocates in, the unit a reload
DMAs, and the unit the activation buffers hold, so nothing has to translate.
A block whose sequence ended part-way through it is still a whole block with
empty elements, exactly as it is in the cache. Storing the partial rows
instead would leave the block nowhere to grow, and ACT grows during decode
just as K and V do; per-token writes are additionally below the drive's
granule.

Within a block region the blocks sit in the order they are produced, by
global number k (Eqs. 10–11):

    immutable range : seq 0's immutable blocks, then seq 1's, …  — a prefill
                      produces every sequence's whole prompt at once,
                      seq-major
    mutable range   : one block per seq 0..N_seq-1, then the next one per
                      seq, … — a synchronised decode step produces one
                      block per sequence across all sequences at once

Two orders because there are two producers, not because the mutable range is
a special category: it is simply the later offsets. A grown block is
overwritten whole in place, so a step's last N_seq blocks are N_seq adjacent
blocks — one contiguous write per layer per step — and the loader reads the
prefix of global numbers the layer's previous visit wrote, in fixed-size
segments. The immutable range holds only the immutable blocks, n per
sequence (n = ⌊S/block_size⌋, `n_immutable`, with S the prompt length; the
code takes `prompt_len // block_size`); the mutable prompt block — the block
a prompt ends part-way through — is the first mutable block of its sequence
(k = N_seq·n + seq), because that is the block decode keeps overwriting —
it is produced with the prefill and saved once the layer's compute is over,
together with every other sequence's mutable prompt block, as one
N_seq-adjacent batch.

Production order means the producer writes in the order it produces and the
loader reads forward. The global number k is the only block coordinate in the
runtime: a producer takes the global numbers of the tokens it produces
(`ks_of_tokens`), the loader cuts block regions into rank runs, and a
consumer that must know where block k sits asks `seq_of_ks` / `j_of_ks`. The
block regions are compact — whole blocks back to back, nothing reserved —
because a per-sequence berth for the mutable prompt block would leave an
unwritten hole in every sequence's run of blocks and fragment the file, and
what costs read bandwidth is the file's continuity, not the page phase of the
writer's source rows.

Three block regions because ACT is the only kind that makes GPU work
(kv_gen), so the loader dispatches a layer's whole activation block region
first and lets K and V follow. K and V advance as two cursors over their own
block regions, which is what makes a run's K and V land together.

The within-block byte order is the token-major (NHD) KV cache block order, so
a whole-block save or reload is a straight DMA with no repacking; a reader
that assumed another order would silently transpose every block. Activation
blocks store post-LN hidden ([tokens, hidden]).

Every block start is `kb`-grained and `kb` is a page multiple, so every save
and reload satisfies O_DIRECT's offset/length constraints. The file is
posix_fallocate'd to its exact total; see `ensure_exists` for why that is not
optional.

A layer's reload fills the slot's K and V buffers in load order, which is
file-offset order (`mem_block`): block 0 is vLLM's null block, then the
activation block region's ranks (their K,V recomputed by kv_gen), then the
K/V block regions', each in rank order. So a run of ranks read as one file
range lands as one run of buffer blocks — the K/V reload is one DMA per block
region, a kv_gen launch stores consecutive blocks, and nothing is scattered.
A block's memory position is therefore per layer (forms differ by α), and
every consumer that needs it — the prefill's block tables, a step's slot
mappings, a save's source block, a reload's destination — evaluates the same
closed form for its layer.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from elastic.runtime.labels import (act_rank, block_is_act, kv_rank,
                                    n_act_blocks, n_kv_blocks)


@dataclass
class BlockKVLayout:
    """Production-order offset function (see the module docstring).

    `alpha` is the per-layer curve (a list of length num_layers). A block's
    form is a pure function of (k, α), with k the block's global number
    across the layer's whole N_seq-sequence stream, so the block region
    sizes — and hence the file's total — follow from (α curve, geometry)
    alone. Blocks produced during decode continue the same numbering: no
    block changes form, and there is no per-sequence state to pin at the
    prompt boundary.

    The global numbering (Eqs. 10–11) is this class's and nobody else's:

        immutable   k = seq·n + j                      (0 ≤ j < n: the
                                                         seq's immutable
                                                         blocks)
        mutable     k = N_seq·n + (j − n)·N_seq + seq  (j ≥ n: one block
                                                         per seq per block
                                                         index, the mutable
                                                         prompt block first)

    with n the immutable blocks per sequence (`n_immutable`, see the module
    docstring). Callers take the block that holds a given token with
    `ks_of_tokens`, and go back from a global number to its sequence and its
    block index in that sequence with `seq_of_ks` / `j_of_ks`."""
    n_seq: int
    num_layers: int
    max_blocks_per_seq: int
    block_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    alpha: list             # per-layer list[float]

    def __post_init__(self) -> None:
        self._alpha = [float(x) for x in self.alpha]
        kb = self.block_bytes
        n_seq = self.n_seq
        N = self.max_blocks_per_seq
        # Per-layer block counts by form, over the layer's whole
        # N_seq-sequence stream: forms follow the global number k across all
        # N_seq sequences' flattened block stream, not each sequence
        # independently. This is what the solver's aggregate byte accounting
        # assumes (total ACT bytes ∝ α over `n_seq·N` blocks), and it costs
        # the one-sided rounding's worst case once per layer instead of once
        # per sequence.
        self._nkv = [n_kv_blocks(N * n_seq, al) for al in self._alpha]
        self._nact = [n_act_blocks(N * n_seq, al) for al in self._alpha]
        # Block region bases. A layer's file space is [ACT | K | V]; K and V
        # are the same size, so V is reachable from K by one per-layer
        # constant, which is what lets a builder address both from one
        # hoisted tuple.
        self._act_region_base: list[int] = []
        self._k_region_base: list[int] = []
        self._v_delta: list[int] = []
        off = 0
        # Compact block regions: a block region is whole blocks back to back
        # and nothing else — no per-sequence slack, no reserved row piece for
        # the mutable prompt block. That block belongs to the mutable range
        # (set_prompt_len); a row-sized berth for it in the immutable range
        # would leave a per-sequence unwritten hole that fragments the file.
        for L in range(self.num_layers):
            act_bytes = self._nact[L] * kb
            kv_bytes = self._nkv[L] * kb
            self._act_region_base.append(off)
            self._k_region_base.append(off + act_bytes)
            self._v_delta.append(kv_bytes)
            off += act_bytes + 2 * kv_bytes
        self._total = off
        # Split between the two production orders (immutable / mutable), in
        # blocks: zero until the gate declares the prompt at a batch's
        # prefill step (see set_prompt_len). A prompt shorter than one block
        # keeps it at zero, and then every block is mutable (decode-ordered).
        self._prompt_len = 0
        self._n_immutable = 0
        # The batches the gate has declared (`declare_batch`) and the loader
        # has not yet taken (`take_batch`), oldest first: each one's longest
        # prompt and its token count.
        self._batches: deque = deque()
        # Per-layer rank tables over the whole stream (`rank_tables`).
        self._rtab: dict = {}

    # ---- immutable / mutable split ----------------------------------

    def set_prompt_len(self, rows: int) -> None:
        """Declare the per-sequence prompt length in tokens.

        The gate knows this only at the batch's prefill step, which is after
        the blob has been fallocate'd — that is fine, because the total is
        independent of the split; only the global number a (sequence, block)
        gets moves.

        The split is `rows // block_size` immutable blocks. A prompt that
        ends part-way through a block leaves that block — the mutable prompt
        block — to the mutable range, because that is the block decode goes
        on overwriting; keeping it in the immutable range would scatter a
        step's N_seq writes across the whole block region. No berth is held for
        those rows in the immutable range — the block regions are compact;
        the mutable prompt block is written whole into its mutable-range
        position at the end of the layer's compute, with every other
        sequence's mutable prompt block."""
        rows = int(rows)
        n = rows // self.block_size
        if rows == self._prompt_len:
            return
        self._prompt_len = rows
        self._n_immutable = n

    def declare_batch(self, prompt_len: int, gen: int) -> None:
        """The batch's shape, declared by the gate at the batch's prefill
        step, before its forward: its longest prompt and the tokens each
        sequence generates, `gen`. The loader's loop runs on these — a batch
        is its prefill step (weights only; the prefill step samples token 1)
        then decode steps t = 1 .. gen−1, step t reloading the blocks of every
        token up to and including its own, ⌈(prompt_len + t)/bs⌉ per
        sequence.

        Each declaration is its batch's own, and the loader takes them in
        order (`take_batch`) as it leaves each batch's prefill step. A batch
        with no decode step (gen 1) does not hold its forward thread until
        the loader has left that step, so the next batch's prefill step — and
        its declaration — can come first. Two can be queued, never more: the
        next batch's forward runs on the step the loader enters at that take,
        and the declaration after it waits for that forward."""
        self._batches.append((int(prompt_len), int(gen)))

    def take_batch(self) -> tuple[int, int]:
        """The oldest declared batch's `(prompt_len, gen)`, taken off the
        queue — the loader's, once per batch, as it leaves the batch's
        prefill step. The declaration is there by then: the loader leaves the
        step only once one of its slots has been released, which is after the
        batch's forward ran that layer."""
        return self._batches.popleft()

    @property
    def n_immutable(self) -> int:
        """Immutable blocks per sequence (n in Eqs. 10–11) — the immutable /
        mutable split."""
        return self._n_immutable

    # ---- the global numbering (the only place j exists) -----------------

    def block_split(self, n_tokens: int) -> "tuple[int, int]":
        """`(n_full, n_blocks)` for a sequence holding `n_tokens`: how many
        of its blocks are complete, and how many it has at all.

        One division names both. `n_full = n_tokens // block_size` is the
        count of complete blocks, and the sequence has one block more than
        that exactly when those blocks do not hold all its tokens. So the
        sequence's blocks `0 .. n_full−1` are the complete ones (the ones a
        GPU batch's span writes densely at the front of its block region),
        and when `n_blocks > n_full` block index `n_full` is the partial
        one."""
        n_full = int(n_tokens) // self.block_size
        n_blocks = n_full + (1 if n_full * self.block_size < n_tokens else 0)
        return n_full, n_blocks

    def ks_of_tokens(self, seq, pos):
        """The global number k of the block that holds position `pos` of
        `seq`, for int64 numpy arrays (broadcast), with
        `j = pos // block_size`: immutable `k = seq·n + j` (0 ≤ j < n),
        mutable `k = N_seq·n + (j − n)·N_seq + seq` (j ≥ n)."""
        seq = np.asarray(seq, dtype=np.int64)
        j = np.asarray(pos, dtype=np.int64) // self.block_size
        n = self._n_immutable
        n_seq = self.n_seq
        return np.where(j < n, seq * n + j, n_seq * n + (j - n) * n_seq + seq)

    def seq_of_ks(self, k):
        """The sequence that owns block `k` (int64 numpy, vectorised):

        `k // n` in the immutable range, `(k − N_seq·n) mod N_seq` in the
        mutable range. The arrival consumer turns the global numbers of a
        delivered rank run into query rows with it."""
        k = np.asarray(k, dtype=np.int64)
        n = self._n_immutable
        n_seq = self.n_seq
        k_mut = n_seq * n       # first mutable global number, N_seq·n
        n_div = max(n, 1)       # np.where evaluates both branches; n may be 0
        return np.where(k < k_mut, k // n_div, (k - k_mut) % n_seq)

    def j_of_ks(self, k):
        """Block index of block `k` INSIDE its own sequence (int64 numpy,
        vectorised) — `seq_of_ks`' twin on the other axis, so that
        `ks_of_tokens(seq_of_ks(k), j_of_ks(k) · bs) == k`.

        `k mod n` in the immutable range, `n + (k − N_seq·n) // N_seq` in the
        mutable range. The arrival consumer turns a delivered block's k into
        the row's block index with it, and a prefill save into its source
        row."""
        k = np.asarray(k, dtype=np.int64)
        n = self._n_immutable
        n_seq = self.n_seq
        k_mut = n_seq * n       # first mutable global number, N_seq·n
        n_div = max(n, 1)       # np.where evaluates both branches; n may be 0
        return np.where(k < k_mut, k % n_div, n + (k - k_mut) // n_seq)

    def rank_tables(self, layer: int):
        """Layer `layer`'s form rule materialised over the whole stream
        `[0, n_seq·max_blocks_per_seq)`: `(act_k, kv_k)` as int64 numpy
        arrays — `act_k[r]` is the k at rank_ACT r, `kv_k[r]` the k at
        rank_KV r. Built once per layer.

        The arrival consumer turns a delivered rank run back into global
        numbers k with these. Same ceil arithmetic as `labels`, so the
        columns cannot disagree with the closed form."""
        ent = self._rtab.get(layer)
        if ent is None:
            n = self.n_seq * self.max_blocks_per_seq
            ca = np.ceil(np.arange(n + 1, dtype=np.float64)
                         * self._alpha[layer]).astype(np.int64)
            ia = ca[1:] > ca[:-1]
            ent = (np.flatnonzero(ia), np.flatnonzero(~ia))
            self._rtab[layer] = ent
        return ent

    # cached_property (not property): these are immutable layout constants,
    # but the offset methods consult them once per block on a hot path. A
    # plain property would re-run the body every time — including a
    # torch.empty() allocation just to read an element size.
    @cached_property
    def elem_bytes(self) -> int:
        return torch.empty([], dtype=self.dtype).element_size()

    @cached_property
    def token_bytes(self) -> int:
        """Bytes for one token's K (or V): num_heads · head_dim · elem_bytes.
        This is the contiguous unit for the activation hidden ([tokens,
        hidden]) and for sizing blocks in the block regions."""
        return self.num_heads * self.head_dim * self.elem_bytes

    @cached_property
    def block_bytes(self) -> int:
        """One block's K (or V) bytes: block_size · token_bytes. Also
        == one activation block's hidden bytes (hidden_size =
        num_heads·head_dim). The layout's `kb` unit — every block of every
        block region is one of these."""
        return self.block_size * self.token_bytes

    @cached_property
    def total_bytes(self) -> int:
        return self._total

    def alpha_of_layer(self, layer: int) -> float:
        return self._alpha[layer]

    # ---- memory position: buffer blocks filled in load (= offset) order ----

    @property
    def mem_blocks_needed(self) -> int:
        """Buffer blocks the closed form addresses: vLLM's null block plus
        one per (sequence, block) of the layer's whole stream."""
        return 1 + self.n_seq * self.max_blocks_per_seq

    def mem_act_base(self) -> int:
        """Buffer block where every layer's activation blocks start in the
        K and V buffers: right after vLLM's null block. rank_ACT r is at
        `mem_act_base + r`, so a run of consecutive activation-block ranks is
        a run of consecutive buffer blocks — which is what lets a kv_gen
        launch take one base block."""
        return 1

    def mem_kv_base(self, layer: int) -> int:
        """Buffer block where the layer's KV blocks start in the K and V
        buffers: after the null block and every activation block of the run
        (all that the layer's activation block region holds), so it is one
        fixed block for the run."""
        return 1 + self._nact[layer]

    def mem_block(self, layer: int, k: int) -> int:
        """Buffer block of block `k` of `layer` — the same (form, rank)
        closed form as the file offset, one block per rank, ranks
        consecutive."""
        al = self._alpha[layer]
        if block_is_act(k, al):
            return 1 + act_rank(k, al)
        return 1 + self._nact[layer] + kv_rank(k, al)

    def mem_blocks_np(self, layer: int, k):
        """Vectorised `mem_block` over an int64 numpy array of global numbers
        k (same ceil arithmetic as `labels`, on the same α)."""
        al = self._alpha[layer]
        k = np.asarray(k, dtype=np.int64)
        ca = np.ceil(k.astype(np.float64) * al).astype(np.int64)
        cb = np.ceil((k + 1).astype(np.float64) * al).astype(np.int64)
        return np.where(cb > ca, 1 + ca, 1 + self._nact[layer] + (k - ca))

    # ---- file offsets of one block, by global number k -------------------

    def k_offset(self, layer: int, k: int) -> int:
        """File offset of KV block `k`'s K copy (V is `+ v_delta(layer)`)."""
        return (self._k_region_base[layer]
                + kv_rank(k, self._alpha[layer]) * self.block_bytes)

    def v_delta(self, layer: int) -> int:
        """Bytes from a block's K copy to its V copy — the V block region
        mirrors the K block region block for block, one per-layer constant
        further on."""
        return self._v_delta[layer]

    def act_offset(self, layer: int, k: int) -> int:
        """File offset of activation block `k`'s hidden copy."""
        return (self._act_region_base[layer]
                + act_rank(k, self._alpha[layer]) * self.block_bytes)

    def region_bases(self, layer: int) -> "tuple":
        """Hot-path constants of one layer's block regions:
        `(act_base, k_base, v_delta)`.

        A block's form and offset both derive from its global number k
        (`elastic.runtime.labels`):

            is_act = block_is_act(k, alpha)
            offset = region_base(is_act) + (act_rank if is_act else kv_rank)(k, alpha) · kb

        V is `k_off + v_delta`. All of it is fixed for the whole run: α is
        fixed (`alpha_of_layer`), so the forms and block region sizes are.
        Builders take this tuple once per layer purely to avoid re-fetching
        the constants on every block; `rank_tables` is the same rule as
        arrays."""
        return (self._act_region_base[layer], self._k_region_base[layer],
                self._v_delta[layer])


class BlockKVStorage:
    def __init__(self, path: Path, layout: BlockKVLayout):
        self.path = Path(path)
        self.layout = layout
        # Every saved unit in this layout is an aligned block, so the NVMe
        # engine saves and reloads it straight to and from NVMe with O_DIRECT,
        # bypassing the page cache: a buffered save would fill memory with
        # dirty KV pages during prefill, and because CPU and GPU share that
        # memory the kernel's page-cache writeback would compete with the
        # forward pass for the same bandwidth.

    # ---- file lifecycle ---------------------------------------------

    def ensure_exists(self) -> None:
        if (not self.path.exists()
                or self.path.stat().st_size < self.layout.total_bytes):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as f:
                # fallocate — a REAL extent reservation, never a sparse
                # truncate. With a sparse file the extents are assigned by
                # whichever write touches a range first, and the writes
                # arrive at scattered offsets, so the blob ends up
                # split across thousands of fragments. Reload then reads a
                # logically sequential range as physically scattered I/O and
                # loses a large fraction of the drive's sequential bandwidth.
                # Preallocating in one call makes the file's physical layout
                # match its logical one.
                os.posix_fallocate(f.fileno(), 0, self.layout.total_bytes)


def _uma_byte_view(t: torch.Tensor):
    """Byte view (ctypes array, no copy) of a contiguous tensor in
    host-accessible shared memory, handed to the NVMe engine as an iovec.

    Ordering: the caller ensures pending GPU writes are visible."""
    import ctypes
    nbytes = t.element_size() * t.numel()
    return (ctypes.c_byte * nbytes).from_address(t.data_ptr())
