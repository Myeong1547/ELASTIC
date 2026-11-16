"""Arrival-based decode attention consumer.

Called from the ELASTIC seam in vLLM's FlashAttentionImpl.forward. Consumes a
layer's reloaded KV in the loader's bulk-op arrival order via LayerArrival:

    per arrived batch (whatever the loader has pushed at that instant):
        the forward-pass stream waits on the batch's last ACT event  # GPU gate
        one partial attention call over the batch's distinct rows via the
            segmented triton3d kernel (decode qlen=1 attends all its past
            keys, so partition + merge is order-free; the LSE is recovered
            from the segment buffers — see _t3d_partial)
        one merge_attn_states (LSE merging) into the running output and LSE
            of each row's sequence, fp32 accumulators (`running_o`,
            `running_lse`)

The loader reloads the first c blocks of every sequence, the current block
(the mutable block that receives the new token) included, so every block of
a row's context arrives through the feed.

An item names one block region's rank run and nothing else (`ArrivalItem`).
The consumer turns it back into global numbers k with the layout's per-layer
tables (`rank_tables` → `act_k` / `kv_k`), into sequences with `seq_of_ks`,
and into query rows through the step's seq_to_row. The position of a
delivered block k in its row is `layout.j_of_ks(k)`, and the row's current
block — the mutable block that receives the new token — is the block of its
last token (`ks_of_tokens`). A block's buffer block is `1 + r` (rank r in the
activation block region) or `mem_kv_base + r` (rank r in the K/V block
regions): the buffer is filled in load order.

The forward pass decides what and how much it attends: per-partial key counts
come from vLLM's seq_lens (seqused_k, incl. the new token) — bs per block
except the current block's `seq_len − (n_blocks − 1)·bs` — the feed carries
readiness only.

Split-KV + LSE merging is mathematically equal to but not fp-bit identical to a
monolithic flash (floating-point non-associativity; the residual difference is
fp16 rounding noise and does not grow with the number of splits), so the
correctness bar for this path is token-identical end-to-end output against the
vanilla oracle. Two merge rules that bar depends on: never merge a row before
its first real contribution (-inf − -inf = NaN), and accumulate in fp32.
Nothing here falls back to a monolithic call on failure, which would make a
token match meaningless.
"""

from __future__ import annotations

import os as _os
import threading
from typing import Optional, Sequence

import numpy as _np
import torch

from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

from elastic.loader.nvme_loader import EventRing

from .layer_arrival import LayerArrival

# Segmented-softmax scratch for the triton3d kernel.
_NSEG = 16                       # triton_attn.py NUM_PAR_SOFTMAX_SEGMENTS
_segm_cache: dict = {}


def _segm_bufs(T: int, H: int, D: int, dev):
    """Lazy per-(rows, heads, dim) fp32 segment buffers for the segmented
    triton3d kernel, allocated once and reused across calls.

    Sized by the widest call, not the step's rows: `consume` cuts every
    distinct-row sub-batch into calls of at most _MAX_ROWS rows.

    Because they are reused and the kernel only writes the segments its
    context actually reaches, every caller that reads them back (the LSE
    recovery in _t3d_partial) must neutral-fill first (max = -inf,
    expsum = 0); otherwise stale finite values from an earlier call are
    folded into the LSE."""
    T = min(T, _MAX_ROWS)
    key = (T, H, D)
    b = _segm_cache.get(key)
    if b is None:
        dp = 1 << (D - 1).bit_length()
        b = (torch.empty((T, H, _NSEG, dp), dtype=torch.float32, device=dev),
             torch.empty((T, H, _NSEG), dtype=torch.float32, device=dev),
             torch.empty((T, H, _NSEG), dtype=torch.float32, device=dev))
        _segm_cache[key] = b
    return b

_running_cache: dict = {}


def _running_bufs(T: int, H: int, D: int, dev):
    """The visit's fp32 running accumulators `(out [T, H, D], lse [H, T])`,
    allocated once per shape and refilled by the caller at every visit. Reuse
    is ordered by the forward-pass stream: a visit's fill runs after the
    previous visit's last read of them (its output copy)."""
    key = (T, H, D, dev)
    b = _running_cache.get(key)
    if b is None:
        b = (torch.empty((T, H, D), dtype=torch.float32, device=dev),
             torch.empty((H, T), dtype=torch.float32, device=dev))
        _running_cache[key] = b
    return b

# The partial kernel is the segmented triton3d one, run with causal=True
# (q_len=1, so the row attends all of its own keys); the partial's LSE is
# recovered from the segment buffers as logsumexp_seg(m_seg + log e_seg),
# under the neutral-fill rule stated in `_segm_bufs`.
#
# ALiBi: the triton kernel applies the same right-aligned bias convention as
# FA2, so merging a partial run requires the per-(head, row) LSE shift by the
# gap of keys after the run described in `arrival_decode`.


_CU_CACHE: dict = {}


def _cu_seqlens(nb: int, device):
    """`arange(nb+1)` for the decode varlen call. Cached per (nb, device):
    the shape repeats on every call of every layer, so rebuilding it would
    cost a device allocation plus a fill per partial."""
    key = (nb, device)
    cu = _CU_CACHE.get(key)
    if cu is None:
        cu = torch.arange(nb + 1, dtype=torch.int32, device=device)
        _CU_CACHE[key] = cu
    return cu


def _t3d_partial(q_sel, out, key_cache, value_cache, bt, seqused, max_k,
                 scale, alibi_slopes, segm):
    """One partial via the segmented triton3d kernel; returns the LSE
    recovered from the segment buffers as [H, nb] fp32.

    The neutral-fill below (max = -inf, expsum = 0) is mandatory — see
    `_segm_bufs`. It is also the whole mask: a segment the kernel does not
    reach keeps (-inf, 0) and contributes `-inf + log 0 = -inf`, and a segment
    it writes holds at least one key, so a finite max and an expsum >= 1."""
    nb = q_sel.shape[0]
    so, sm, se = segm
    so, sm, se = so[:nb], sm[:nb], se[:nb]
    sm.fill_(float("-inf"))
    se.zero_()
    unified_attention(
        q=q_sel, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=_cu_seqlens(nb, q_sel.device),
        max_seqlen_q=1, seqused_k=seqused, max_seqlen_k=max_k,
        softmax_scale=scale, causal=True, window_size=(-1, -1),
        block_table=bt, softcap=0.0, q_descale=None, k_descale=None,
        v_descale=None, seq_threshold_3D=nb + 1,
        num_par_softmax_segments=_NSEG, softmax_segm_output=so,
        softmax_segm_max=sm, softmax_segm_expsum=se,
        alibi_slopes=alibi_slopes)
    return torch.logsumexp(sm + se.log(), dim=2).T.contiguous()   # [H, nb]

# The varlen call's index operands (query rows, block table, seqused and,
# under ALiBi, the per-row gap) and the K/V and ACT appends' row selections
# are bump-allocated out of the slot's registered index scratch
# (`ActBufferPool.arrival_index`), so the host write is the upload: no copy,
# no H2D node, no sync. The scratch is sized once, from the closed-form bound
# on one visit's calls (`arrival_index_bytes`).

# Partial-call coarsening. LSE merging is order-free, so these are
# correctness-neutral and trade arrival latency for kernel-call size:
#  - Within an arrived batch one row's delivered blocks merge into one
#    partial: the block list is the row's blocks in delivery order with the
#    item carrying its current block last — which the seqused_k = Σ
#    valid-tokens convention requires, since only the last block of a call
#    may be token-partial (`_partial_calls`). Without ALiBi that is
#    exact even across gaps in the row's positions: the kernel applies no
#    positional bias (position is already in the cached K), so a partial over
#    an arbitrary set of a row's keys is position-free — the block table maps
#    call-local positions to buffer blocks and softmax does not care where
#    the keys sat in the row's context. That makes one arrived batch == one
#    varlen call. Under ALiBi the row is split into position-contiguous runs
#    (position = `j_of_ks(k)`): the per-run LSE shift
#    (-m·gap, see arrival_decode) is constant per call only for a
#    contiguous run, so BLOOM keeps per-run partials.
#  - DRAIN_TOKENS: volume trigger — an open feed is not consumed until the
#    queued items cover this many KV tokens (close always flushes), so a
#    partial attention call fires after a fixed amount of arrived context
#    regardless of how the loader split it. Converted to blocks via the
#    layout's block size at the call site. The package sets it before this
#    module loads (elastic/vllm_runtime/__init__.py, 32768).
_PARTIAL_ATTN_TOKENS = int(_os.environ["ELASTIC_ARRIVAL_DRAIN_TOKENS"])
if _PARTIAL_ATTN_TOKENS <= 0:
    raise ValueError(
        f"ELASTIC_ARRIVAL_DRAIN_TOKENS must be > 0, got {_PARTIAL_ATTN_TOKENS}")
#  - MAX_ROWS: row-group cap per varlen call. An arrived batch's distinct-row
#    sub-batch goes out as ceil(rows/MAX_ROWS) calls instead of one:
#    row-disjoint groups, same segments per row, same partial-call timing, so
#    only the kernel batch composition changes and tokens are bit-identical.
#    With the volume-capped partial attention call the call shape is (N_seq
#    rows × DRAIN_TOKENS/N_seq keys), and a very wide, shallow call underuses
#    the kernel.
_MAX_ROWS = int(_os.environ.get("ELASTIC_ARRIVAL_MAX_ROWS", "128"))
if _MAX_ROWS <= 0:
    raise ValueError(f"ELASTIC_ARRIVAL_MAX_ROWS must be > 0, got {_MAX_ROWS}")


def arrival_index_bytes(rows: int, max_blocks: int, block_size: int,
                        alibi: bool) -> int:
    """The most index scratch one visit's `arrival_decode` bumps — what
    `ActBufferPool` allocates per slot, once.

    `rows` sequences (the layout's `n_seq`, at least any step's row count)
    of at most `max_blocks` blocks: a visit delivers at most
    K = rows·max_blocks blocks, each once. Every yield of the feed but the
    last carries at least one DRAIN_TOKENS budget, so a visit has at most
    D = K // budget + 1 arrived batches. A partial names one row and
    at least one delivered block of its context: without ALiBi an arrived
    batch has at most one per row and one distinct-row sub-batch; under ALiBi
    a row splits into position-contiguous runs, so partials are bounded by K
    and an arrived batch has up to max_blocks sub-batches. A sub-batch goes
    out in ceil(n / MAX_ROWS) calls, and a call bumps 8 + 4·maxblk + 4 (+ 4
    gap under ALiBi) bytes per partial, maxblk ≤ max_blocks, each operand
    rounded up to 256. The K/V append and the ACT append each bump 8 bytes per
    row, once per row, in at most D calls."""
    K = rows * max_blocks
    budget = -(-_PARTIAL_ATTN_TOKENS // block_size)
    drains = K // budget + 1
    if alibi:
        parts, subs, n_ops = K, drains * max_blocks, 4
    else:
        parts, subs, n_ops = min(K, rows * drains), drains, 3
    calls = -(-parts // _MAX_ROWS) + subs
    per_part = 8 + 4 * max_blocks + 4 + (4 if alibi else 0)
    return (parts * per_part + calls * n_ops * 255
            + 2 * (rows * 8 + drains * 255))

# Partial-attention calls issued in this process; the units measurement reads
# it to record calls per visit.
STATS = {"flash_calls": 0}


# ---- per-layer arrival context bridge --------------------------------
# Set by the forward pass's per-layer pre_cb from the gate's step rows; read by
# the seam in FlashAttentionImpl.forward. One forward thread runs the layers in
# order, so a thread-local current ctx is race-free. The prefill step
# reloads nothing and attends through vLLM's own call, so there it is None.

_ARR_CTX = threading.local()


class ArrivalCtx:
    __slots__ = ("arrival", "layer_idx", "row_sids", "layout", "seq_lens",
                 "seq_to_row", "block_size", "act_pool", "act_rows")

    def __init__(self, arrival: LayerArrival, layer_idx: int,
                 row_sids: Sequence[int], layout,
                 seq_lens: Sequence[int], seq_to_row: dict, block_size: int,
                 act_pool):
        self.arrival = arrival
        self.layer_idx = layer_idx
        # Row -> sequence id. A row's blocks are not a table: a delivered
        # block k's position in the row is `j_of_ks(k)` and its buffer
        # block is the layout's closed form for this layer — the same
        # position the reload landed it in.
        self.row_sids = row_sids
        self.layout = layout
        self.seq_lens = seq_lens
        self.seq_to_row = seq_to_row
        self.block_size = block_size
        # Activation buffer pool, for the attention-index scratch
        # (`ActBufferPool.arrival_index`).
        self.act_pool = act_pool
        # This decode step's ACT rows, left here by the capture LN:
        # `(activation buffer rows view, per-row activation buffer row index,
        # y)`. They go into their blocks when each block's activation load has
        # completed (`arrival_decode`), which then clears this.
        self.act_rows = None


def set_arrival_ctx(ctx: Optional[ArrivalCtx]) -> None:
    _ARR_CTX.cur = ctx


def get_arrival_ctx() -> Optional[ArrivalCtx]:
    return getattr(_ARR_CTX, "cur", None)


def _partial_calls(batch, act_k, kv_k, mkv: int, layout, s2r, seq_lens,
                    n_blocks, block_size: int, alibi: bool):
    """One arrived batch, turned into the partial attention calls' operands —
    the layout's closed form over the batch's blocks as arrays, never a walk
    per block.

    `batch` is the arrived batch's items in delivery order; an item is one
    block region's rank run, so its blocks are `act_k`/`kv_k` over the run
    (global numbers k, ascending) at buffer blocks `1 + r` / `mkv + r`. `s2r`
    maps a sequence id to its row of this step, `seq_lens`/`n_blocks` are per
    row. A block belongs to row `s2r[seq_of_ks(k)]` at position `j_of_ks(k)`.
    Every delivered block is in its row's context: the loader reads the first
    c blocks of every sequence, every sequence is a row of the step, and
    every row holds c blocks.

    Returns `(t_rows, a_rows, partials)`:
      t_rows   rows whose current block (position n_blocks − 1, the one the
               new token goes into) the batch delivers, in delivery order
               (by item, then by sequence);
      a_rows   those of them whose current block came in an ACT item;
      partials `(row, blocks, ntokens, end_token)` per call operand.
               Without ALiBi one per row: the row's blocks in delivery order,
               except that the item carrying its current block goes last —
               seqused_k = Σ ntokens attends the first Σ ntokens keys, so
               only the last block of a call may be token-partial, and the
               rest may come in any order (the merge is order-free and the
               kernel position-free). Under ALiBi a row is cut into
               position-contiguous runs in position order: the merge's
               per-call LSE shift (−m·gap) is constant only over a contiguous
               run. Rows in the order they first appear in the batch.
               Valid keys per block: block_size, the current block
               `seq_len − (n_blocks − 1)·block_size`; a run's end_token is
               `min(seq_len, (pos_last + 1)·block_size)` (the gap is
               `seq_len − end_token`, flash_merge)."""
    _ks, _mem, _it = [], [], []
    kinds = _np.empty(len(batch), dtype=_np.int64)
    for i, item in enumerate(batch):
        r0 = int(item.rank0)
        n = int(item.n)
        kinds[i] = int(item.kind)
        if item.kind == 0:
            _ks.append(act_k[r0:r0 + n])
            m0 = 1 + r0
        else:
            _ks.append(kv_k[r0:r0 + n])
            m0 = mkv + r0
        _mem.append(_np.arange(m0, m0 + n, dtype=_np.int64))
        _it.append(_np.full(n, i, dtype=_np.int64))
    ks = _np.concatenate(_ks)
    mem = _np.concatenate(_mem)
    it = _np.concatenate(_it)
    seqs = layout.seq_of_ks(ks)
    rows = s2r[seqs]
    # The order the batch is taken in: item by item, a run split by sequence
    # in ascending sequence order.
    grp = it * len(s2r) + seqs
    T = int(n_blocks.shape[0])
    # A row's place among the calls: where it first appears in the batch.
    first = _np.full(T, _np.iinfo(_np.int64).max, dtype=_np.int64)
    _np.minimum.at(first, rows, grp)
    present = _np.flatnonzero(first != _np.iinfo(_np.int64).max)
    row_rank = _np.zeros(T, dtype=_np.int64)
    row_rank[present[_np.argsort(first[present], kind="stable")]] = \
        _np.arange(present.shape[0], dtype=_np.int64)
    pos = layout.j_of_ks(ks)
    nb = n_blocks[rows]
    tail = pos == nb - 1
    _ti = _np.flatnonzero(tail)
    _ti = _ti[_np.argsort(grp[_ti], kind="stable")]
    t_rows = rows[_ti].tolist()
    a_rows = rows[_ti][kinds[it[_ti]] == 0].tolist()
    if alibi:
        order = _np.lexsort((pos, row_rank[rows]))
    else:
        tail_item = _np.full(T, -1, dtype=_np.int64)
        tail_item[rows[_ti]] = it[_ti]
        key = row_rank[rows] * 2 + (it == tail_item[rows])
        order = _np.argsort(key, kind="stable")
    rows, mem, pos, tail = rows[order], mem[order], pos[order], tail[order]
    cut = rows[1:] != rows[:-1]
    if alibi:
        cut |= pos[1:] != pos[:-1] + 1
    starts = _np.concatenate(([0], _np.flatnonzero(cut) + 1))
    ends = _np.append(starts[1:], rows.shape[0])
    sl = seq_lens[rows]
    ntok = _np.where(tail, sl - (nb[order] - 1) * block_size, block_size)
    ntoks = _np.add.reduceat(ntok, starts)
    end_tok = _np.minimum(sl[ends - 1], (pos[ends - 1] + 1) * block_size)
    partials = [(int(rows[a]), mem[a:b], int(nt), int(et))
                for a, b, nt, et in zip(starts.tolist(), ends.tolist(),
                                        ntoks.tolist(), end_tok.tolist())]
    return t_rows, a_rows, partials


def arrival_decode(query: torch.Tensor, output: torch.Tensor,
                   key_cache: torch.Tensor, value_cache: torch.Tensor, *,
                   scale: float, ctx: ArrivalCtx, alibi_slopes=None,
                   append_cb) -> torch.Tensor:
    """Consume layer `ctx.layer_idx`'s KV in arrival order; write the merged
    decode attention into `output` in place (model dtype).

    query/output: [T, H, D], one decode token per row (row order == the gate's
    ctx row order == vLLM's req-index packing).

    The partial kernel is the segmented triton3d one (see _t3d_partial).

    ALiBi is exact. The kernel applies the bias right-aligned to each call's
    own keys, -m·(seqused_k-1-j_local), so a partial over a run with `gap`
    keys after it is biased -m·gap too little on every key — a per-(head, row)
    constant within the call. A constant logit shift leaves the softmax
    weights (and hence the partial's output) untouched and shifts only the
    LSE, so adding c = -m·gap to the partial's LSE before merge_attn_states
    makes the merge exact; merging a partial run without that shift is
    silently wrong under ALiBi.

    K/V append at load completion: the new token goes into the block
    that receives it once that block has arrived, and only then is the block
    attended. When a batch delivers rows' current blocks, `append_cb(rows)`
    (the seam's cache write of the already projected k/v, for those rows
    only) runs on this stream after the batch's wait and before the blocks
    are attended — once per row. vLLM's own cache write before attention is
    not made in a decode step (the seam's `do_kv_cache_update`): the block
    is still being reloaded then.

    ACT append at load completion: a row whose current block is an
    activation block goes into its place in that block in the activation
    buffer (`ctx.act_rows`, left by the capture LN) when the block's
    activation load has completed — right after the item carrying it is taken
    in (the batch's wait on the kv_gen launch's event puts the append after
    kv_gen on this stream, and kv_gen was launched only after the block's
    reload DMA landed). Once per row, and that is the only write of the row
    into the activation buffer; the layer end then saves the block whole."""
    T, H, D = query.shape
    dev = query.device
    arr = ctx.arrival
    layer_idx = ctx.layer_idx
    layout = ctx.layout
    # One visit == one bump arena. Rewinding here (not per call) is what
    # keeps a call's operands alive until its kernel retires; reuse across
    # visits is gated by the slot rotation, like the slot's activation buffer.
    _stg = ctx.act_pool
    _stg.arrival_index_reset(layer_idx)
    block_size = ctx.block_size
    seq_lens = ctx.seq_lens
    # Per row: how many blocks its context has, and its current block —
    # the last of them, the one the new token goes into. Neither needs
    # the row's list of global numbers built out: a delivered block's position
    # in its row is `layout.j_of_ks`, and the current block's k is the global
    # number of the block of the row's last token (`ks_of_tokens`).
    seq_lens_np = _np.asarray(seq_lens, dtype=_np.int64)
    n_blocks_np = -(-seq_lens_np // block_size)
    sids_np = _np.asarray(ctx.row_sids, dtype=_np.int64)
    cur_k = layout.ks_of_tokens(sids_np, seq_lens_np - 1)
    # Sequence id -> row of this step, as an array. Every sequence of the
    # layout is a row of the step.
    s2r = _np.empty(int(layout.n_seq), dtype=_np.int64)
    s2r[_np.fromiter(ctx.seq_to_row.keys(), dtype=_np.int64)] = \
        _np.fromiter(ctx.seq_to_row.values(), dtype=_np.int64)
    # The layer's form rule over the whole stream: rank in a block region →
    # global number k.
    # Cached per layer by the layout, so this is one dict lookup per call.
    act_k, kv_k = layout.rank_tables(layer_idx)
    mkv = int(layout.mem_kv_base(layer_idx))
    # Rows whose current block is an activation block owe one append into
    # the activation buffer (see the docstring); `_act_owed[row]` until it is
    # made.
    _act = ctx.act_rows
    if _act is not None:
        _act_owed = layout.mem_blocks_np(layer_idx, cur_k) < mkv
    # The running output and LSE of each row's sequence, fp32 accumulators
    # for LSE merging; LSE stored [H, T]: a batch's columns gather/scatter
    # as [H, nb], the merge_attn_states LSE layout. Never merged into before
    # their fill.
    running_o, running_lse = _running_bufs(T, H, D, dev)
    running_o.zero_()
    running_lse.fill_(float("-inf"))
    # Segmented-kernel scratch (cached per shape).
    segm = _segm_bufs(T, H, D, dev)
    stream = torch.cuda.current_stream()

    def _maybe_append(t_rows):
        """Put the new token into the current blocks this batch
        delivered: on this stream after the batch's wait, so after each
        block's reload (and, for an activation block, its kv_gen), and before
        the blocks are attended. A block is delivered once, so each row's
        new token goes in once. The block's save leaves at the layer end,
        marked after this."""
        if not t_rows:
            return
        _n = len(t_rows)
        _hv, _dv = _stg.arrival_index(layer_idx, _n * 8)
        _hv[:_n * 8].view(_np.int64)[:] = t_rows
        append_cb(_dv[:_n * 8].view(torch.int64))

    def _act_append(rows):
        """Append these rows' ACT into their blocks in the activation
        buffer: their block's activation load is complete. Forward-pass
        stream, so it is ordered after everything the loader's item stands
        for, and before the layer end that saves the block."""
        rows = _np.asarray(rows, dtype=_np.int64)
        rows = rows[_act_owed[rows]]
        if not rows.shape[0]:
            return
        _act_buf, _ridx, _y = _act
        _n = int(rows.shape[0])
        _hv, _dv = _stg.arrival_index(layer_idx, _n * 8)
        _hv[:_n * 8].view(_np.int64)[:] = rows
        _sel = _dv[:_n * 8].view(torch.int64)
        _act_buf[_ridx.index_select(0, _sel)] = _y.index_select(0, _sel)
        _act_owed[rows] = False
    n_calls = 0

    def flash_merge(sub):
        """One varlen flash over distinct-row partials plus one merge.
        sub = list of (row, blocks, ntokens, end_token)."""
        nonlocal n_calls
        nb = len(sub)
        maxblk = max(int(s[1].shape[0]) for s in sub)
        # 8·nb (qrows i64) + 4·nb·maxblk (block table i32) + 4·nb (seqused
        # i32) + under ALiBi 4·nb (gap f32), each 256-B aligned so the dtype
        # views are legal.
        _o1 = (nb * 8 + 255) & ~255
        _o2 = _o1 + ((nb * maxblk * 4 + 255) & ~255)
        _o3 = _o2 + ((nb * 4 + 255) & ~255)
        _need = _o3 + (((nb * 4 + 255) & ~255)
                       if alibi_slopes is not None else 0)
        _npv, _tv = _stg.arrival_index(layer_idx, _need)
        _q_np = _npv[:nb * 8].view(_np.int64)
        _bt_np = _npv[_o1:_o1 + nb * maxblk * 4].view(_np.int32)
        _su_np = _npv[_o2:_o2 + nb * 4].view(_np.int32)
        _bt_np[:] = 0
        seq_max = 0
        for i, (_row, blks, nt, _end) in enumerate(sub):
            _q_np[i] = _row
            _su_np[i] = nt
            if nt > seq_max:
                seq_max = nt
            # One vectorised store per partial: `blks` is the run of
            # buffer blocks the layout already produced, so the row's
            # cells are a slice, not a scalar loop. The cells past it
            # stay 0 (the zeroing above): the kernel's block-index load
            # is unmasked and reads up to the tile boundary, and 0 is
            # vLLM's null block.
            _base = i * maxblk
            _bt_np[_base:_base + blks.shape[0]] = blks
        qrows = _tv[:nb * 8].view(torch.int64)
        bt = _tv[_o1:_o1 + nb * maxblk * 4].view(
            torch.int32).view(nb, maxblk)
        seqused = _tv[_o2:_o2 + nb * 4].view(torch.int32)
        q = query.index_select(0, qrows)
        out = torch.empty(nb, H, D, dtype=query.dtype, device=dev)
        lse_f = _t3d_partial(q, out, key_cache, value_cache, bt, seqused,
                             seq_max, scale, alibi_slopes, segm)
        o = out
        if alibi_slopes is not None:
            # Exact ALiBi partials (docstring): the kernel biased this run's
            # keys right-aligned to the run, i.e. -m·gap too little, where
            # gap = keys after the run in the row's context. Constant per
            # (head,row) ⇒ out is exact; shift the LSE by c = -m·gap.
            _npv[_o3:_o3 + nb * 4].view(_np.float32)[:] = [
                float(int(seq_lens[s[0]]) - s[3]) for s in sub]
            gap = _tv[_o3:_o3 + nb * 4].view(torch.float32)     # [nb]
            lse_f = lse_f - alibi_slopes.float().unsqueeze(1) * gap.unsqueeze(0)
        pref_out = running_o.index_select(0, qrows)
        pref_lse = running_lse.index_select(1, qrows).contiguous()
        new_out = torch.empty(nb, H, D, dtype=torch.float32, device=dev)
        new_lse = torch.empty(H, nb, dtype=torch.float32, device=dev)
        merge_attn_states(new_out, pref_out, pref_lse, o.float(), lse_f,
                          output_lse=new_lse)
        running_o.index_copy_(0, qrows, new_out)
        running_lse.index_copy_(1, qrows, new_lse)
        n_calls += 1

    def consume(partials):
        """Attend (row, blocks, ntokens, end_token) partials, splitting into
        distinct-row sub-batches (a varlen call takes each row once; a row's
        second run of the same arrived batch goes to the next sub-batch), each
        capped at _MAX_ROWS rows per varlen call (see the knob note at the
        top — row-disjoint groups, bit-identical, kernel-shape only). Each
        group is one varlen call plus one fp32 merge_attn_states (LSE
        merging), at its full width."""
        while partials:
            seen_rows = set()
            sub, rest = [], []
            for p in partials:
                (sub if p[0] not in seen_rows else rest).append(p)
                seen_rows.add(p[0])
            for i in range(0, len(sub), _MAX_ROWS):
                flash_merge(sub[i:i + _MAX_ROWS])
            partials = rest

    # The ELASTIC_ARRIVAL_DRAIN_TOKENS budget, expressed in KV blocks.
    _min_blocks = layout.block_split(_PARTIAL_ATTN_TOKENS)[1]
    # Streaming consumption: attend whatever has arrived at each partial
    # attention call, so the forward pass overlaps the blocks the loader is
    # still loading.
    for batch in arr.arrived_batches(layer_idx, min_blocks=_min_blocks):
        # The batch's ACT items were made on the kv_gen stream; wait once
        # for the last of them (that stream is in order, so it covers the
        # rest) before anything below touches their blocks — the K/V append
        # of the new token, the ACT row append, the partial calls. The
        # wait is on the device; this thread goes on. KV items were handed
        # over after their CQE and carry nothing to wait for.
        _last_ev = None
        for item in batch:
            if item.event is not None:
                _last_ev = item.event
        if _last_ev is not None:
            EventRing.wait(stream, _last_ev)
        _t_rows, _a_rows, partials = _partial_calls(
            batch, act_k, kv_k, mkv, layout, s2r, seq_lens_np, n_blocks_np,
            block_size, alibi_slopes is not None)
        _maybe_append(_t_rows)
        if _a_rows and _act is not None:
            # These rows' current activation blocks have just been taken in:
            # their load is complete, so the rows go in now.
            _act_append(_a_rows)
        consume(partials)

    ctx.act_rows = None
    output.copy_(running_o)
    STATS["flash_calls"] += n_calls
    return output
