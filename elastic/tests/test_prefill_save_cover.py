"""The prefill's two save sites together cover exactly what the reload reads.

A prefill writes a sequence's immutable blocks with the span (the GPU batch)
that produced them (`_make_attn_done_cb` for K/V, `_XWheel.save_span` /
`_capture_ln_save` for ACT — all three enumerate
`_prefill_blocks(mutable=False)`, the rows' blocks with k below `N_seq·n`), and
its mutable prompt block — the block a prompt ends part-way through — at the
end of the layer's compute, with every other sequence's (`_make_post_cb`,
which enumerates `_prefill_blocks(mutable=True)`, the blocks with k at or
above `N_seq·n`). Here n = n_immutable, the immutable blocks per sequence,
and the split is Eqs. 10–11: `k = seq·n + j` for j < n,
`k = N_seq·n + (j − n)·N_seq + seq` for j >= n.

Neither half is checkable on its own: the first decode step reloads every
block its token count says is on disk, so a block written by neither site is
served back as fallocate zeros — plausible tokens, no error anywhere — and a
block written by both is a wasted write whose two sources may disagree. So
what has to hold is the union:

    {immutable blocks of every span} ∪ {mutable prompt blocks at layer end}
        == every sequence's first ⌈S / block_size⌉ blocks

with no k appearing twice and no two writes landing on one offset. The
GPU batch split must not change it either: GPU batches are whole sequences,
so a sequence's immutable blocks are enumerated by exactly one span.

Exhaustive over small shapes, including the ones that make the split awkward:
S below one block, S one block short of the split, S not a multiple of
block_size (a real mutable prompt block) and S an exact multiple (none at
all).
"""

import itertools

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.labels import block_is_act


def _row_ks(lay, sid, n):
    """The global numbers k of a row's `n` tokens' blocks, in position
    order: `ks_of_tokens` at each block's first token."""
    nb = lay.block_split(n)[1]
    return lay.ks_of_tokens(
        sid, np.arange(nb, dtype=np.int64) * lay.block_size)


def _span_prompt_ks(lay, step_rows, t0, t1):
    """The blocks a span writes: its rows' blocks whose k the offset formula
    puts in the IMMUTABLE range of the file, `k < N_seq·n` — what
    `_prefill_blocks(mutable=False)` selects. Spelled here rather than
    imported, so the test states the rule instead of replaying it."""
    n_seq, n_immutable = lay.n_seq, lay.n_immutable
    out = []
    for (row_start, sid, _base, n) in step_rows:
        if not (t0 <= row_start < t1):
            continue
        ks = _row_ks(lay, sid, n)
        out += [int(k) for k in ks if k < n_seq * n_immutable]
    return out

BS = 16
_SHAPES = list(itertools.product((1, 2, 3, 6), (15, 16, 17, 31, 33, 64, 65),
                                 (0.0, 0.5, 0.73, 1.0)))


def _layer_end_tails(lay, step_rows):
    """The blocks `_make_post_cb`'s prefill branch writes at the layer end:
    its rows' blocks whose k the offset formula puts in the MUTABLE range,
    `k >= N_seq·n` — a row's mutable prompt block, and nothing when the prompt
    ends on a block boundary. The same split as the span's, read from the
    other side; no test of whether a block is full."""
    n_seq, n_immutable = lay.n_seq, lay.n_immutable
    tails = []
    for (_row_start, sid, _base, n) in step_rows:
        ks = _row_ks(lay, sid, n)
        tails += [(sid, int(k)) for k in ks if k >= n_seq * n_immutable]
    return tails


def _layout(n_seq, S, alpha):
    lay = BlockKVLayout(n_seq=n_seq, num_layers=1,
                        max_blocks_per_seq=(S + 40 + BS - 1) // BS,
                        block_size=BS, num_heads=2, head_dim=8,
                        dtype=torch.float16, alpha=[alpha])
    lay.set_prompt_len(S)
    return lay


def _prefill_writes(lay, step_rows, n_gpu_batches):
    """(k of the activation blocks, k of the KV blocks) the two save sites
    emit for one whole prefill, with the batch cut into `n_gpu_batches`
    whole-sequence GPU batches."""
    n_seq = lay.n_seq
    S = step_rows[0][3]
    al = lay.alpha_of_layer(0)
    act, kv = [], []
    # The spans: each GPU batch's immutable blocks, the split the seams use.
    edges = [round(i * n_seq / n_gpu_batches) * S for i in range(n_gpu_batches + 1)]
    for t0, t1 in zip(edges[:-1], edges[1:]):
        for k in _span_prompt_ks(lay, step_rows, t0, t1):
            (act if block_is_act(int(k), al) else kv).append(int(k))
    # The layer end: the mutable prompt blocks, over the whole batch at once.
    for _sid, k in _layer_end_tails(lay, step_rows):
        (act if block_is_act(int(k), al) else kv).append(int(k))
    return act, kv


@pytest.mark.parametrize("n_seq,S,alpha", _SHAPES)
@pytest.mark.parametrize("n_gpu_batches", (1, 2))
def test_the_two_save_sites_tile_the_prompt_exactly(n_seq, S, alpha, n_gpu_batches):
    lay = _layout(n_seq, S, alpha)
    step_rows = [(p * S, p, 0, S) for p in range(n_seq)]
    act, kv = _prefill_writes(lay, step_rows, min(n_gpu_batches, n_seq))
    written = sorted(act + kv)
    # What the first decode step will ask the loader to reload.
    n = (S + BS - 1) // BS
    on_disk = sorted(np.concatenate(
        [lay.ks_of_tokens(s_, np.arange(n, dtype=np.int64) * BS)
         for s_ in range(n_seq)]).tolist())
    assert written == on_disk, (
        f"n_seq={n_seq} S={S} α={alpha} mb={n_gpu_batches}: "
        f"missing={sorted(set(on_disk) - set(written))[:8]} "
        f"extra={sorted(set(written) - set(on_disk))[:8]} "
        f"dups={len(written) - len(set(written))}")


@pytest.mark.parametrize("n_seq,S,alpha", _SHAPES)
def test_no_two_prefill_writes_land_on_one_offset(n_seq, S, alpha):
    """Each k resolves through the block region of its own form, so a form
    routed the wrong way is a collision rather than a silent overwrite."""
    lay = _layout(n_seq, S, alpha)
    step_rows = [(p * S, p, 0, S) for p in range(n_seq)]
    act, kv = _prefill_writes(lay, step_rows, 1)
    offs = ([lay.act_offset(0, k) for k in act]
            + [lay.k_offset(0, k) for k in kv])
    assert len(set(offs)) == len(offs)
    kb = lay.block_bytes
    assert all(o % kb == 0 for o in offs)


@pytest.mark.parametrize("n_seq,S", [(3, 33), (6, 65), (2, 17)])
def test_a_mutable_prompt_block_exists_exactly_when_the_prompt_is_ragged(n_seq, S):
    """The split's whole point: immutable blocks in the immutable range, the
    ragged last block (the mutable prompt block) in the mutable range at
    `k = N_seq·n + seq` — N_seq adjacent numbers, so the layer-end batch is one
    contiguous run per block region."""
    lay = _layout(n_seq, S, 0.5)
    step_rows = [(p * S, p, 0, S) for p in range(n_seq)]
    tails = _layer_end_tails(lay, step_rows)
    if S % BS == 0:
        assert tails == []
        return
    assert [sid for sid, _k in tails] == list(range(n_seq))
    n = lay.n_immutable
    assert [k for _sid, k in tails] == [n_seq * n + s for s in range(n_seq)]
    ks = [k for _sid, k in tails]
    assert ks == list(range(ks[0], ks[0] + n_seq))   # N_seq adjacent numbers k


@pytest.mark.parametrize("n_seq,S", [(4, 65), (2, 33)])
def test_a_span_never_emits_a_partial_block(n_seq, S):
    """The span writes the immutable range only, and a row's unfilled last
    block (its mutable prompt block) is not in it — the offset formula put
    that k in the mutable range. So the split needs no test of whether a
    block is full."""
    lay = _layout(n_seq, S, 1.0)
    step_rows = [(p * S, p, 0, S) for p in range(n_seq)]
    got = sorted(_span_prompt_ks(lay, step_rows, 0, n_seq * S))
    want = sorted(int(k) for s in range(n_seq)
                  for k in lay.ks_of_tokens(
                      s, np.arange(S // BS, dtype=np.int64) * BS))
    assert got == want
    if S % BS:
        assert n_seq * lay.n_immutable not in got  # seq 0's mutable block


@pytest.mark.parametrize("n_seq,S", [(4, 65), (3, 33), (2, 17), (6, 65), (1, 65)])
def test_the_mutable_act_block_is_one_whole_block_from_the_pack(n_seq, S):
    """The mutable prompt block goes out whole, from where the LayerNorm
    wrote it.

    The activation source is a row pack, not a block grid: a sequence holds
    rows `[row_start, row_start+S)`, a stride that is not a block multiple
    when `S % block_size`, so the kb span of its partial block continues
    into the next sequence's first rows. Those bytes go out with it. They
    are never attended — they sit past `seqused_k` — and this sequence
    overwrites them as it appends, so nothing is cut, cleared or padded: one
    source, one whole-block request, and adjacent mutable prompt blocks
    still merge into one.

    test_write_plan_act::
    test_mutable_prompt_blocks_at_the_back_from_their_place checks the
    source `_prefill_blocks` produces.
    """
    lay = _layout(n_seq, S, 1.0)                  # α=1: every block is ACT
    n_full, n_blocks = lay.block_split(S)
    assert n_blocks > n_full                       # these shapes are ragged
    own = S - n_full * BS                          # rows of its own
    assert 0 < own < BS
    kb = lay.block_bytes
    for seq in range(n_seq):
        row0 = seq * S + n_full * BS               # its partial block's first row
        # its own rows end exactly where the next sequence's begin
        assert row0 + own == (seq + 1) * S
        # and the request covers the whole block from there
        assert kb == BS * lay.token_bytes
    # the mutable prompt blocks are N_seq adjacent k, so their requests merge
    ks = [int(lay.ks_of_tokens(s_, (n_blocks - 1) * BS))
          for s_ in range(n_seq)]
    assert ks == list(range(ks[0], ks[0] + n_seq))


@pytest.mark.parametrize("n_seq,S", [(4, 64), (3, 32)])
def test_a_block_aligned_prompt_writes_nothing_at_the_layer_end(n_seq, S):
    """`n_blocks == n_full`: the spans wrote every block, so the layer end
    has no partial block to write — the `+1` never happens."""
    lay = _layout(n_seq, S, 1.0)
    n_full, n_blocks = lay.block_split(S)
    assert (n_full, n_blocks) == (S // BS, S // BS)
    step_rows = [(p * S, p, 0, S) for p in range(n_seq)]
    assert _layer_end_tails(lay, step_rows) == []
