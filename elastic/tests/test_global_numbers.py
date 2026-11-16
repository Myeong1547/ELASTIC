"""The enumeration of global numbers is the layout's and nobody else's:
`BlockKVLayout.ks_of_tokens` hands a producer the global numbers k of the
blocks its tokens land in, `seq_of_ks` / `j_of_ks` give a consumer
the sequence that owns a block k and the block's index inside it; "on disk"
is a prefix of the global numbers (every sequence steps together), so a
reader needs no set. The global number k is the only block coordinate the
runtime carries, so these closed forms are the whole contract between the
three of them.

Pinned here, exhaustively over tiny geometries (n = n_immutable, the
immutable blocks per sequence, as in Eqs. 10–11):

  * a sequence's blocks in position order have strictly ascending k;
  * the closed form itself — `k = seq·n + j` for j < n (immutable blocks),
    `k = N_seq·n + (j − n)·N_seq + seq` for j >= n (mutable blocks) — so the
    prompt is seq-major (one run per sequence) and a step's last N_seq blocks
    are N_seq adjacent global numbers (one write per layer per step);
  * block n of every sequence has k = `N_seq·n + seq`, the sequence's first
    mutable block — the mutable prompt block when S % bs != 0, else the
    first block decode creates;
  * every token of a block maps to that block's k, and `ks_of_tokens`
    is the closed form elementwise and broadcasts;
  * uniform counts c cover exactly `range(N_seq·c)` (no gaps, no
    duplicates) and `seq_of_ks` / `j_of_ks` invert the enumeration number
    for number;
  * a re-declared split moves the enumeration with it (every batch declares
    its own).

None of this depends on α: which global numbers a sequence owns is
geometry, which form a block k has is the layer's rule
(`test_rank_tables.py`). The layouts here still carry a full α curve so the
same instances also exercise the form-independent paths at every α.
"""

import math

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout

_ALPHAS = (0.0, 0.37, 0.5, 0.73, 1.0)

# (n_seq, N, bs, rows), N = max_blocks_per_seq here: the immutable/mutable
# split n = rows // bs (n_immutable). Covers n == 0 (no prompt declared, and a
# prompt shorter than one block), n not a multiple of n_seq, n == n_seq,
# n == N (the prompt fills every block), n_seq == 1, and a block size other
# than 16.
_GEOMS = [
    (4, 10, 16, 0),
    (4, 10, 16, 5),
    (4, 10, 16, 97),
    (4, 10, 16, 160),
    (3, 7, 16, 37),
    (5, 8, 16, 80),
    (1, 6, 16, 33),
    (6, 9, 4, 13),
]


def _layout(n_seq, N, bs, rows, alphas=_ALPHAS):
    lay = BlockKVLayout(n_seq=n_seq, num_layers=len(alphas),
                        max_blocks_per_seq=N, block_size=bs, num_heads=2,
                        head_dim=8, dtype=torch.float16, alpha=list(alphas))
    lay.set_prompt_len(rows)
    assert lay.n_immutable == rows // bs
    return lay


def _expected_ks(seq, c, n_seq, n_immutable):
    """The contract's closed form, written out independently of the class."""
    return [seq * n_immutable + j if j < n_immutable
            else n_seq * n_immutable + (j - n_immutable) * n_seq + seq
            for j in range(c)]


def _seq_ks(lay, seq, c):
    """The global numbers k of `seq`'s first `c` blocks, in position order,
    through the layout: `ks_of_tokens` at each block's first token."""
    return lay.ks_of_tokens(
        seq, np.arange(c, dtype=np.int64) * lay.block_size)


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_a_sequences_global_numbers_ascend_in_position_order(n_seq, N, bs, rows):
    lay = _layout(n_seq, N, bs, rows)
    for s in range(n_seq):
        full = _seq_ks(lay, s, N)
        assert full.dtype == np.int64 and full.shape == (N,)
        assert np.all(np.diff(full) > 0), (s, full)
    assert _seq_ks(lay, 0, 0).shape == (0,)
    assert _seq_ks(lay, 0, 0).dtype == np.int64


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_ks_of_tokens_closed_form(n_seq, N, bs, rows):
    lay = _layout(n_seq, N, bs, rows)
    n = lay.n_immutable
    for s in range(n_seq):
        assert _seq_ks(lay, s, N).tolist() == _expected_ks(s, N, n_seq, n)


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_a_prompt_is_one_run_of_consecutive_global_numbers(n_seq, N, bs, rows):
    """Immutable blocks are numbered seq-major: a prefill produces a whole
    sequence's immutable blocks at once, and they are one run of global
    numbers."""
    lay = _layout(n_seq, N, bs, rows)
    n = lay.n_immutable
    for s in range(n_seq):
        assert _seq_ks(lay, s, n).tolist() == list(range(s * n, s * n + n))


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_a_steps_last_n_seq_blocks_are_n_seq_adjacent_global_numbers(
        n_seq, N, bs, rows):
    """A synchronised decode step produces one block per sequence, and the
    N_seq of them are N_seq adjacent global numbers — one contiguous write per
    layer per step, and one N_seq-adjacent batch for the mutable prompt
    blocks."""
    lay = _layout(n_seq, N, bs, rows)
    n = lay.n_immutable
    for j in range(n, N):
        wave = sorted(int(lay.ks_of_tokens(s, j * bs)) for s in range(n_seq))
        w0 = n_seq * n + (j - n) * n_seq
        assert wave == list(range(w0, w0 + n_seq)), j


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_block_n_is_the_first_mutable_global_number(n_seq, N, bs, rows):
    lay = _layout(n_seq, N, bs, rows)
    n = lay.n_immutable
    if n >= N:
        pytest.skip("the prompt fills every block: there is no tail")
    tails = [int(lay.ks_of_tokens(s, n * bs)) for s in range(n_seq)]
    for s in range(n_seq):
        assert tails[s] == n_seq * n + s
        assert tails[s] not in _seq_ks(lay, s, n).tolist()
    # The batch's N_seq blocks at index n are one N_seq-adjacent run.
    assert tails == list(range(n_seq * n, n_seq * n + n_seq))


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_every_token_maps_to_the_block_holding_it(n_seq, N, bs, rows):
    """Token `n_tokens − 1` sits in block ⌈n_tokens / bs⌉ − 1, so every
    token of a block shares that block's global number."""
    lay = _layout(n_seq, N, bs, rows)
    n = lay.n_immutable
    for s in range(n_seq):
        blocks = _expected_ks(s, N, n_seq, n)
        for n_tokens in range(1, N * bs + 1):
            nb = math.ceil(n_tokens / bs)
            assert int(lay.ks_of_tokens(s, n_tokens - 1)) == \
                blocks[nb - 1], (s, n_tokens)


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_ks_of_tokens_is_the_closed_form_elementwise(n_seq, N, bs, rows):
    lay = _layout(n_seq, N, bs, rows)
    n = lay.n_immutable
    seq = np.repeat(np.arange(n_seq, dtype=np.int64), N * bs)
    pos = np.tile(np.arange(N * bs, dtype=np.int64), n_seq)
    got = lay.ks_of_tokens(seq, pos)
    assert got.dtype == np.int64 and got.shape == seq.shape
    exp = [_expected_ks(int(s), int(p) // bs + 1, n_seq, n)[-1]
           for s, p in zip(seq, pos)]
    assert got.tolist() == exp
    # Broadcasts: one sequence, many positions.
    one = lay.ks_of_tokens(np.int64(n_seq - 1), pos[:N * bs])
    assert one.tolist() == exp[(n_seq - 1) * N * bs:]


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_uniform_counts_cover_a_gapless_global_number_prefix(n_seq, N, bs, rows):
    """Once every sequence has its immutable blocks (c >= n — every reload,
    since a reload happens after the prefill), a uniform count c of blocks per
    sequence is exactly the global numbers `range(N_seq·c)`: no gap a reader
    would have to skip, no number owned twice. Below the split the immutable
    blocks are seq-major, so the union is each sequence's stretch
    `seq·n + j`, not a prefix."""
    lay = _layout(n_seq, N, bs, rows)
    n_immutable = lay.n_immutable
    for c in range(N + 1):
        union = sorted(np.concatenate(
            [_seq_ks(lay, s, c) for s in range(n_seq)]).tolist())
        if c >= n_immutable:
            assert union == list(range(n_seq * c)), c
        else:
            assert union == sorted(s * n_immutable + j for s in range(n_seq)
                                   for j in range(c)), c
        assert len(set(union)) == n_seq * c


@pytest.mark.parametrize("n_seq,N,bs,rows", _GEOMS)
def test_seq_of_ks_inverts_the_enumeration(n_seq, N, bs, rows):
    lay = _layout(n_seq, N, bs, rows)
    for s in range(n_seq):
        ks = _seq_ks(lay, s, N)
        owner = lay.seq_of_ks(ks)
        assert owner.dtype == np.int64
        assert owner.tolist() == [s] * N, s
    # And over the whole stream at once: block k's owner is the one sequence
    # whose blocks include k.
    owner_of = {}
    for s in range(n_seq):
        for k in _seq_ks(lay, s, N).tolist():
            assert k not in owner_of
            owner_of[k] = s
    allk = np.arange(n_seq * N, dtype=np.int64)
    assert lay.seq_of_ks(allk).tolist() == [owner_of[k] for k in range(n_seq * N)]
    assert int(lay.seq_of_ks(np.int64(n_seq * N - 1))) == owner_of[n_seq * N - 1]



def test_no_prompt_split_declared_numbers_every_block_step_major():
    """Before the gate declares the prompt, n == 0 and every block is a
    mutable block, numbered step-major: sequence s owns global numbers s,
    s+N_seq, s+2·N_seq, …"""
    n_seq, N = 4, 10
    lay = _layout(n_seq, N, 16, 0)
    assert lay.n_immutable == 0
    for s in range(n_seq):
        assert _seq_ks(lay, s, N).tolist() == [s + n_seq * j for j in range(N)]
        assert int(lay.ks_of_tokens(s, 0)) == s
    ks = np.arange(n_seq * N, dtype=np.int64)
    assert lay.seq_of_ks(ks).tolist() == (ks % n_seq).tolist()
    # Every sequence holding 3 blocks == the first N_seq·3 global numbers, a
    # prefix.
    assert sorted(np.concatenate(
        [_seq_ks(lay, s, 3) for s in range(n_seq)]).tolist()) == list(range(n_seq * 3))


def test_a_redeclared_split_moves_the_enumeration():
    """Every batch declares its own split, re-declared for a later batch;
    the enumeration follows the new split, and the same split re-declared is
    a no-op."""
    n_seq, N = 4, 10
    lay = _layout(n_seq, N, 16, 97)
    before = _seq_ks(lay, 3, N).tolist()
    lay.set_prompt_len(97)                          # idempotent
    assert _seq_ks(lay, 3, N).tolist() == before
    lay.set_prompt_len(160)
    assert lay.n_immutable == 10
    assert _seq_ks(lay, 2, N).tolist() == list(range(20, 30))
    lay.set_prompt_len(0)
    assert _seq_ks(lay, 2, N).tolist() == [2 + n_seq * j for j in range(N)]


def test_j_of_ks_inverts_the_enumeration():
    """`j_of_ks` is the other half of the inverse: `seq_of_ks` names
    the row, this names the block's index inside it. The arrival consumer
    needs both to turn the global numbers of a delivered run into query
    rows."""
    bs = 4
    for n_seq, n_immutable, N in ((1, 4, 9), (3, 0, 5), (4, 7, 7), (5, 2, 11)):
        lay = _layout(n_seq, N, bs, n_immutable * bs)
        for s in range(n_seq):
            for c in range(N + 1):
                ks = _seq_ks(lay, s, c)
                assert list(lay.j_of_ks(ks)) == list(range(c)), (
                    n_seq, n_immutable, s, c)
                assert list(lay.seq_of_ks(ks)) == [s] * c
