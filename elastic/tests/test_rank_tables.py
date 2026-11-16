"""`BlockKVLayout.rank_tables` is the form rule materialised per layer over
the layer's whole stream `[0, n_seq·max_blocks_per_seq)`: the arrival consumer
turns a delivered run of block-region ranks back into global numbers k with
its two inverses.

That only works if the tables agree, entry for entry, with
`elastic.runtime.labels` — the canonical rule that decides where a block
goes. A disagreement is not a slow path, it is a read from the wrong file
offset into the right cache block, which is silent wrong tokens.

So this pins both columns against the closed-form functions themselves:

  act_k[r]  == the k whose act_rank is r   (block_is_act(k), ACT)
  kv_k[r]   == the k whose kv_rank is r    (not block_is_act(k), KV)

The tables are a pure function of (k, α): the prompt split is not a
coordinate of the form, so a re-declared split leaves them as they are.
"""

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.labels import (act_rank, block_is_act, kv_rank,
                                    n_act_blocks, n_kv_blocks)


def _layout(alpha, n, n_seq, N=None, num_layers=1):
    """A layout with `n` immutable blocks per sequence (rows carry a
    remainder so the split is n without being a multiple of anything)."""
    N = N if N is not None else n + 8
    lay = BlockKVLayout(n_seq=n_seq, num_layers=num_layers,
                        max_blocks_per_seq=N, block_size=16, num_heads=2,
                        head_dim=8, dtype=torch.float16,
                        alpha=[alpha] * num_layers)
    lay.set_prompt_len(n * 16 + 5)
    assert lay.n_immutable == n
    return lay


# Covers: pure ACT, pure KV, a rounding-heavy irrational-ish alpha, a
# typical solved curve's mean, no prompt split declared, and a prompt split
# that is not a multiple of N_seq.
CASES = [
    (1.0, 4, 3), (0.0, 4, 3), (0.5, 4, 3), (0.706, 5, 7),
    (0.827, 3, 2), (0.25, 1, 8), (0.9999, 2, 5), (0.001, 6, 4),
    (0.5, 0, 4), (0.706, 0, 3),
]


@pytest.mark.parametrize("alpha,n_immutable,n_seq", CASES)
def test_act_k_and_kv_k_are_the_exact_inverses(alpha, n_immutable, n_seq):
    """Each inverse names every block of its form exactly once, ascending,
    and rank r of a block region is the r-th block of that form."""
    lay = _layout(alpha, n_immutable, n_seq)
    act_k, kv_k = lay.rank_tables(0)
    n = n_seq * lay.max_blocks_per_seq
    for col in (act_k, kv_k):
        assert col.dtype == np.int64
    assert act_k.tolist() == [k for k in range(n) if block_is_act(k, alpha)]
    assert kv_k.tolist() == [k for k in range(n)
                             if not block_is_act(k, alpha)]
    for r, k in enumerate(act_k.tolist()):
        assert act_rank(k, alpha) == r, f"ACT rank {r} -> k {k}"
    for r, k in enumerate(kv_k.tolist()):
        assert kv_rank(k, alpha) == r, f"KV rank {r} -> k {k}"
    assert len(act_k) == n_act_blocks(n, alpha)
    assert len(kv_k) == n_kv_blocks(n, alpha)
    # They also size the layout's block regions: the ACT block region's
    # buffer blocks end where the KV block region's begin, and V is one K
    # block region past K.
    assert lay.mem_kv_base(0) - 1 == len(act_k)
    assert lay.v_delta(0) == len(kv_k) * lay.block_bytes


def test_tables_are_built_once_per_layer():
    lay = _layout(0.5, 4, 3)
    t1 = lay.rank_tables(0)
    t2 = lay.rank_tables(0)
    assert all(a is b for a, b in zip(t1, t2))


def test_redeclared_prompt_split_leaves_the_tables():
    """A re-declared prompt split moves every global number k a sequence
    owns, but the form of block k is (k, α) alone, so the tables stand."""
    lay = _layout(0.5, 4, 3, N=12)
    t0 = [c.copy() for c in lay.rank_tables(0)]
    lay.set_prompt_len(6 * 16)
    assert lay.n_immutable == 6
    assert [c.tolist() for c in lay.rank_tables(0)] == \
        [c.tolist() for c in t0]
    # The numbers k a sequence owns did move with the split.
    assert lay.ks_of_tokens(1, np.arange(6) * 16).tolist() == \
        list(range(6, 12))


def test_per_layer_alpha_gives_per_layer_tables():
    alphas = [0.0, 0.37, 0.5, 0.73, 1.0]
    lay = BlockKVLayout(n_seq=3, num_layers=len(alphas),
                        max_blocks_per_seq=10, block_size=16, num_heads=2,
                        head_dim=8, dtype=torch.float16, alpha=alphas)
    for L, al in enumerate(alphas):
        act_k, kv_k = lay.rank_tables(L)
        assert act_k.tolist() == [k for k in range(30) if block_is_act(k, al)]
        assert kv_k.tolist() == [k for k in range(30)
                                 if not block_is_act(k, al)]
    assert len(lay.rank_tables(0)[0]) == 0           # α = 0: no ACT
    assert len(lay.rank_tables(4)[1]) == 0           # α = 1: all ACT
