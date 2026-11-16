"""A layer's reload fills the slot's K and V buffers in load (= file-offset)
order: `BlockKVLayout.mem_block`.

  (a) it is a bijection from a layer's N_seq·N global numbers onto blocks
      1..N_seq·N (N = max_blocks_per_seq here; block 0 is vLLM's null block);
  (b) consecutive ranks of one form are consecutive blocks — a run of
      ranks is one contiguous buffer span, so a K/V reload is one DMA and a
      kv_gen launch stores consecutive blocks;
  (c) in the buffer the KV blocks start right after the activation blocks,
      and both are packed;
  (d) the vectorised form agrees with the scalar one, entry for entry, and
      with the layer's `rank_tables`;
  (e) the blocks on disk at a uniform count c of blocks per sequence occupy
      a prefix of each form's buffer blocks — the reason a reload is one DMA
      per block region at every step, not just at the end;
  (f) a sequence's own blocks, taken through the layout's enumeration
      (`ks_of_tokens`), are what the closed form places.
"""

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.labels import (act_rank, block_is_act, kv_rank,
                                    n_act_blocks)


def _layout(n_seq=5, N=7, alphas=(0.0, 0.37, 0.5, 0.73, 1.0)):
    return BlockKVLayout(n_seq=n_seq, num_layers=len(alphas),
                         max_blocks_per_seq=N, block_size=16, num_heads=2,
                         head_dim=8, dtype=torch.float16, alpha=list(alphas))


def _seq_ks(lay, seq, n):
    """The global numbers k of `seq`'s first `n` blocks, in position order:
    `ks_of_tokens` at each block's first token."""
    return lay.ks_of_tokens(
        seq, np.arange(n, dtype=np.int64) * lay.block_size)


def test_mem_block_is_a_bijection_onto_the_buffer():
    lay = _layout()
    n = lay.n_seq * lay.max_blocks_per_seq
    for L in range(lay.num_layers):
        blocks = sorted(lay.mem_block(L, k) for k in range(n))
        assert blocks == list(range(1, n + 1)), (L, blocks[:8])
    assert lay.mem_blocks_needed == n + 1


def test_consecutive_ranks_are_consecutive_blocks():
    lay = _layout()
    n = lay.n_seq * lay.max_blocks_per_seq
    for L in range(lay.num_layers):
        al = lay.alpha_of_layer(L)
        for k in range(n):
            m = lay.mem_block(L, k)
            if block_is_act(k, al):
                assert m == 1 + act_rank(k, al)
            else:
                assert m == lay.mem_kv_base(L) + kv_rank(k, al)
        # The two forms' buffer blocks are packed and adjacent.
        assert lay.mem_kv_base(L) == 1 + n_act_blocks(n, al)


def test_vectorised_form_matches_scalar():
    lay = _layout()
    n = lay.n_seq * lay.max_blocks_per_seq
    k = np.arange(n, dtype=np.int64)
    for L in range(lay.num_layers):
        vec = lay.mem_blocks_np(L, k)
        assert vec.dtype == np.int64
        assert vec.tolist() == [lay.mem_block(L, int(x)) for x in k], L


def test_vectorised_form_matches_rank_tables():
    """`rank_tables` and `mem_blocks_np` are the same rule materialised two
    ways (one as columns, one inline); the prefill's block tables come from
    the latter and the arrival consumer's global numbers from the former, so
    they must agree: ACT rank r is buffer block 1 + r, KV rank r is
    `mem_kv_base + r`."""
    lay = _layout()
    for L in range(lay.num_layers):
        act_k, kv_k = lay.rank_tables(L)
        assert lay.mem_blocks_np(L, act_k).tolist() == \
            list(range(1, 1 + len(act_k))), L
        kvb = lay.mem_kv_base(L)
        assert lay.mem_blocks_np(L, kv_k).tolist() == \
            list(range(kvb, kvb + len(kv_k))), L


@pytest.mark.parametrize("rows", [0, 16 * 3 + 5, 16 * 7])
def test_blocks_on_disk_occupy_a_prefix_of_each_forms_buffer_blocks(rows):
    """When every sequence has c >= n blocks, n = n_immutable (every reload:
    the prompt is on disk before the first decode step), the global numbers
    on disk are `range(N_seq·c)`, and those map onto the first ⌈N_seq·c·α⌉ of
    the activation blocks' buffer blocks and the first N_seq·c − ⌈N_seq·c·α⌉
    of the KV blocks' — two dense prefixes, whatever the split. That is what
    makes the reload of any step one DMA per block region."""
    lay = _layout()
    lay.set_prompt_len(rows)
    n_seq, N = lay.n_seq, lay.max_blocks_per_seq
    for L in range(lay.num_layers):
        al = lay.alpha_of_layer(L)
        for c in range(max(lay.n_immutable, 1), N + 1):
            ks = np.concatenate([_seq_ks(lay, s, c) for s in range(n_seq)])
            assert sorted(ks.tolist()) == list(range(n_seq * c))
            mem = lay.mem_blocks_np(L, ks)
            na = n_act_blocks(n_seq * c, al)
            act = sorted(int(m) for m in mem if m < lay.mem_kv_base(L))
            kv = sorted(int(m) for m in mem if m >= lay.mem_kv_base(L))
            assert act == list(range(1, 1 + na)), (L, c)
            assert kv == list(range(lay.mem_kv_base(L),
                                    lay.mem_kv_base(L) + n_seq * c - na)), (L, c)


def test_a_sequences_blocks_are_where_the_closed_form_puts_them():
    """A sequence's blocks in position order are `ks_of_tokens` at each
    block's first token; their memory blocks are `mem_block` of those k,
    no other table involved, and across the batch every block is placed
    exactly once."""
    lay = _layout()
    lay.set_prompt_len(16 * 3 + 5)
    n_seq, N = lay.n_seq, lay.max_blocks_per_seq
    for L in range(lay.num_layers):
        seen = set()
        for s in range(n_seq):
            ks = _seq_ks(lay, s, N)
            mem = lay.mem_blocks_np(L, ks).tolist()
            assert mem == [lay.mem_block(L, int(k)) for k in ks]
            assert 0 not in mem                   # block 0 is the null block
            assert not (seen & set(mem)), (L, s)
            seen.update(mem)
        assert seen == set(range(1, n_seq * N + 1))

