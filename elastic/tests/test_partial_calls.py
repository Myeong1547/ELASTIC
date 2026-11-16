"""The arrival consumer's grouping into partial attention calls: which rows
the batch of one partial attention call appends to, and the calls' operands
it builds.

`arrival_attention._partial_calls` turns one arrived batch into
`(t_rows, a_rows, partials)` with the layout's closed form over the batch's
blocks as arrays. The kernel's inputs — each row's block list in order, its
token count, the row order and hence the MAX_ROWS grouping — must be exactly
what the per-block rule below produces, or the tokens drift by fp rounding.
The rule is written out here in its plain per-item, per-sequence form and
the function is held to it on random layouts and batches: immutable and
mutable blocks, activation and KV items, rows whose current block is or is
not in the batch, with and without ALiBi.
"""
import random

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.labels import n_act_blocks
from elastic.vllm_runtime.arrival_attention import _partial_calls
from elastic.vllm_runtime.layer_arrival import ArrivalItem

BS = 16


def _rule(batch, act_k, kv_k, mkv, layout, seq_to_row, seq_lens, sids,
          block_size, alibi):
    """The per-block rule: items in delivery order, each run split by
    sequence in ascending order; a row's entries in item order with the one
    carrying its current block last; under ALiBi the row's blocks in
    position order, cut into position-contiguous runs."""
    T = len(seq_lens)
    n_blocks = [-(-int(seq_lens[r]) // block_size) for r in range(T)]
    cur = [int(layout.ks_of_tokens(int(sids[r]), int(seq_lens[r]) - 1))
             for r in range(T)]
    per_row, row_order, t_rows, a_rows = {}, [], [], []
    for item in batch:
        r0, n = int(item.rank0), int(item.n)
        if item.kind == 0:
            ks, mem0 = act_k[r0:r0 + n], 1 + r0
        else:
            ks, mem0 = kv_k[r0:r0 + n], mkv + r0
        seqs = layout.seq_of_ks(ks)
        order = np.argsort(seqs, kind="stable")
        ss = seqs[order]
        cut = np.flatnonzero(ss[1:] != ss[:-1]) + 1
        for a, b in zip([0] + cut.tolist(), cut.tolist() + [n]):
            row = seq_to_row[int(ss[a])]
            idx = order[a:b]
            kg = ks[idx]
            if bool((kg == cur[row]).any()):
                t_rows.append(row)
                if item.kind == 0:
                    a_rows.append(row)
            if row not in per_row:
                per_row[row] = []
                row_order.append(row)
            per_row[row].append((kg, mem0 + idx))
    partials = []
    for row in row_order:
        nb = n_blocks[row]
        ps, ms, tail = [], [], None
        for kg, mg in per_row[row]:
            pe = layout.j_of_ks(kg)
            if int(pe[-1]) == nb - 1:
                tail = (pe, mg)
            else:
                ps.append(pe)
                ms.append(mg)
        if tail is not None:
            ps.append(tail[0])
            ms.append(tail[1])
        pos, mem = np.concatenate(ps), np.concatenate(ms)
        if alibi:
            o = np.argsort(pos, kind="stable")
            pos, mem = pos[o], mem[o]
            cut = np.flatnonzero(pos[1:] != pos[:-1] + 1) + 1
            bounds = [0] + cut.tolist() + [len(pos)]
        else:
            bounds = [0, len(pos)]
        sl = int(seq_lens[row])
        ntok = np.where(pos < nb - 1, block_size, sl - (nb - 1) * block_size)
        for a, b in zip(bounds[:-1], bounds[1:]):
            partials.append((row, mem[a:b], int(ntok[a:b].sum()),
                             min(sl, (int(pos[b - 1]) + 1) * block_size)))
    return t_rows, a_rows, partials


def _case(rng, alibi):
    n_seq = rng.choice([1, 2, 3, 5, 8, 13])
    P_tok = rng.choice([0, 16, 31, 48, 65, 100])
    gen = rng.randint(1, 40)
    mbps = (P_tok + gen + 15) // BS + 1
    alpha = rng.choice([0.0, 0.219, 1 / 3, 0.5, 0.71, 1.0])
    lay = BlockKVLayout(n_seq=n_seq, num_layers=1, max_blocks_per_seq=mbps,
                        block_size=BS, num_heads=2, head_dim=8,
                        dtype=torch.float16, alpha=[alpha])
    lay.set_prompt_len(P_tok)
    # This step: every sequence of the batch is a row, in some row order,
    # and every row has the same length.
    sids = list(range(n_seq))
    rng.shuffle(sids)
    seq_to_row = {s: r for r, s in enumerate(sids)}
    seq_len = P_tok + rng.randint(1, gen)
    seq_lens = [seq_len] * n_seq
    c = -(-seq_len // BS)
    K = n_seq * c
    na = n_act_blocks(K, alpha)
    act_k, kv_k = lay.rank_tables(0)
    mkv = int(lay.mem_kv_base(0))
    items = []
    for kind, n_block_regions in ((0, na), (1, K - na)):
        r0 = 0
        while r0 < n_block_regions:
            n = min(n_block_regions - r0, rng.randint(1, max(1, 3 * n_seq)))
            items.append(ArrivalItem(kind, r0, n, None))
            r0 += n
    # The batch of one partial attention call is any contiguous slice of the
    # delivery order.
    a = rng.randint(0, len(items) - 1)
    b = rng.randint(a + 1, len(items))
    return (items[a:b], act_k, kv_k, mkv, lay, seq_to_row, seq_lens, sids)


def _same(got, want):
    gt, ga, gp = got
    wt, wa, wp = want
    assert gt == wt
    assert ga == wa
    assert len(gp) == len(wp)
    for (r1, m1, n1, e1), (r2, m2, n2, e2) in zip(gp, wp):
        assert r1 == r2 and n1 == n2 and e1 == e2
        assert np.array_equal(np.asarray(m1), np.asarray(m2))


@pytest.mark.parametrize("alibi", [False, True])
def test_partial_calls_match_rule(alibi):
    rng = random.Random(1234 + int(alibi))
    for _ in range(1500):
        batch, act_k, kv_k, mkv, lay, s2r_d, seq_lens, sids = _case(rng, alibi)
        s2r = np.empty(lay.n_seq, dtype=np.int64)
        for s, r in s2r_d.items():
            s2r[s] = r
        sl = np.asarray(seq_lens, dtype=np.int64)
        got = _partial_calls(batch, act_k, kv_k, mkv, lay, s2r, sl,
                              -(-sl // BS), BS, alibi)
        want = _rule(batch, act_k, kv_k, mkv, lay, s2r_d, seq_lens, sids,
                     BS, alibi)
        _same(got, want)
