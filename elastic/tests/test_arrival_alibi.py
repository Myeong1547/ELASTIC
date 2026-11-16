"""The real `arrival_decode` (streaming LSE merging) against a monolithic FA2
reference, with and without ALiBi — multi-row, with rank runs that span
rows. Needs a GPU.

The items are what the loader pushes: block-region rank runs of the real
`BlockKVLayout` covering the first c blocks of every sequence, whose buffer
blocks are the layout's closed form (filled in load order). The reference
attends each row's whole context through a block table built from the same
closed form, so the two paths agree on where every block sits and differ only
in how the keys are partitioned. At the default partial attention call volume
all items fit one partial call; a one-block call volume makes every item its
own call, which is what exercises the cross-call LSE merging and the ALiBi gap
shift.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from elastic.loader.nvme_loader import EventRing, alloc_aligned
from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.vllm_runtime import arrival_attention
from elastic.vllm_runtime.arrival_attention import (ArrivalCtx,
                                                    arrival_decode,
                                                    arrival_index_bytes)
from elastic.vllm_runtime.layer_arrival import ArrivalItem, LayerArrival
from elastic.vllm_runtime.layer_pool import ActBufferPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a GPU")

dev = "cuda"
H, D, BS = 4, 64, 16
seq_lens = [45, 45, 45]          # rows 0..2 == layout seqs 0..2
T = len(seq_lens)
scale = D ** -0.5

# Registered host buffers live as long as the process: freed while still
# registered, their pages would make a later registration over them fail.
_ACT_POOL_KEEP = []


@pytest.fixture(scope="module")
def case():
    torch.manual_seed(7)
    # Three sequences, three blocks each, α = 0.5 on the single layer.
    # Prompt of 20 rows → one immutable block per seq; every later block is
    # mutable, numbered step-major (one per seq per block index), so the
    # global numbers are
    #   k : 0 1 2 | 3 4 5 | 6 7 8      seq 0 1 2 | 0 1 2 | 0 1 2
    # and α = 0.5 makes k = 0,2,4,6,8 activation blocks (ACT ranks 0..4) and
    # k = 1,3,5,7 KV blocks (KV ranks 0..3). Buffer blocks: 0 = null, ACT
    # rank r → 1 + r, KV rank r → mem_kv_base + r.
    layout = BlockKVLayout(n_seq=3, num_layers=1, max_blocks_per_seq=3,
                           block_size=BS, num_heads=H, head_dim=D,
                           dtype=torch.float16, alpha=[0.5])
    layout.set_prompt_len(20)
    n_blocks = [-(-n // BS) for n in seq_lens]              # [3, 3, 3]
    row_ks = [layout.ks_of_tokens(s, np.arange(n_blocks[s]) * BS)
                 for s in range(T)]
    npb = layout.mem_blocks_needed                           # 1 + 3·3
    kc = torch.randn(npb, BS, H, D, dtype=torch.float16, device=dev)
    vc = torch.randn(npb, BS, H, D, dtype=torch.float16, device=dev)
    q = torch.randn(T, H, D, dtype=torch.float16, device=dev)
    slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], dtype=torch.float32,
                          device=dev)
    return SimpleNamespace(layout=layout, n_blocks=n_blocks,
                           row_ks=row_ks, kc=kc, vc=vc, q=q,
                           slopes=slopes)


def _run_arrival(c, alibi):
    arr = LayerArrival(num_layers=1)
    arr.begin(0)
    ev = EventRing(4).record(torch.cuda.current_stream())
    # The plan: the first c = 3 blocks of every sequence, K = 9 global
    # numbers — ACT ranks 0..4 (k = 0,2,4,6,8) and KV ranks 0..3 (k = 1,3,5,7).
    # Pushed as the loader would: ACT ranks 0..1 (k=0,2: block 0 of seqs 0
    # and 2) · KV ranks 0..1 (k=1,3: seq 1 block 0 and seq 0 block 1 — one
    # run spanning two rows) · ACT ranks 2..4 (k=4,6,8: seq 1 block 1 and the
    # current blocks of seqs 0 and 2) · KV ranks 2..3 (k=5,7: seq 2 block 1
    # and seq 1's current block).
    arr.push(0, ArrivalItem(kind=0, rank0=0, n=2, event=ev))
    arr.push(0, ArrivalItem(kind=1, rank0=0, n=2))
    arr.push(0, ArrivalItem(kind=0, rank0=2, n=3, event=ev))
    arr.push(0, ArrivalItem(kind=1, rank0=2, n=2))
    arr.close(0)
    act_pool = ActBufferPool(
        n_slots=1, act_buffer_bytes=4096, alloc_aligned=alloc_aligned,
        slot_of=lambda layer_idx: 0,
        index_bytes=arrival_index_bytes(T, 3, BS, alibi), launch_blocks=1)
    _ACT_POOL_KEEP.append(act_pool)
    ctx = ArrivalCtx(arrival=arr, layer_idx=0, row_sids=[0, 1, 2],
                     layout=c.layout, seq_lens=seq_lens,
                     seq_to_row={0: 0, 1: 1, 2: 2}, block_size=BS,
                     act_pool=act_pool)
    out = torch.empty(T, H, D, dtype=torch.float16, device=dev)
    # The caches already hold the new token: the append has nothing to
    # put in.
    arrival_decode(c.q, out, c.kc, c.vc, scale=scale, ctx=ctx,
                   alibi_slopes=(c.slopes if alibi else None),
                   append_cb=lambda rows: None)
    torch.cuda.synchronize()
    return out


def _run_mono(c, alibi):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    maxb = max(c.n_blocks)
    bt = torch.zeros(T, maxb, dtype=torch.int32, device=dev)
    for s in range(T):
        mems = c.layout.mem_blocks_np(0, c.row_ks[s])
        bt[s, :len(mems)] = torch.tensor(mems, dtype=torch.int32, device=dev)
    out = torch.empty(T, H, D, dtype=torch.float16, device=dev)
    flash_attn_varlen_func(
        q=c.q, k=c.kc, v=c.vc, out=out,
        cu_seqlens_q=torch.arange(T + 1, dtype=torch.int32, device=dev),
        max_seqlen_q=1,
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        max_seqlen_k=max(seq_lens), softmax_scale=scale, causal=False,
        window_size=(-1, -1), block_table=bt,
        alibi_slopes=(c.slopes if alibi else None), fa_version=2)
    return out


@pytest.mark.parametrize("drain", [32768, BS])
@pytest.mark.parametrize("alibi", [False, True])
def test_arrival_decode_matches_the_monolithic_reference(case, alibi, drain,
                                                        monkeypatch):
    monkeypatch.setattr(arrival_attention, "_PARTIAL_ATTN_TOKENS", drain)
    got = _run_arrival(case, alibi).float()
    want = _run_mono(case, alibi).float()
    err = (got - want).abs().max().item()
    assert err < 2e-3, (f"alibi={alibi} drain={drain}: arrival_decode vs "
                        f"mono max_err={err}")
