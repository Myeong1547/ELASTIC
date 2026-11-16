"""kv_gen — the fused KV recomputation + store kernel for the activation
reload path (one call recomputes the K, or the V, of a run of blocks).

A layer's activation blocks are stored in file-offset order, so the loader
reads the blocks of a kv_gen launch as one dense [rows, hidden] run. A launch
covers a run of consecutive activation-block ranks (rank_ACT), and the slot's
K and V buffers are filled in load (= file-offset) order, so the launch's
blocks are the consecutive buffer blocks `1 + rank`: the destination of row
rm is the closed form `PhysBase + rm//BLK`.

The kernel runs one big-M gemm with the bias fused over the launch's
contiguous hidden (group-m swizzle for L2 reuse) and stores each output row
through the cache's own (block, slot, head, dim) strides into that run of
buffer blocks, with one pinned tile config (`_configs`).
"""
from __future__ import annotations

import triton
import triton.language as tl



def _configs():
    # Determinism invariant: exactly one config, so the process never chooses
    # a tile shape at runtime. `triton.autotune` over several configs picks the
    # winner by timing them, and timing is noisy, so the winning shape can
    # differ between runs of the same binary on the same input. Different tiles
    # mean a different accumulation order in the reduction, which perturbs the
    # last bits of the fp16 output and flips greedy-decoded tokens, and the
    # output must match a non-offloaded reference run token for token.
    # Pinning also means one JIT compile instead of one per candidate, so the
    # first call is not dominated by compilation.
    #
    # GROUP_M sets how many M-tiles share one pass over W, hence how many
    # times W is re-fetched: traffic = |X| + (num_pid_m/GROUP_M)·|W|. W is the
    # dominant memory term because it is re-read many times and lives in
    # host-registered shared memory, which streams slower than device-local
    # memory. Raising GROUP_M cuts W's multiplier, but the X panel it holds
    # resident must still fit the GPU's L2 — past that point the panel is
    # evicted between tiles and both operands get re-fetched. The values below
    # are the tile shape this kernel is tuned for on this GPU.
    bm, bn, bk, gm, w, s = 128, 256, 64, 8, 8, 3
    return [triton.Config(
        {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_M': gm},
        num_warps=w, num_stages=s)]


@triton.autotune(configs=_configs(), key=['N', 'K'])
@triton.jit
def _kv_gen_kernel(
    X, W, Bias, Out, PhysBase,
    M, N, K, BLK: tl.constexpr, HD: tl.constexpr,
    sxm, sxk, swn, swk, sob, sos, soh, sod,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Out[phys, slot, head, dim] = (X @ W^T + Bias)[row, col].  X (M,K) is in
    # contiguous file-offset order, W (N,K) is row-major. Output column col=rn
    # maps to (head=rn//HD, dim=rn%HD) and row rm to (block=rm//BLK,
    # slot=rm%BLK), with phys=PhysBase+block — the launch's blocks are
    # consecutive buffer blocks.
    #
    # The store is expressed through the cache's own (block, slot, head, dim)
    # element strides (sob, sos, soh, sod) rather than an assumed layout; for
    # the token-major cache (soh=HD, sod=1, sob=BLK*N, sos=N) it collapses to
    # plain out_row*N+rn addressing.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_ptr = X + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptr = W + rn[None, :] * swn + rk[:, None] * swk      # w^T tile [k,n]=W[n,k]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K),
                    other=0.0)
        w = tl.load(w_ptr, mask=(rn[None, :] < N) & (rk[:, None] + k0 < K),
                    other=0.0)
        acc += tl.dot(x, w)
        x_ptr += BLOCK_K * sxk
        w_ptr += BLOCK_K * swk
    b = tl.load(Bias + rn, mask=rn < N, other=0.0)
    acc += b[None, :]
    acc = acc.to(tl.float16)
    blk = rm // BLK
    slot = rm % BLK
    # int64: `phys * sob` overflows int32 on a large cache. Rows past M
    # address a block past the launch and are dropped by the store mask below.
    phys = tl.cast(PhysBase, tl.int64) + blk.to(tl.int64)
    head = rn // HD
    dim = rn % HD
    o_ptr = (Out + phys[:, None] * sob + slot[:, None] * sos
                 + head[None, :] * soh + dim[None, :] * sod)
    tl.store(o_ptr, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def kv_gen(x, w, bias, out_cache, phys_base, block_size):
    """x:(M,K) activation hidden, contiguous, blocks in the order they were
    read. w:(N,K). bias:(N,). out_cache:(num_blocks, block_size,
    num_kv_heads, head_dim) — the K-half (or V-half) of the layer's paged
    cache, in shared host-visible memory.

    Block-granular store: `phys_base` is the buffer block of x's FIRST
    block; block j of x lands in buffer block `phys_base + j`. Row rm belongs
    to block rm//BLK at slot rm%BLK, which requires x's rows to be
    block-aligned — every block contributes exactly BLK rows, including
    blocks appended during decode. The caller must hand over one run of
    consecutive buffer blocks; that is what a kv_gen launch covers (see the
    module docstring).

    The store reads the cache's actual (block, slot, head, dim) strides
    instead of assuming a contiguous (block, slot, N) view."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K, (w.shape, K)
    assert out_cache.dim() == 4, (
        f"out_cache must be 4D (num_blocks, block_size, num_kv_heads, "
        f"head_dim), got {tuple(out_cache.shape)}")
    nkvh, hd = out_cache.shape[2], out_cache.shape[3]
    assert nkvh * hd == N, (nkvh, hd, N)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M'])
                         * triton.cdiv(N, meta['BLOCK_N']),)
    _kv_gen_kernel[grid](
        x, w, bias, out_cache, int(phys_base),
        M, N, K, block_size, hd,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        out_cache.stride(0), out_cache.stride(1),
        out_cache.stride(2), out_cache.stride(3),
    )
