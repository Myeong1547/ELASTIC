"""kv_gen kernels — the GPU half of the KV/activation trade.

An activation block stores raw post-LN hidden on NVMe; the reload path
projects it back to K/V with the layer's own K/V projection weights. The
projection is exact (no quantisation, no decompression stage), so the
gemm is the entire reload compute and the recomputed K/V is bit-for-bit
what the prefill would have written.

`kv_gen` is the fused gemm + store that lands the result directly in
the paged KV cache.

The rule that sets each block's form from α lives in `elastic.runtime.labels`.
"""
