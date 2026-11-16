"""ACT reload support — owns the kv_gen stream for the ACT→K/V path.

An activation block is persisted to NVMe as raw fp16 hidden state, which
is about half the bytes of K + V. The reload itself lives in the loader
(`loader_thread._pipeline_layer_reads`): the activations loaded into the
activation buffer are fed, one kv_gen launch at a time, to the fused
Triton `kv_gen` kernel — called once for K and once for V — which reads
the layer's k/v projection weights straight from the weight buffer and
stores K, V into one contiguous run of `runner.kv_caches[layer]`
blocks. The recomputation is lossless, because K = hidden @ W_k + b_k is
exactly the projection the forward pass ran.

This class holds what that path needs per process: the dedicated kv_gen
CUDA stream and the cache dimensions.
"""

from __future__ import annotations

import torch


class ACTLoader:
    """Per-process helper, set up from `_on_kv_cache_ready` once vLLM has
    allocated the kv_caches."""

    def __init__(self, kv_caches: list[torch.Tensor]):
        # Dedicated stream for the kv_gen kernels; the forward pass runs on
        # a higher-priority stream
        # (streaming_loader._use_worker_priority_stream). Created lazily on
        # first access since stream construction needs an initialised cuda
        # context.
        self._kv_gen_stream = None
        ref = kv_caches[0]
        self.num_kv_heads = int(ref.shape[3])
        self.head_dim = int(ref.shape[4])

    def _stream(self) -> torch.cuda.Stream:
        if self._kv_gen_stream is None:
            # Priority 0, the lowest CUDA offers; the forward pass's stream
            # is raised above it, so kv_gen never preempts the forward pass.
            # forward(l) is the critical path: it is gated on kv_gen(l), and
            # the loader can only reload into the slot the forward pass
            # vacates, so starving the forward pass stalls the whole
            # pipeline.
            # Priority affects scheduling only; the result is bit-exact
            # either way.
            self._kv_gen_stream = torch.cuda.Stream(priority=0)
        return self._kv_gen_stream
