"""elastic — NVMe-offloaded LLM batch inference, KV/activation hybrid caching.

A model too large for memory streams its weights and its past KV off NVMe
once per layer per decode step, so NVMe bandwidth — not compute — bounds
the step. ELASTIC stores a fraction α of each layer's blocks as raw
hidden state (activation blocks, "ACT") instead of K and V. An activation
block is half the bytes, so those blocks cost half the NVMe traffic; the GPU
pays for them by recomputing K/V from the hidden state (kv_gen) during the
load-bound idle it would have spent waiting. α is the knob that trades NVMe
bytes for GPU work, and a solver picks one α per layer from measured cost
primitives.

"ACT" here is the layer-normalised hidden state that the K and V
projections consume. Storing exactly that tensor is what makes the KV
recomputation lossless: it is literally the input the prefill projected.

Subpackages:
    elastic.vllm_runtime — the vLLM integration: the `elastic_streaming`
                           model loader, the loader (ReadRouter) and the
                           writeback (WriteRouter), the layer slot pools and
                           arrival-based decode attention
    elastic.loader       — ctypes binding to the native NVMe engine
                           (`NvmeEngine`: io_uring + O_DIRECT; C library in
                           native_loader/)
    elastic.cost         — unit-cost measurement (units_<model>_S<S>.json)
    elastic.policy       — the hybrid policy (Algorithm 3) that solves
                           the per-layer α curve
    elastic.kernels      — kv_gen, the ACT → K/V recomputation kernel
    elastic.runtime      — each block's form (activation or KV), the NVMe
                           KV/activation file layout and storage, the weight
                           table
    elastic.cli          — stage_opt and run_bench

CLI entry: `python -m elastic` lists the pipeline's entry points.
"""
__version__ = "0.1.0"
