# ELASTIC Implementation

ELASTIC runs batch LLM inference on a cache-coherent unified-memory system with NVMe offloading. Its shared execution structure, the **ELASTIC Engine**, connects a rotating set of layer slots, a storage layout that follows block production order, arrival-based attention, and writeback. KV/activation hybrid caching reduces the bytes carried through that structure; the measured policy chooses each layer's activation ratio, the share of its blocks stored as activation blocks.

This document follows the implementation's data flow. Installation and commands are in the [usage guide](USAGE.md).

[한국어](DESIGN_KO.md) · [README](../README.md) · [Usage](USAGE.md)

## 1. Model initialization and layer slots

[`StreamingModelLoader`](../elastic/vllm_runtime/streaming_loader.py) registers the vLLM load format `elastic_streaming`. It creates the model on `meta`, materializes embeddings, the final LayerNorm, and the output head, and leaves decoder-layer parameters without permanent weight storage. Before each layer runs, [`WeightTable.bind_layer`](../elastic/runtime/weight_table.py) binds those parameters to views of the loaded slot's weight buffer.

A slot is one logical unit with four buffers:

| Buffer | What it holds |
|---|---|
| Weight buffer (slot.weight) | The current layer's staged parameters |
| K buffer and V buffer (slot.k, slot.v) | That layer's loaded or kv_gen-recomputed K/V blocks |
| Activation buffer (slot.act) | Loaded activations, activations awaiting writeback, and prefill working storage |

The buffers use aligned host allocations registered with CUDA. The CPU I/O path and GPU tensor views access the same backing memory. This path assumes cache-coherent UMA; it does not implement a separate host-to-device copy stage.

`StreamingModelLoader._compute_n_slots` derives the slot count `n_slots` (N_slots in the paper, Eq. 9) from the supplied slot memory budget and the configured batch and maximum sequence length. Activation buffers reserve their full capacity independently of the selected activation ratio. Framework allocations, scratch, resident extras, and the OS also need memory outside that budget.

[`LayerSlotPool`](../elastic/vllm_runtime/layer_pool.py) rotates slots by a monotonically increasing **visit** number:

```text
slot = visit % n_slots
```

A visit is one execution of one layer. It keeps increasing across decode steps and batches. A fixed `layer % n_slots` mapping would reuse the wrong slot when the number of layers is not divisible by `n_slots`. The loader (`ReadRouter`) rebinds each layer's KV cache when it acquires the new visit's slot, before any reload is submitted.

## 2. Storage follows block production order

The staged weight blob and manifest are produced by [`stage_opt`](../elastic/cli/stage_opt.py). The runtime cache is a separate blob managed by [`BlockKVLayout` and `BlockKVStorage`](../elastic/runtime/block_kv_storage.py). Each layer's file space holds three block regions:

```text
[ ACT_i: activation blocks ][ K_i: K of KV blocks ][ V_i: V of KV blocks ]
```

The storage unit is a whole block (vLLM's PagedAttention block of `block_size` tokens). A sequence's last, incomplete block (a mutable block) keeps its reserved elements and is overwritten as it grows. KV blocks retain the token-major cache layout; activation blocks store hidden states after the pre-attention LayerNorm and before Q/K/V projection.

Within a layer, one global number `k` identifies a block across the batch. With `N_seq` sequences (`BlockKVLayout.n_seq`), prompt length `S`, and `n = ⌊S/block_size⌋` immutable blocks per sequence (`BlockKVLayout.n_immutable`), block `j` (the block index within the sequence, from 0) of sequence `seq` (0 ≤ `seq` < `N_seq`) takes the following number (Eqs. 10–11 in the paper):

| Block range | Global number `k` | Write behavior |
|---|---|---|
| Immutable blocks (`0 ≤ j < n`) | `k = seq · n + j` | Produced during prefill; not overwritten |
| Mutable blocks (`j ≥ n`: the mutable prompt block and decode blocks) | `k = N_seq · n + (j − n) · N_seq + seq` | One current block per sequence is overwritten during decode |

This places each decode step's current blocks next to one another in `k` order. The form rule maps them to a dense range in each block region (`ACT_i`, `K_i`, `V_i`). The loader can consequently reload block-region prefixes as long contiguous ranges instead of following per-sequence fragments.

[`labels.py`](../elastic/runtime/labels.py) assigns each block's form (activation block or KV block) and its rank, the block's dense position inside its block region. `BlockKVLayout` owns both the file offsets and the memory-block mapping. In the K and V buffers, block 0 is reserved as vLLM's null block; the activation blocks follow from block 1, where kv_gen writes their recomputed K/V, and the directly loaded KV blocks start at `mem_kv_base`, the fixed block after all of the layer's activation blocks for the run. Writers, readers, and attention metadata use this same mapping.

`BlockKVStorage.ensure_exists` reserves the cache file with `posix_fallocate` when creating or enlarging it. Logical block placement and file allocation are separate concerns; this method does not repair an already existing file's physical extent layout.

## 3. Reads publish usable data as it arrives

[`ReadRouter`](../elastic/vllm_runtime/loader_thread.py), the loader, acquires a released slot and builds one layer's read plan in this order:

1. Weight segments into the slot's weight buffer (slot.weight).
2. `ACT_i` segments into its activation buffer (slot.act); kv_gen launches over the loaded activations are cut separately from the segments, at `ELASTIC_LOAD_CHUNK_TOKENS` tokens.
3. `K_i` and `V_i` segments directly into its K and V buffers (slot.k, slot.v).

The loader publishes the slot when all of its weights have arrived. The forward pass can then start the layer without waiting for its entire context. Weight availability and context-block availability are separate signals.

A block arrives when it becomes available for attention. A KV block arrives when both its K and V have been loaded: a directly loaded range of KV blocks is published after both its K and V reads complete. An activation block arrives when its kv_gen has completed. kv_gen runs on a dedicated CUDA stream: after a launch's reads and the layer's weights are ready, [`kv_gen`](../elastic/kernels/kv_gen.py) applies the layer's K/V projections and stores the results directly into the destination blocks of the K and V buffers. [`ACTLoader`](../elastic/vllm_runtime/act_loader.py) owns this kv_gen stream.

[`LayerArrival`](../elastic/vllm_runtime/layer_arrival.py) carries the resulting readiness feed to the forward pass. Each item contains a block-region kind, a starting rank within that block region, a block count, and, for activation blocks, a CUDA event recorded after kv_gen. It carries no copied tensor data. The loader closes the feed after all reads have completed and all kv_gen launches have been issued; the forward pass still waits on the recorded events before using their results.

## 4. Attention consumes the ready ranges

[`arrival_decode`](../elastic/vllm_runtime/arrival_attention.py) consumes ready ranges in partial attention calls, each over `ELASTIC_ARRIVAL_DRAIN_TOKENS` tokens of arrived blocks. That volume is both a lower and an upper bound (a single arrived item larger than it goes out whole), and the remainder is flushed when the layer's feed closes. It converts block-region ranks back to sequence rows and positions using `BlockKVLayout`, waits on the call's kv_gen event on the forward pass's stream, and computes attention over those blocks. Partial outputs and log-sum-exp values are merged by LSE merging into the running output and LSE of each query row (`running_o`, `running_lse`, kept in FP32).

The current block, the mutable block that receives the new token, is reloaded too. The forward pass appends the new token's K and V only after that block arrives, then attends it. If the block is an activation block, it also appends the new token's hidden state to the activation buffer after the reload and kv_gen. This ordering prevents an incoming read from overwriting the new token.

Once every sequence's blocks have been consumed, the merged attention output continues through the output projection and FFN. There is no requirement to load the whole layer's context before beginning partial attention.

Prefill has no past context to reload. It uses vLLM's prefill attention and establishes the cache for later decode steps. Large prefills use a zig-zag prefill over GPU batches: a layer's weights are loaded once and shared by its GPU batches, and immutable blocks are saved as they are produced. The zig-zag prefill and capture callbacks live in [`streaming_loader.py`](../elastic/vllm_runtime/streaming_loader.py).

## 5. Writeback determines when a slot can be reused

[`WriteRouter`](../elastic/vllm_runtime/nvme_write_router.py), the writeback, receives file ranges and views of the live K, V, and activation buffers. It does not copy them into a second save buffer. The source slot must therefore remain valid until the writes complete.

The forward pass registers each write with an increasing ticket, then marks that ticket on its CUDA stream after the producing operations. [`StreamSignal`](../elastic/loader/nvme_loader.py) publishes the ticket's progress and wakes the writeback when the GPU reaches the mark. The writeback submits the newly eligible writes and reaps their completions.

A slot is reusable only when all three conditions hold:

- The forward pass has handed off this visit with its CPU-side forward-done marker (`forward_done_visit`).
- The GPU has passed the visit's layer-end ticket.
- Every write licensed up to that ticket has completed.

The writeback then releases the visit's slot (`release_tail`) and wakes the loader. A Python layer callback returning is not sufficient to release the underlying bytes. During decode, the writeback saves the current block of each sequence (the step's last N_seq blocks); during prefill, immutable blocks can leave earlier, and the mutable prompt blocks leave at layer end.

## 6. One NVMe engine owns device I/O

[`NvmeEngine`](../elastic/loader/nvme_loader.py) exposes separate read and write ports to Python. A single C thread in [`loader.c`](../native_loader/src/loader.c) owns their shared io_uring. It refills reads to a fixed queue depth and gives admitted write batches priority while writes remain active. Write composition and the fill bound on in-flight write bytes are handled there.

| Component | Owns |
|---|---|
| Loader (`ReadRouter`, thread `elastic-head`) | Read plans, reload completions, kv_gen launches, arrival publication |
| Forward pass (forward thread) | Model execution, partial attention, buffer updates, stream tickets |
| Writeback (`WriteRouter`, thread `elastic-tail`) | Eligible saves, write completions, slot release |
| NVMe engine (`nvme_engine`) | io_uring submission/refill, write composition, I/O completion delivery |

The files use `O_DIRECT`. The native enqueue path checks the actual direct-I/O alignment requirements. Short transfers are continued inside the NVMe engine; Python receives a completed item or an error and checks its result before advancing readiness or release state. See the [native loader documentation](../native_loader/README.md) for the C interface.

## 7. Hybrid caching and the measured policy

The execution structure above is shared by KV-only and hybrid runs. For the supported OPT multi-head layout, one activation (hidden vector) has half the bytes of a K/V pair. Activation block saves and reloads reduce storage traffic and add kv_gen work; the underlying representation uses the original hidden states rather than a lossy codec.

The activation ratio `alpha_curve` (α_i per layer) is fixed for the run and may be supplied as one scalar or a per-layer curve. It changes the deterministic block forms and block-region sizes, while slots, reads, arrival consumption, and writeback retain the same lifecycle.

The hybrid policy is a configuration stage for the ELASTIC Engine:

- [`elastic/cost`](../elastic/cost/) builds a cost record from measured read throughput and GPU work (kv_gen, attention, and dense-layer operations), together with a configured sustained write rate (`ELASTIC_UNITS_WRITE_GBPS`, default 1.82 GB/s).
- [`from_units.py`](../elastic/policy/from_units.py) checks measurement provenance and converts the record into costs for the chosen workload and slot memory budget.
- [`solver.py`](../elastic/policy/solver.py) selects a per-layer activation ratio using the predicted disk/compute balance (`T_NVMe^i` against `T_comp^i`) and the ahead time the finite slot memory allows (`C_mem`).

Changing the context, the kv_gen launch size, or the partial attention call volume requires matching measurements. The runtime loads the resulting curve; it does not adapt the ratios token by token. Commands and output files belong in the [usage guide](USAGE.md).

## Integration scope

The documented path uses pre-norm OPT, FP16, Linux io_uring, NVMe, cache-coherent CUDA UMA, and the repository's [vLLM patch](../patches/vllm-elastic.patch). The patch supplies layer callbacks, KV-cache allocation/binding, the arrival-based attention entry point, and prefill/runner integration. Eager execution is required because decoder parameters are rebound between visits. The current model loader (`StreamingModelLoader`) rejects CUDA-graph execution, quantized KV caches, and post-norm OPT.
