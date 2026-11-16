"""StreamingModelLoader — vLLM ModelLoader that meta-initialises the
model, allocates a rotating pool of layer slots, and runs a background
loader thread that fills slots from a `stage_opt` blob
via io_uring.

Flow per `load_model` call:

  1. Meta-init the model (`with torch.device("meta")`). No allocations
     beyond Python wrappers — a several-hundred-GiB model never touches
     the device here.

  2. Replace every meta param and buffer with a zero-byte device
     placeholder, except the attention quant scales, which keep their shape
     and are set to 1.0. Layer params get their bytes from slot views at
     forward time (6), extras from the blob in (3).

  3. Materialise extras (embeddings / final LayerNorm / lm_head)
     directly from the blob. Small, always-on; not part of the slot
     rotation.

  4. Allocate `n_slots` layer slots, each sized for the largest decoder
     layer. Slot bytes are UMA-aligned via nvme_loader.alloc_aligned so
     io_uring O_DIRECT can write into them with zero memcpy.

  5. Once vLLM has allocated the KV caches (`_on_kv_cache_ready`), bring up
     the NVMe engine, the writeback (`WriteRouter`, `elastic-tail`) and the
     loader (`ReadRouter`, `elastic-head`). The loader starts loading layer
     0, then 1, ..., wrapping at L, for as long as the rotation gives it a
     slot.

  6. Attach pre-/capture-/post-forward callbacks on every decoder layer
     (getattr callbacks patched into vLLM's opt/bloom forward, not
     PyTorch hooks):
       pre  → await_ready blocks until a slot is published holding this
              layer's bytes, then binds the layer's params to that
              slot's CUDA view.
       post → hands the step's new blocks to the write path by form,
              stamps the slot with the layer end's ticket and marks it on
              the stream; the writeback hears it pass, sends the writes,
              and releases the slot once they have landed.

KV/ACT loading runs alongside the weight path: the loader reloads saved
KV and activation blocks (ReadRouter, through the NVMe engine's read port)
and the forward pass attends each layer's reloaded KV in arrival order
(arrival_attention), so no per-layer barrier is needed.

Configuration:

    LLM(
        model='facebook/opt-1.3b',
        load_format='elastic_streaming',
        model_loader_extra_config={
            'blob_dir': '/path/to/staged',      # required (`stage_opt` output)
            'slots_mem_bytes': 96 * 2**30,      # required — n_slots is derived from this
            'alpha_curve': 0.5,                 # required — solver curve or scalar
            'kv_save_blob': '/path/kv.blob',    # required
        },
        enforce_eager=True,                      # required (cudagraph + .data swap don't mix)
        ...
    )

Every batch must have exactly `max_num_seqs` prompts of one length, each
running to the same `max_tokens` (`ignore_eos=True`), and the vLLM engine must
admit the whole batch in one step (`enable_chunked_prefill=False`,
`max_num_batched_tokens = max_num_seqs · max_model_len`); the arrival gate
refuses a step whose row count differs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import (
    is_quantized_kv_cache,
    set_default_torch_dtype,
)

from elastic.loader.nvme_loader import NvmeEngine, alloc_aligned
from elastic.runtime.weight_table import WeightTable, _replace_param

from elastic.runtime.block_kv_storage import BlockKVLayout, BlockKVStorage
from elastic.runtime.labels import act_rank, block_is_act

from .act_loader import ACTLoader
from .nvme_write_router import WriteRouter
from .layer_pool import LayerSlotPool, ActBufferPool
from .loader_thread import ReadRouter



logger = init_logger(__name__)


def _bind_kv_buffers(runner, pool, n_slots: int) -> None:
    """Make the K and V buffers follow the slot rotation, like the other two
    members of the slot.

    The slot is one coupled unit — weight buffer, K and V buffers,
    activation buffer — and the rotation that carries it is
    `visit mod n_slots`, so a layer's K and V buffers move by
    `L mod n_slots` every step just as its weight buffer does. vLLM binds a
    KV tensor per layer once, at cache init, from `l mod n_slots`: right for
    the first step and wrong from the second on, and a layer attending the
    wrong buffer is silent garbage rather than a crash. So the loader
    re-points it whenever it acquires a slot. Two places hold the binding
    and both are read per call, never cached:
      `runner.kv_caches[l]`      — every ELASTIC reader (the reload's DMA
                                   destination, kv_gen's store target,
                                   the write router's save source)
      `attn.kv_cache`            — vLLM's own attention, read at forward time

    Re-pointing when the loader acquires the slot rather than at the forward
    pass's pre callback is what makes it correct: the loader DMAs this
    layer's KV in long before the forward pass gets there, so the binding
    must already be the new tenancy's when the first byte lands. It is also
    safe there — the loader cannot acquire layer l's next tenancy until the
    writeback has released the previous one, which requires the forward pass
    to be done with it and its saves to have completed, so nobody is still
    reading the old binding.
    """
    from vllm.model_executor.models.utils import extract_layer_index
    n_layers = len(runner.kv_caches)
    # The n_slots distinct buffers, in slot order: vLLM's alias puts layer j
    # in buffer `j mod n_slots`, so the first n_slots entries are the buffers
    # themselves.
    bufs = [runner.kv_caches[j] for j in range(n_slots)]
    # layer index -> vLLM Attention module, via the same forward context vLLM
    # binds through (`bind_kv_cache`), so the two views agree by construction.
    ctx = runner.vllm_config.compilation_config.static_forward_context
    attn: list = [None] * n_layers
    for name, mod in ctx.items():
        if hasattr(mod, "kv_cache"):
            attn[extract_layer_index(name)] = mod

    def _rebind(layer_idx: int, slot_idx: int) -> None:
        buf = bufs[slot_idx]
        runner.kv_caches[layer_idx] = buf
        attn[layer_idx].kv_cache = buf
        # `pool.n_moved` counts rebinds where the slot is not
        # `layer % n_slots`, i.e. exactly the steps in which the rotation
        # has shifted the layer off its first-step buffer. It stays 0 when
        # n_slots divides the layer count.
        if slot_idx != layer_idx % n_slots:
            pool.n_moved += 1

    pool.set_bind_hook(_rebind)


def _alloc_uma_buffer(nbytes: int, keepalive: list) -> torch.Tensor:
    """Allocate `nbytes` of 4 KiB-aligned host memory, cudaHostRegister it, and
    return a CUDA accelerator view (UVA) whose `data_ptr()` is BOTH host- and
    device-accessible — the allocation the patched vLLM
    `_allocate_kv_cache_tensors` makes for the kv_caches too. The forward
    kernel writes it via the device view; a save's DMA reads the same
    physical bytes via the host pointer (zero-copy). The backing numpy array
    + host tensor + raw address are appended to `keepalive` so the registered
    memory isn't GC'd (which would dangle the registration)."""
    import ctypes

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.posix_memalign.restype = ctypes.c_int
    libc.posix_memalign.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
    addr = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(addr), 4096, nbytes) != 0:
        raise OSError(f"posix_memalign({nbytes}) failed")
    arr_t = (ctypes.c_uint8 * nbytes).from_address(addr.value)
    arr = np.frombuffer(arr_t, dtype=np.uint8)
    host_t = torch.from_numpy(arr)
    err = torch.cuda.cudart().cudaHostRegister(
        host_t.data_ptr(), host_t.nbytes, 0)
    if getattr(err, "value", err) != 0:
        raise RuntimeError(f"cudaHostRegister({nbytes}) failed: {err}")
    cuda_t = get_accelerator_view_from_cpu_tensor(host_t).view(torch.int8)
    # Zero on the host, synchronously. posix_memalign hands back whatever the
    # allocator held, so the buffer must be cleared — but not by a device
    # memset: `cuda_t.zero_()` is an asynchronous kernel on the current
    # stream, and a caller that fills the buffer from the host right after
    # this returns (the per-layer slot mappings at a step's gate, the
    # prefill's block tables) races it — the memset lands later and wipes a
    # prefix of what the host wrote, and the kernels that read the buffer
    # then see zeros (slot 0 = the null block) for some layers. Host memory
    # is cleared by a host store; the launch that follows orders it.
    ctypes.memset(addr.value, 0, nbytes)
    keepalive.append((arr, host_t, addr))
    return cuda_t


# The α curve enters here once, from the solver (or as one scalar), and is
# fixed for the run. IEEE rounding can hand us α a few ULPs outside [0,1]
# (e.g. 1.0000000000000002 from the solver's inverse map); snap those to the
# boundary. A gross excursion (> tol) or a NaN (the solver's infeasible cell)
# is a bad curve and raises.
_ALPHA_TOL = 1e-9


def _clamp_alpha(alpha: float) -> float:
    if alpha != alpha:
        raise ValueError("alpha is NaN")
    if alpha < -_ALPHA_TOL or alpha > 1.0 + _ALPHA_TOL:
        raise ValueError(f"alpha out of [0,1]: {alpha}")
    return min(1.0, max(0.0, alpha))


@register_model_loader("elastic_streaming")
class StreamingModelLoader(BaseModelLoader):
    """Streaming model loader. See module docstring."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = dict(load_config.model_loader_extra_config or {})
        blob_dir = extra.get("blob_dir")
        if blob_dir is None:
            raise ValueError(
                "load_format='elastic_streaming' requires "
                "model_loader_extra_config={'blob_dir': '<staged dir>'}")
        self.blob_dir = Path(blob_dir)
        # n_slots (the number of persistent layer slots) is derived from the
        # caller-supplied slot memory budget — there is no default. On
        # coherent unified-memory hardware a device allocation is system RAM,
        # so the figure is a policy choice, deliberately smaller than
        # capacity.
        if "slots_mem_bytes" not in extra:
            raise ValueError(
                "slots_mem_bytes is required in model_loader_extra_config. "
                "n_slots is derived from it: per-slot M_slot = weight buffer + "
                "K and V buffers + activation buffer, and n_slots is the "
                "largest count with n_slots·M_slot ≤ slots_mem_bytes "
                "(_compute_n_slots/_act_buffers_bytes). The caller decides "
                "the figure; on unified-memory hardware it is bounded by "
                "free system RAM, so leave a margin below that.")
        self.slots_mem_bytes = int(extra["slots_mem_bytes"])
        if self.slots_mem_bytes <= 0:
            raise ValueError(
                f"slots_mem_bytes must be positive, got "
                f"{self.slots_mem_bytes}")
        # The save blob (KV/ACT persistence): the WriteRouter writes new K/V
        # and ACT bytes here and the loader reloads them. Required: the
        # rotation's writeback — what releases a slot once its layer is done
        # and its writes have landed — is the write path's thread, and
        # without one no slot would ever be released.
        kv_save_blob = extra.get("kv_save_blob")
        if not kv_save_blob:
            raise ValueError(
                "elastic_streaming needs a save blob: pass kv_save_blob in "
                "model_loader_extra_config (--kv-save-blob).")
        self.kv_save_blob = Path(kv_save_blob)
        # Alpha curve — per-layer activation ratio α_i; required. Accepts a
        # single float (broadcast to every layer) or a per-layer list.
        # There is no default: a missing curve means the policy has not
        # been decided, and guessing one silently changes the layout.
        if "alpha_curve" not in extra:
            raise ValueError(
                "alpha_curve is required — pass it in "
                "model_loader_extra_config. "
                "The solver in elastic.policy.solver produces this; "
                "feed its output here verbatim. No default.")
        self.alpha_curve_raw = extra["alpha_curve"]   # per-layer list later

        manifest_path = self.blob_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.blob_dir}; "
                "stage with `python -m elastic.cli.stage_opt` first")
        self.blob_path = self.blob_dir / "model.bin"
        if not self.blob_path.exists():
            raise FileNotFoundError(
                f"model.bin not found in {self.blob_dir}")
        self.manifest = json.loads(manifest_path.read_text())
        # stage_opt writes only float16, and the weight table and the extras
        # view the blob's bytes as float16.
        if self.manifest["dtype"] != "float16":
            raise ValueError(
                f"{manifest_path}: dtype {self.manifest['dtype']!r}; the "
                "loader reads only float16 blobs")
        self.weight_table = WeightTable.from_manifest(self.manifest)

    # ------- BaseModelLoader API ---------------------------------------

    def download_model(self, model_config: ModelConfig) -> None:
        # `stage_opt` already wrote the blob; nothing to download here.
        return None

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # Real per-layer streaming happens via pre-forward callbacks set up
        # inside `load_model`. This entry point is only used by vLLM's
        # standalone reload paths, which our streaming model doesn't
        # support.
        raise NotImplementedError(
            "StreamingModelLoader streams per-layer at forward time; "
            "use load_model.")

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        # Configurations this runtime does not serve, refused once here:
        #   - CUDA graphs: capture's dummy forwards would run every layer's
        #     callbacks and take the rotation's visits (the model also swaps
        #     `.data` per layer, which a captured graph cannot follow);
        #   - a quantized KV cache: the arrival path's partial attention
        #     reads the cache pages as the model dtype;
        #   - a post-norm OPT (350m): the ACT capture owns the pre-attention
        #     LayerNorm, which that model does not have.
        if not model_config.enforce_eager:
            raise ValueError(
                "load_format='elastic_streaming' requires enforce_eager=True")
        if is_quantized_kv_cache(vllm_config.cache_config.cache_dtype):
            raise ValueError(
                f"load_format='elastic_streaming' does not serve a quantized "
                f"KV cache (kv_cache_dtype="
                f"{vllm_config.cache_config.cache_dtype!r})")
        _hf = model_config.hf_config
        if _hf.model_type == "opt" and not _hf.do_layer_norm_before:
            raise ValueError(
                "load_format='elastic_streaming' does not serve a post-norm "
                "OPT (do_layer_norm_before=False): the ACT capture is the "
                "pre-attention LayerNorm")
        # The zig-zag prefill's GPU batch size (tokens).
        prefill_gpu_batch_tokens = int(os.environ["ELASTIC_PREFILL_MB_TOKENS"])
        if prefill_gpu_batch_tokens <= 0:
            raise ValueError(
                f"ELASTIC_PREFILL_MB_TOKENS must be > 0, got "
                f"{prefill_gpu_batch_tokens}")

        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)

        # 1) Build the model on meta — no device allocations beyond
        #    Python wrappers.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            _log_meta_state(model)

            # 2) Replace every meta tensor (params + buffers) with a
            #    zero-byte device placeholder. `model.to_empty(device)`
            #    would allocate every param's full storage in one burst,
            #    and on unified memory a device allocation is system RAM,
            #    so a large model exhausts RAM before any streaming
            #    starts. Real storage comes later:
            #      • extras  → `_materialize_extras_into_model` (`_replace_param`)
            #      • layers  → pre-forward callback swaps `.data` to a slot view
            # Attention quant-scale buffers (`_q_scale` / `_k_scale` /
            # `_v_scale` / `_prob_scale`) are validated by vLLM's
            # attention path to be shape [1] or [num_heads]. A zero-byte
            # placeholder fails that check, so we keep their original
            # meta shape and initialise to 1.0 (the non-quant identity).
            # These buffers are tiny — ~hundreds of bytes per layer.
            _SCALE_SUFFIX = ("_q_scale", "_k_scale", "_v_scale",
                              "_prob_scale")
            def _is_scale(n: str) -> bool:
                return any(n.endswith(s) for s in _SCALE_SUFFIX)

            for name, p in list(model.named_parameters(recurse=True)):
                if not p.is_meta:
                    continue
                # `p.data = empty(0)` is rejected by Parameter's
                # set_data type check when the source is on `meta`.
                # Replace the Parameter itself instead.
                if _is_scale(name):
                    real = torch.ones(p.shape, dtype=p.dtype,
                                        device=target_device)
                else:
                    real = torch.empty(0, dtype=p.dtype,
                                         device=target_device)
                _replace_param(model, name, real)
            for name, b in list(model.named_buffers(recurse=True)):
                if not b.is_meta:
                    continue
                parent_path, _, attr = name.rpartition(".")
                parent = (model.get_submodule(parent_path)
                            if parent_path else model)
                if _is_scale(name):
                    real = torch.ones(b.shape, dtype=b.dtype,
                                        device=target_device)
                else:
                    real = torch.empty(0, dtype=b.dtype,
                                         device=target_device)
                setattr(parent, attr, real)

            with torch.device(target_device):
                # 3) Extras (small, always-on).
                self._materialize_extras_into_model(model, target_device)

                # vLLM's post-load processing (quant repacking, etc.).
                # The decoder layers hold zero-byte placeholder params, but
                # those modules have no quant methods on the non-quant path,
                # so it is a no-op for them.
                process_weights_after_loading(model, model_config,
                                                target_device)


        # 4) Persistent layer-slot pool sized from the slot memory budget.
        #    The resolved slot count is published back into the env by
        #    _compute_n_slots, so the UMA KV allocator (vLLM side), the weight
        #    pool and the loader all agree: weight and KV rotate together in
        #    one set of n_slots slots.
        num_layers_total = len(self.weight_table.layers)
        M = self._compute_n_slots(
            slots_mem_bytes=self.slots_mem_bytes,
            model_config=model_config,
            vllm_config=vllm_config,
            num_layers=num_layers_total,
            max_layer_bytes=self.weight_table.max_layer_bytes,
        )
        pool = LayerSlotPool(
            n_slots=M,
            weight_buffer_bytes=self.weight_table.max_layer_bytes,
            alloc_aligned=alloc_aligned,
            # Length of the layer→visit binding table. Not part of the
            # rotation — the slot is `visit mod n_slots` and nothing else.
            n_layers=num_layers_total,
        )

        # 5) The loader and the writeback need the NVMe engine, which needs
        #    the KV caches, so they come up in `_on_kv_cache_ready` below.
        #    Nothing runs a layer before that: vLLM runs neither its profile
        #    forward nor its warm-up forward for this model.
        decoder_layers = self._decoder_layers(model)
        block_load_state: dict[str, Any] = {}
        # Arrival-based decode attention: the forward pass consumes each
        # layer's reloaded KV in the loader's arrival order (partial attention
        # per arrival, merged by LSE). One shared readiness feed, loader
        # (producer) ↔ forward pass (consumer).
        from .layer_arrival import LayerArrival
        _arrival = LayerArrival(num_layers=len(decoder_layers))
        block_load_state["arrival"] = _arrival

        # 6) Pre-/post-forward callbacks on each decoder layer.
        # block_size / alpha_curve are set at kv-ready (`_on_kv_cache_ready`);
        # `in_batch_prefill` and `prefill_mb_groups` (the zig-zag prefill's
        # GPU batches, None when the batch is one GPU batch) are the gate's,
        # set per step (`_make_arrival_gate`), which groups by
        # `prefill_gpu_batch_tokens`.
        gate_state: dict[str, Any] = {
            "block_size": None,
            "alpha_curve": None,
            "in_batch_prefill": False,
            "prefill_gpu_batch_tokens": prefill_gpu_batch_tokens,
            "prefill_mb_groups": None,
        }

        # Save routing context — the write router + gate_state.
        # The layer seams read gate_state's step rows + alpha_curve to
        # dispatch saves.
        save_ctx: dict[str, Any] = {
            "write_router": None,
            "gate_state": gate_state,
        }

        for i, layer_module in enumerate(decoder_layers):
            # The decoder layer (OPTDecoderLayer / BloomBlock) is patched in
            # `vllm/model_executor/models/{opt,bloom}.py` to call these
            # `_elastic_*` attributes at the right points of its forward;
            # the callables are attached directly, not through PyTorch's
            # register_forward_*_hook.
            layer_module._elastic_pre = _make_pre_cb(
                i, pool, self.weight_table, gate_state)
            layer_module._elastic_capture_ln = _make_capture_ln(
                i, gate_state, save_ctx)
            layer_module._elastic_post = _make_post_cb(
                i, pool, save_ctx)
            layer_module._elastic_attn_done = _make_attn_done_cb(
                i, gate_state, save_ctx)
        # x-wheel: the zig-zag prefill's residual stream lives in the
        # activation buffers, pre-LN (prep_span/save_span drive it through
        # this ctx). OPT only — BLOOM's prefill is the plain whole-batch loop.
        _dec = getattr(getattr(model, "model", None), "decoder", None)
        if _dec is not None:
            _dec._elastic_xw = _XWheel(gate_state, save_ctx, decoder_layers,
                                       pool)

        # Stash the components on the model so the shutdown below can find
        # them. vLLM never inspects extra attrs.
        model._elastic_loader = None       # the loader, once it is up

        def _elastic_shutdown() -> None:
            """The runtime's end, called by vLLM's `GPUModelRunner.shutdown`
            before it frees the KV caches. In order, and with no timeout:

              1. the loader is told to stop; it has nothing left to load;
              2. the writeback gets a stop ticket behind the last layer end,
                 lands every write still owed and exits — and its exit wakes
                 the loader, which then exits too;
              3. the NVMe engine closes: what is on the device lands first
                 (the STOP line reports anything that never reached it)."""
            head = getattr(model, "_elastic_loader", None)
            _rt = getattr(model, "_elastic_write_router", None)
            _eng = getattr(model, "_elastic_nvme_engine", None)
            if head is not None:
                head.request_stop()
            if _rt is not None:
                _rt.stop()
            if head is not None:
                head.join()
            if _eng is not None:
                _eng.close()

        model._elastic_shutdown = _elastic_shutdown

        def _on_kv_cache_ready(runner) -> None:
            ref = runner.kv_caches[0]
            # Shape: (2, num_blocks, block_size, num_kv_heads, head_dim)
            gate_state["block_size"] = int(ref.shape[2])
            # Resolve alpha_curve_raw → per-layer list.
            n_layers = len(runner.kv_caches)
            raw = self.alpha_curve_raw
            if isinstance(raw, (int, float)):
                gate_state["alpha_curve"] = [float(raw)] * n_layers
            else:
                gate_state["alpha_curve"] = [float(p) for p in raw]
                if len(gate_state["alpha_curve"]) != n_layers:
                    raise ValueError(
                        f"alpha_curve has {len(gate_state['alpha_curve'])} "
                        f"entries but model has {n_layers} layers")
            gate_state["alpha_curve"] = [
                _clamp_alpha(a) for a in gate_state["alpha_curve"]]
            # Seq ids are assigned per batch by the gate at the batch's
            # prefill step (`SeqIdAllocator.assign`): ids 0..N_seq-1 in row
            # order. Storage is sized for max_num_seqs (the BlockKVLayout
            # below).
            from .seq_id_allocator import SeqIdAllocator
            max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
            gate_state["seq_id_alloc"] = SeqIdAllocator()
            gate_state["runner"] = runner
            # The attention layers by name, with their layer index: the
            # per-layer dicts the gate hands the runner are keyed by name.
            from vllm.model_executor.models.utils import extract_layer_index
            gate_state["layer_names"] = [
                (_n, extract_layer_index(_n)) for _n in
                runner.kv_cache_config.kv_cache_groups[0].layer_names]
            # The loader's view of the KV caches.
            block_load_state["kv_caches"] = runner.kv_caches
            gate_state["block_load_state"] = block_load_state

            # ACTLoader owns the kv_gen stream the loader launches kv_gen
            # on, straight into the just-allocated kv_caches.
            block_load_state["act_loader"] = ACTLoader(runner.kv_caches)

            # NVMe write path (WriteRouter) — NVMe-engine-backed async
            # K/V+ACT writes to the blob.
            # max_blocks_per_seq: how many blocks one sequence can grow to,
            # ⌈max_model_len / block_size⌉. Bound by max_model_len.
            max_blocks_per_seq = (
                (model_config.max_model_len
                 + int(ref.shape[2]) - 1)
                // int(ref.shape[2]))
            storage_layout = BlockKVLayout(
                n_seq=max_seqs,
                num_layers=n_layers,
                max_blocks_per_seq=max_blocks_per_seq,
                block_size=int(ref.shape[2]),
                num_heads=int(ref.shape[3]),
                head_dim=int(ref.shape[4]),
                dtype=ref.dtype,
                # Per-layer α sets how a layer's blocks split between
                # its ACT block region and its K/V block regions. It must
                # match the form routing's alpha_curve exactly, or saves
                # and reloads address different bytes. Decode blocks need
                # no extra configuration: they are the later offsets
                # of the same block regions.
                alpha=list(gate_state["alpha_curve"]),
            )
            # The slot's K and V buffers are filled in load (= file-offset)
            # order (`mem_block`): null block, then the activation blocks,
            # then the KV blocks, each placed by its rank. vLLM's pool
            # (num_gpu_blocks_override = N_seq·max_blocks_per_seq + 1) is
            # exactly that many blocks; anything smaller cannot hold the
            # layout.
            if storage_layout.mem_blocks_needed > int(ref.shape[1]):
                raise RuntimeError(
                    f"KV buffer has {int(ref.shape[1])} blocks but the "
                    f"layout addresses {storage_layout.mem_blocks_needed} "
                    f"(1 + {max_seqs} seqs × {max_blocks_per_seq} "
                    f"blocks) — size num_gpu_blocks_override for the "
                    f"full batch")
            # The prefill's block tables. A block's buffer position is the
            # layout's closed form per layer (`mem_block`), so every layer's
            # table is its own; vLLM's block ids are not positions here. One
            # registered UMA buffer for the run, the layout's whole shape
            # (L, max_num_seqs, max_blocks_per_seq) int32: a layer's pre
            # callback fills its rows from the host at the batch's prefill
            # step and the device reads them in place — layer l's table is
            # its first N_seq rows at full width, like vLLM's own block table.
            _bt_shape = (n_layers, max_seqs, max_blocks_per_seq)
            _bt_keep: list = []
            _bt_dev = _alloc_uma_buffer(
                4 * n_layers * max_seqs * max_blocks_per_seq, _bt_keep)
            gate_state["block_tables"] = {
                "dev": _bt_dev.view(torch.int32).view(*_bt_shape),
                "host": np.frombuffer(_bt_keep[-1][0],
                                      dtype=np.int32).reshape(_bt_shape),
                "keep": _bt_keep}
            # Save and reload are both O_DIRECT: the block layout makes
            # every save 4 KiB-aligned and whole-block, so saves DMA
            # straight to NVMe with no page-cache writeback (which on
            # unified memory would compete with compute for DRAM
            # during large prefills), and an O_DIRECT reload is
            # coherent with an O_DIRECT save.
            storage = BlockKVStorage(self.kv_save_blob, storage_layout)
            storage.ensure_exists()
            # The NVMe engine: one C pthread, one io_uring ring,
            # two FIFOs (read / write) popped with write priority — the
            # device is used serially, as the solver prices it. The two
            # routers below are its only clients and never touch the
            # device themselves.
            nvme_engine = NvmeEngine(
                str(self.kv_save_blob), read_ring=256, write_ring=256)
            model._elastic_nvme_engine = nvme_engine
            # Every NVMe save (prefill and decode, KV and ACT) goes
            # through the writeback: producers register requests, the
            # writeback hands the licensed ones to the NVMe engine's write
            # FIFO and releases a slot once its writes have landed. Write
            # completions come off the NVMe engine's write port.
            write_router = WriteRouter(
                storage_layout, runner.kv_caches,
                engine=nvme_engine, pool=pool)
            model._elastic_write_router = write_router
            # KV and activation block reads go through the NVMe engine's
            # read port: the ReadRouter enqueues segments whose
            # destinations are cudaHostRegister'd UMA memory (KV: straight
            # into kv_caches; ACT: into the activation buffers for kv_gen),
            # and the NVMe engine DMAs into them with no CPU→GPU copy.
            kv_nvme = nvme_engine.read_port
            block_load_state["storage_layout"] = storage_layout
            # Activation buffers, registered once for the whole run:
            # registering and unregistering per block would device-sync
            # and thereby serialize every block read. `_k_bytes` below is
            # one block's K plane, which is also one activation block's
            # hidden, so activation buffers are sized in K planes.
            _elem = ref.element_size()
            _k_bytes = (int(ref.shape[2]) * int(ref.shape[3])
                         * int(ref.shape[4]) * _elem)
            # A run of consecutive ranks in a block region is one file
            # range landing as one run of buffer blocks. Reload reads it in
            # ELASTIC_READ_SPAN_MB segments (the segment size of the weight
            # and K/V reads): a single block is far too small an operation
            # to reach the bulk bandwidth, a multi-MiB segment is not.
            # kv_gen is cut apart from the reads, into launches of
            # ELASTIC_LOAD_CHUNK_TOKENS tokens; a kv_gen launch goes once
            # every segment it overlaps has landed. A kv_gen launch is a
            # launch grouping only — it is a contiguous slice of the
            # layer's activation buffer, which kv_gen (gemm + store fused,
            # called once for K and once for V) reads (UVA) directly,
            # storing K and V each as one contiguous run of buffer blocks:
            # no gather copy, no repacking. Grouping also keeps the op count
            # off the GPU command queue, which one small CUDA op per block
            # would flood.
            _bs_tok = int(ref.shape[2])
            _launch_tokens = int(os.environ["ELASTIC_LOAD_CHUNK_TOKENS"])
            if _launch_tokens <= 0:
                raise ValueError(
                    f"ELASTIC_LOAD_CHUNK_TOKENS must be > 0, got "
                    f"{_launch_tokens}")
            _launch_blocks = max(1, _launch_tokens // _bs_tok)
            # One activation buffer per slot, sized by
            # _act_buffers_bytes — the same function _compute_n_slots
            # charges against the slot memory budget, so accounting ==
            # allocation. Host UMA (cudaHostRegister). A layer's loaded
            # activations live as long as its slot: until the writeback
            # releases that layer's visit.
            _act_buffers_total = self._act_buffers_bytes(
                pool.n_slots, max_blocks_per_seq * max_seqs, _k_bytes)
            # total == n_slots units: the prefill residual stream
            # lives in the activation buffers themselves (x-wheel).
            _act_buffer_bytes = _act_buffers_total // pool.n_slots
            # ACT reload reads only the hidden (K-sized) plane of each
            # block, so an activation buffer has the K and V buffers'
            # shape with one unit per block instead of two:
            # act_buffer_bytes == n_blocks_total · k_bytes
            #              == T_max · hidden · dtype
            # — the `0.5·kv` unit (M_A) the slot budget reserves n_slots
            # of.
            # The decode attention's index scratch, per slot: the
            # closed-form bound on one visit's calls. ALiBi splits a
            # row into position-contiguous runs, so it has its own.
            from .arrival_attention import arrival_index_bytes
            _alibi = any(
                getattr(getattr(_m, "impl", None), "alibi_slopes",
                        None) is not None
                for _m in runner.vllm_config.compilation_config
                .static_forward_context.values())
            act_pool = ActBufferPool(
                n_slots=pool.n_slots,
                act_buffer_bytes=_act_buffer_bytes,
                alloc_aligned=alloc_aligned,
                # One rotation, so one resolver: the activation buffers
                # never recompute `l mod n_slots`. The rotation is over
                # visits, and the loader publishes where each layer
                # landed.
                slot_of=pool.slot_of_layer,
                index_bytes=arrival_index_bytes(
                    max_seqs, max_blocks_per_seq, _bs_tok, _alibi),
                launch_blocks=_launch_blocks,
            )
            _bind_kv_buffers(runner, pool, pool.n_slots)
            block_load_state["act_pool"] = act_pool
            # ACT rows have no buffer besides the activation buffers: a
            # dedicated one would duplicate a budget line and the blob's
            # bytes. Every ACT row is written in the activation buffers the
            # reload lands in — a prefill row by its LN (the capture cb, or
            # `prep_span` in the zig-zag prefill), a decode row by
            # `arrival_decode` once its block's ACT load is in — and saved
            # asynchronously by the router. The pool is sized for the α=1
            # worst case over n_slots layers.
            # Wire the router into the post-callback routing ctx.
            save_ctx["write_router"] = write_router
            block_load_state["write_router"] = write_router
            # The loader: everything it reads through and into exists
            # from here.
            head = ReadRouter(
                pool=pool,
                kv_nvme=kv_nvme,
                weight_path=self.blob_path,
                weight_table=self.weight_table,
                block_load_state=block_load_state,
            )
            model._elastic_loader = head
            head.start()

            runner._kv_arrival_gate = _make_arrival_gate(gate_state)
            shape = tuple(ref.shape)
            logger.info(
                "bound %d kv_cache tensors (layer 0 shape=%s); "
                "block_size=%s", n_layers, shape, gate_state["block_size"])

        model._elastic_on_kv_cache_ready = _on_kv_cache_ready

        logger.info(
            "StreamingModelLoader: n_slots=%d, max_layer_bytes=%.2f MiB",
            pool.n_slots, self.weight_table.max_layer_bytes / (1 << 20),
        )

        return model.eval()

    # ------- helpers ----------------------------------------------------

    def _materialize_extras_into_model(self, model: nn.Module,
                                         target_device: torch.device) -> None:
        """Read the manifest's extras section from the staged blob and
        replace the matching `nn.Parameter` in `model` with a device tensor
        holding those bytes (until here it is a zero-byte placeholder)."""
        extras = self.manifest["extras"]
        n_materialized = 0
        with self.blob_path.open("rb") as fh:
            for em in extras:
                name = str(em["name"])
                offset = int(em["offset"])
                nbytes = int(em["nbytes"])
                shape = tuple(em["shape"])
                param = model.get_parameter(name)

                buf = bytearray(nbytes)
                fh.seek(offset)
                n_read = fh.readinto(buf)
                if n_read != nbytes:
                    raise IOError(
                        f"short read for extra {name!r}: got {n_read} "
                        f"bytes, expected {nbytes}")
                arr = np.frombuffer(buf, dtype=np.uint8)
                cpu_t = torch.from_numpy(arr).view(torch.float16).view(shape)
                dev_t = cpu_t.to(target_device, non_blocking=False)
                # Wholesale Parameter swap. vLLM pads some tensors
                # (`VocabParallelEmbedding` rounds vocab up to a
                # tensor-parallel-friendly multiple, e.g. 50272 → 50304)
                # so the param's declared shape may differ from the HF
                # source we just read. _replace_param sets a
                # new Parameter sized for the HF source — vLLM's
                # embedding lookup is content-only and treats the new
                # smaller (or differently-shaped) tensor as authoritative.
                # Tied references (lm_head ↔ embed_tokens) are at the
                # module level, so they pick up the new Parameter through
                # the module's __getattr__ — no re-tying needed.
                _replace_param(model, name, dev_t)
                n_materialized += 1
                del buf, arr, cpu_t, dev_t, param

        logger.info(
            "StreamingModelLoader: materialised %d extras from %s",
            n_materialized, self.blob_path)

    @staticmethod
    def _decoder_layers(model: nn.Module) -> nn.ModuleList:
        # Per-architecture location of the decoder ModuleList.
        #   OPT:   OPTForCausalLM.model.decoder.layers
        #   BLOOM: BloomForCausalLM.transformer.h
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        return model.model.decoder.layers

    @staticmethod
    def _act_buffers_bytes(n_slots: int, n_blocks_total: int,
                             k_bytes: int) -> int:
        """Size the activation buffers → bytes.

        Single source of truth for the pool: called by _compute_n_slots
        (accounting — the pool lives inside the slot memory budget) and by
        the pool construction (allocation), so accounting == allocation by
        construction.

        One activation buffer per slot, each the K and V buffers' layout with
        one unit per block instead of two (an activation block is half of a
        KV block): `n_blocks_total · k_bytes` == 0.5·kv per layer, n_slots of
        them — exactly what the solver folds into the per-layer M_slot
        (`solver.py`: M_slot = weight + kv + 0.5·kv) and what
        _compute_n_slots reserves.

        α does not enter. A layer at α<1 leaves the end of its activation
        buffer unused; sizing at the α=1 worst case is what makes an
        activation buffer sufficient for any curve and for the zig-zag
        prefill's residual stream. The activation buffers stay
        α-independent because one has to stay valid until the forward pass
        is done with the slot, not only until kv_gen is enqueued, and the
        budget is paid on that basis.
        """
        # The prefill residual stream lives in the n_slots activation buffers
        # themselves (x-wheel), so the slab is exactly n_slots units — the
        # same figure as the 0.5·kv per slot the solver folds into M_slot
        # (`_m_slot`).
        return int(n_slots) * n_blocks_total * k_bytes

    @staticmethod
    def _compute_n_slots(slots_mem_bytes: int,
                             model_config,
                             vllm_config,
                             num_layers: int,
                             max_layer_bytes: int) -> int:
        """Derive n_slots (= the layer-slot count = resident-KV depth, one
        coupled rotation) from the slot memory budget.

        m_w_kv = weight (max_layer_bytes) + M_KV (Eq. 7) = max_batch ×
        ⌈max_seq/block_size⌉·block_size × 2 × n_heads × head_dim × dtype —
        whole blocks, as the solver's `_m_kv`; vLLM's null block is not
        charged;
        act_buffers = _act_buffers_bytes(n) (n activation buffers of 0.5·kv).
        n_slots = largest depth with  n_slots·m_w_kv +
        act_plan(n_slots) ≤ slots_mem_bytes  (act_plan =
        _act_buffers_bytes, the same function that sizes the actual
        reload pool, so accounting == allocation), clamped to
        [1, num_layers] and to any ELASTIC_SLOTS already in the env,
        which acts as an upper bound.
        """
        # max_batch: prefer scheduler's max_num_seqs (concurrent
        # requests), which is the upper bound on the resident KV
        # in the cache. max_num_batched_tokens covers a single
        # forward's token count but not KV residency.
        max_batch = int(vllm_config.scheduler_config.max_num_seqs)
        # max_seq: HF/vLLM expose this as max_model_len.
        max_seq = int(model_config.max_model_len)
        hf_text = model_config.hf_config.get_text_config()
        # OPT and BLOOM are multi-head attention: one K/V head per head.
        hidden_size = getattr(hf_text, "hidden_size")
        n_heads = getattr(hf_text, "num_attention_heads")
        head_dim = hidden_size // n_heads
        dtype_size = torch.empty(0, dtype=model_config.dtype).element_size()
        kv_per_token_per_layer = 2 * int(n_heads) * int(head_dim) * int(dtype_size)
        # Resident KV: only n_slots layers' KV is resident at once, the rest
        # streams in and out of NVMe per layer. This is the single source of
        # truth for what actually fits, and it uses the real allocated length
        # (max_seq == max_model_len, i.e. prompt+gen). Weights and KV rotate
        # together in one coupled slot, so a resident layer costs its
        # KV plus one weight buffer. An ELASTIC_SLOTS already in the env is an
        # upper bound and is capped to what fits; the resolved count is
        # published back to the env so the UMA KV allocator, the weight pool
        # and the loader all agree.
        _n_req = int(os.environ.get("ELASTIC_SLOTS", "0"))
        # M_KV (Eq. 7): reserved in whole blocks of max_seq tokens, as the
        # solver's `_m_kv`; vLLM's null block is not charged.
        import math
        _blk = int(vllm_config.cache_config.block_size)
        _blocks_per_seq = math.ceil(int(max_seq) / _blk)
        m_kv = (int(max_batch) * _blocks_per_seq * _blk
                * int(kv_per_token_per_layer))
        m_w_kv = m_kv + int(max_layer_bytes)
        # Single memory constraint:
        #   n_slots·m_w_kv + act_plan(n_slots) ≤ slots_mem_bytes,
        #   with m_w_kv = weight + batch·KV for one layer. The activation
        #   buffers are charged against the same figure, sized by the same
        #   function that allocates them. The caller's figure is used
        #   directly: on coherent unified-memory hardware there is no GPU
        #   carveout, so the real limit is system RAM and any safety margin
        #   belongs in the figure itself. The gap between that figure and
        #   physical RAM covers the framework, scratch and the OS.
        slots_mem = int(slots_mem_bytes)
        base_fit = min(slots_mem // m_w_kv, int(num_layers))
        # Activation-buffer plan inputs, mirrored from the pool
        # construction: per-layer blocks across the batch and the per-block
        # K-plane bytes (= hidden · block_size · dtype for MHA — identical to
        # the kv_caches shape product the pool uses). No α: an activation
        # buffer is the α=1 worst case by construction.
        _n_blocks_total = _blocks_per_seq * int(max_batch)
        _kb = _blk * (int(kv_per_token_per_layer) // 2)
        max_fit = 0
        for _n in range(int(base_fit), 0, -1):
            _sb = StreamingModelLoader._act_buffers_bytes(
                _n, _n_blocks_total, _kb)
            # Unified activation-buffer reservation, sized at the α=1 worst
            # case of 0.5·kv per resident layer, i.e. 0.5·kv·n_slots. One
            # activation buffer serves both roles: the prefill residual
            # stream (x-wheel, using part of it) and the decode ACT reload
            # landing / kv_gen input (at α≤1 also only part). Sizing it
            # worst-case is what makes it a sufficient buffer for both
            # whatever the α curve is; the cost is the smaller n_slots that
            # results. The physical pool is exactly this reservation
            # (n_slots activation buffers of 0.5·kv, α-independent), so
            # allocation == accounting.
            if _n * m_w_kv + int(_sb) <= slots_mem:
                max_fit = _n
                break
        supplied = _n_req if _n_req > 0 else int(num_layers)
        n_slots_eff = min(int(supplied), int(max_fit))
        if n_slots_eff < 1:
            raise ValueError(
                f"slots_mem_bytes={slots_mem_bytes} cannot hold even one "
                f"resident layer: per-layer KV {m_kv} B "
                f"(max_batch={max_batch} × {_blocks_per_seq}·{_blk} tokens × "
                f"{kv_per_token_per_layer}/tok) + weight {int(max_layer_bytes)} "
                f"B. Lower max_num_seqs / max_model_len or raise the slots memory.")
        if n_slots_eff == 1 and int(num_layers) > 1:
            # One slot is not a rotation: layer p's output lands in the slot
            # of the next visit, which with one slot is p's own — the layer
            # would wait for itself to finish before it may start.
            raise ValueError(
                f"slots_mem_bytes={slots_mem_bytes} leaves n_slots=1 "
                f"(per-layer KV {m_kv} B + weight "
                f"{int(max_layer_bytes)} B + staging): the rotation needs at "
                f"least two slots. Lower max_num_seqs / max_model_len or "
                f"raise the slots memory.")
        # Publish the resolved count, n_slots == L included: that still runs
        # the rotating machinery (per-step save+reload, arrival attention).
        published = str(n_slots_eff)
        prev = str(_n_req) if _n_req > 0 else "0"
        if published != prev:
            logger.info("resolved ELASTIC_SLOTS %s -> %s to fit "
                        "slots_mem (max_fit=%d of %d layers)",
                        prev, published, max_fit, num_layers)
        os.environ["ELASTIC_SLOTS"] = published
        # Weights rotate together with KV: exactly n_slots_eff weight
        # buffers, one per resident layer. The slot memory budget is the
        # ceiling (it caps n_slots_eff via max_fit above), not a target.
        logger.info(
            "LayerSlotPool n_slots=%d (max_fit=%d, num_layers=%d), "
            "chunk=%.2f GiB (weight %.2f + batch·KV %.2f), "
            "resident n_slots·chunk=%.2f GiB, slots_mem=%.2f GiB",
            n_slots_eff, max_fit, num_layers,
            m_w_kv / (1 << 30), int(max_layer_bytes) / (1 << 30),
            m_kv / (1 << 30),
            n_slots_eff * m_w_kv / (1 << 30),
            slots_mem / (1 << 30))
        return n_slots_eff


# ----------------------- callback factories ----------------------------


def _make_pre_cb(layer_idx: int, pool: LayerSlotPool,
                  weight_table: WeightTable, gate_state: dict):
    """Pre-layer callback (called from patched OPTDecoderLayer.forward via
    `module._elastic_pre`). Blocks until the loader publishes a slot holding
    this layer's bytes, binds the layer's params to that slot's CUDA view,
    and readies what this layer's attention reads: at a prefill step its
    block table and slot mapping, at a decode step its arrival context for
    the ELASTIC seam in FlashAttentionImpl.forward."""
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

    def cb(module: nn.Module) -> None:
        slot = pool.await_ready(layer_idx)
        weight_table.bind_layer(layer_idx, slot.t_buf, module)
        setattr(module, "_elastic_active_slot", slot)

        # Publish this layer's arrival context (thread-local) for the ELASTIC
        # seam in FlashAttentionImpl.forward. The batch prefill reloads
        # nothing and attends through vLLM's own call, so it gets none — set
        # explicitly, since the previous batch's last decode ctx is still in
        # place; a decode step gets the gate's step rows.
        from .arrival_attention import ArrivalCtx, set_arrival_ctx
        if gate_state["in_batch_prefill"]:
            # This layer's block table and slot mapping. A block's buffer
            # position is per layer, so both are rebuilt here, for this
            # layer: the table's rows from the host (the batch's blocks'
            # global numbers k through `mem_blocks_np`), then vLLM's own
            # slot-mapping kernel over it into the step's one slot-mapping
            # buffer, which every layer's metadata names. The rewrite is
            # stream-ordered behind the previous layer's K/V writes. The
            # table's last reader was this layer at the previous batch's
            # prefill, gen·L visits back (≥ n_slots): the slot just awaited
            # was released only once the GPU had passed the layer end
            # n_slots visits back, which is behind it.
            _tbl = gate_state["block_tables"]
            _ks = gate_state["prefill_ranks"]
            _nb, _g = _ks.shape
            _tbl["host"][layer_idx, :_nb, :_g] = (
                gate_state["block_load_state"]["storage_layout"]
                .mem_blocks_np(layer_idx, _ks))
            _qsl, _pos, _sm = gate_state["prefill_sm"]
            _bt = _tbl["dev"][layer_idx]
            _nt = int(_pos.shape[0])
            _compute_slot_mapping_kernel[(_nb + 1,)](
                _nt, _nt, _qsl, _pos, _bt, _bt.stride(0),
                int(gate_state["block_size"]), _sm,
                TOTAL_CP_WORLD_SIZE=1, TOTAL_CP_RANK=0,
                CP_KV_CACHE_INTERLEAVE_SIZE=1, PAD_ID=PAD_SLOT_ID,
                BLOCK_SIZE=1024)
            set_arrival_ctx(None)
        else:
            bls = gate_state["block_load_state"]
            step_ctx = gate_state["latest_arrival_ctx"]
            set_arrival_ctx(ArrivalCtx(
                arrival=bls["arrival"], layer_idx=layer_idx,
                row_sids=step_ctx["row_sids"],
                layout=bls["storage_layout"],
                seq_lens=step_ctx["seq_lens"],
                seq_to_row=step_ctx["seq_to_row"],
                block_size=step_ctx["block_size"],
                # UMA scratch the varlen call's index operands are bump-
                # allocated from (`ActBufferPool.arrival_index`).
                act_pool=bls["act_pool"]))

    return cb


def _event_ks(layout, rows, mutable: bool, t0: int = 0, t1=None):
    """The global numbers k of one event's blocks, ascending.

    A row's blocks are the ones its tokens touch, block indices
    `[base // bs, (base + n − 1) // bs]`, and their numbers k are the
    layout's own closed form (`ks_of_tokens` at each block's first token).
    The OFFSET FORMULA says which part of the file each one lies in: a k
    below `n_seq·n_immutable` — N_seq times `n_immutable` (n in Eqs. 10–11),
    not the row's token count `n` above — is an immutable block, one at or
    above it a mutable block, which is where the layout puts a row's mutable
    prompt block. That split is the only selection here — nothing asks
    whether a block is full, whose sequence it is, or what form it takes.
    The order of k is file order inside every block region
    (`act_rank`/`kv_rank` are monotone), so an event's numbers come out in
    the order its requests want them.

    A row is `(row_start, sid, base, n)`; rows outside `[t0, t1)` are not
    this event's."""
    n_seq, n_immutable = layout.n_seq, layout.n_immutable
    bs = int(layout.block_size)
    sr = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    if t1 is not None:
        sr = sr[(sr[:, 0] >= t0) & (sr[:, 0] < t1)]
    j0 = sr[:, 2] // bs
    cnt = np.maximum((sr[:, 2] + sr[:, 3] - 1) // bs + 1 - j0, 0)
    tot = int(cnt.sum())
    if not tot:
        return np.zeros(0, dtype=np.int64)
    first = np.cumsum(cnt) - cnt
    j = (np.arange(tot, dtype=np.int64) - np.repeat(first, cnt)
         + np.repeat(j0, cnt))
    ks = layout.ks_of_tokens(np.repeat(sr[:, 1], cnt), j * bs)
    ks = ks[(ks >= n_seq * n_immutable) == bool(mutable)]
    ks.sort()
    return ks


def _prefill_blocks(layout, layer: int, rows, raw, mutable: bool,
                    t0: int = 0, t1=None):
    """The prefill's blocks of one event as `(k, views, cb)`, in order of k.

    The mapping is the order of k to position in the prefill, and that is
    the whole of it: a global number k names its sequence and its block
    index inside that sequence (`seq_of_ks`, `j_of_ks`), so its bytes start
    at that sequence's row in the batch's hidden plus that many blocks —
    `(row_start + pos·bs)·tok_b`, kb bytes from there.

    A row's mutable prompt block is not a case. Its k lies in the mutable
    range, so the offset formula isolates it, and its bytes come from the
    same mapping (the kb span runs on past the row's own tokens into the
    next sequence's rows — never attended, overwritten as the row appends).
    `views` is a thunk, only called for activation blocks."""
    bs = int(layout.block_size)
    tok_b = layout.token_bytes
    kb = layout.block_bytes
    ks = _event_ks(layout, rows, mutable, t0, t1)
    if not len(ks):
        return []
    starts = np.zeros(layout.n_seq, dtype=np.int64)
    for (row_start, sid, base, n) in rows:
        starts[int(sid)] = int(row_start)
    p0s = (starts[layout.seq_of_ks(ks)]
           + layout.j_of_ks(ks) * bs) * tok_b
    out = []
    for k, p0 in zip(ks.tolist(), p0s.tolist()):
        def views(p0=p0):
            return [raw[p0:p0 + kb]]
        out.append((k, views, layout.mem_block(layer, k)))
    return out


def _decode_blocks(layout, layer: int, rows, raw):
    """A decode step's blocks as `(k, views, cb)`, in order of k — each
    row's current block, the mutable block that receives the new token. The
    same mapping, read in the activation buffer instead of the pack: an
    activation block sits at `act_rank(k)·kb`, where the reload landed its
    earlier rows and where this step's row was appended. The request carries
    the block."""
    al = layout.alpha_of_layer(layer)
    kb = layout.block_bytes
    ks = _event_ks(layout, rows, True)
    out = []
    for k in ks.tolist():
        ar = act_rank(k, al) if block_is_act(k, al) else -1

        def views(ar=ar):
            place = ar * kb
            return [raw[place:place + kb]]
        out.append((k, views, layout.mem_block(layer, k)))
    return out


def _event_writes(router, layer: int, blocks, act: bool = True,
                  kv: bool = True):
    """One production event's write requests: `(act_entries, kv_entries)`,
    each `[(file offset, [views]), …]`.

    THE FILE OFFSET DECIDES THE BOUNDARY. A block extends the block
    region's open request only while its offset is that request's running
    end; otherwise it starts a new one, wherever its bytes are. The
    isolation is the layout's offset formula and nothing else — a form, a
    sequence or a block being the row's unfilled last one buys no special
    path here, and memory adjacency is never looked at. In practice an
    event's blocks of one block region do form a single range (a span's
    sequences own the immutable numbers [s0·n, (s1+1)·n), the mutable prompt
    blocks the first N_seq numbers of the mutable range, a decode step N_seq
    adjacent numbers, and the form rule keeps each block region's subset
    consecutive), so this usually yields one request per block region. ACT
    requests first, then K, then V."""
    layout = router.layout
    al = layout.alpha_of_layer(layer)
    vd = layout.v_delta(layer)
    kb = layout.block_bytes
    out = {"a": [], "k": [], "v": []}
    end = {"a": -1, "k": -1, "v": -1}     # the open request's running end

    def add(key, off, views):
        cur = out[key]
        if cur and off == end[key]:
            cur[-1][1].extend(views)
        else:
            cur.append((off, list(views)))
        # Every block of every block region is one `kb` unit, so the next
        # offset a request can absorb is arithmetic, not a sum over the
        # views.
        end[key] = off + kb * len(views)

    for (k, views, cb) in blocks:
        if block_is_act(k, al):
            if act:
                add("a", layout.act_offset(layer, k), views())
        elif kv:
            off = layout.k_offset(layer, k)
            kvw, vvw = router.kv_block_views(layer, cb)
            add("k", off, [kvw])
            add("v", off + vd, [vvw])
    return out["a"], out["k"] + out["v"]


def _make_attn_done_cb(layer_idx: int, gate_state: dict, save_ctx: dict):
    """Span KV enqueue at the point of production.

    Called right after self_attn in the layer body: the QKV projection has
    run and reshape_and_cache has landed this span's K/V in the slot's
    cache pages, so the bytes the save DMA-reads are final here. Complete
    (immutable) blocks only — every KV block of the span's rows, to its
    offset, one request per block region (a contiguous offset range) in
    offset order (`_event_writes`). The ticket is published and the run
    enqueued right here, so a span's FIFO order is ACT (at its LN) → K → V
    and nothing is stashed for a later callback. The request carries its own
    ticket, marked right here on the stream (`enqueue_marked`), so it goes
    the moment the GPU passes this point. The rows' mutable prompt blocks
    are not this span's: their numbers are in the mutable range, which
    numbers a block index across the whole batch (N_seq·n + seq), so those
    N_seq blocks are one run only once every sequence has passed — the layer
    end (`_make_post_cb`)."""
    def cb(module) -> None:
        if not gate_state["in_batch_prefill"]:
            return
        # Under the zig-zag prefill the span is the GPU batch; the plain
        # prefill is one span, the whole batch.
        span = getattr(module, "_elastic_gpu_batch_span", None) or (0, None)
        router = save_ctx["write_router"]
        step_rows = gate_state["latest_step_rows"]
        t0, t1 = span
        L = router.layout
        _a, kv_e = _event_writes(
            router, layer_idx,
            _prefill_blocks(L, layer_idx, step_rows, None, False, t0, t1),
            act=False)
        if kv_e:
            router.enqueue_marked(layer_idx, "kv_batch", kv_e)
    return cb


def _n_views(entries) -> int:
    """How many blocks a list of requests carries: one view per block."""
    return sum(len(v) for _, v in entries)


def _make_post_cb(layer_idx: int, pool: LayerSlotPool, save_ctx: dict):
    """Post-layer callback (called from patched OPTDecoderLayer.forward
    via `module._elastic_post`). Marks the layer end on the forward pass's
    compute stream; the writeback hears it pass, hands the layer's writes to
    the NVMe engine, and releases the slot once they have landed. Before the
    mark it registers those writes: a prefill's mutable prompt blocks (every
    sequence's, one batch per kind), or a decode step's last N_seq blocks (the
    current block of every row: ACT from its place in the activation buffer,
    KV from its cache block)."""

    _BEACON = os.environ.get("ELASTIC_HOOK_BEACON")

    def cb(module: nn.Module) -> None:
        slot = module._elastic_active_slot
        # The layer end. One ticket names it, in this order: reserve it, hand
        # over the writes it licenses, stamp the slot with it (`done_seq`,
        # then `forward_done_visit`), and only then mark it on the stream.
        # When the GPU passes the mark the writeback hears it, and
        # everything the mark licenses is already registered — so the
        # writes go the moment the layer's compute is over, and not a moment
        # before.
        router = save_ctx["write_router"]
        done = router.reserve()
        gate_state = save_ctx["gate_state"]
        step_rows = gate_state["latest_step_rows"]
        L = router.layout
        raw = gate_state["block_load_state"]["act_pool"] \
            .act_buffer(layer_idx).raw_buf
        # ACT first, KV last: the layer's file space on disk is the block
        # regions `[ACT][K][V]`, and every producer emits in that order (the
        # reload walks the file the same way). The descriptor that pins
        # the activation buffer — the one the slot's release
        # condition waits on — goes to the front, so it is held the
        # shortest time. Byte-identical either way.
        if gate_state["in_batch_prefill"]:
            # Prefill layer end: this event owns the rows' numbers in the
            # mutable range — each row's mutable prompt block; together they
            # are the first N_seq numbers of that range, N_seq·n + seq. Those
            # are N_seq adjacent numbers across the batch, so the run is whole
            # only now that every sequence has passed: the later number is why
            # it leaves here, not a rule of its own. One request per block
            # region. A mutable prompt block that is an activation block is
            # a whole block from where the LayerNorm wrote it (the pack; it
            # runs on past the row's own tokens, never attended, overwritten
            # as the row appends); a KV one is its cache block.
            act_e, kv_e = _event_writes(
                router, layer_idx,
                _prefill_blocks(L, layer_idx, step_rows, raw, True))
            if act_e:
                router.enqueue_writes(layer_idx, "act_batch", act_e,
                                      gate_seq=done, path="tailp")
                if _BEACON:
                    try:
                        with open(_BEACON, "a") as fh:
                            fh.write(f"tail_write path=tailp "
                                     f"L={layer_idx} "
                                     f"blocks={_n_views(act_e)} "
                                     f"entries={len(act_e)}\n")
                    except Exception:
                        pass
            if kv_e:
                router.enqueue_writes(layer_idx, "kv_batch", kv_e,
                                      gate_seq=done)
        else:
            # Decode layer end: each row's current block (the step's last
            # N_seq blocks), every one to its offset, as one request per block
            # region. An activation block sits at its place in the
            # activation buffer — the reload put its earlier rows there and
            # `arrival_decode` appended this step's row there — so the
            # request carries the block, like the KV side carries its cache
            # block.
            act_e, kv_e = _event_writes(
                router, layer_idx,
                _decode_blocks(L, layer_idx, step_rows, raw))
            if act_e:
                router.enqueue_writes(layer_idx, "act_batch", act_e,
                                      gate_seq=done, path="taild")
                if _BEACON:
                    try:
                        with open(_BEACON, "a") as fh:
                            fh.write(f"taild L={layer_idx} "
                                     f"blocks={_n_views(act_e)} "
                                     f"entries={len(act_e)}\n")
                    except Exception:
                        pass
            if kv_e:
                # Zero-copy — reads directly from the live slot's K and V
                # buffers; the slot's release condition holds the buffers
                # until this write lands.
                router.enqueue_writes(layer_idx, "kv_batch", kv_e,
                                      gate_seq=done)
        # Resident KV save-before-reuse is enforced by the slot release
        # (a slot is not handed out until every save reading its buffers
        # has its CQE), so the forward pass only enqueues here and is never
        # gated on disk write latency.
        #
        # Stamp the slot, then mark the layer end. The stamp (`done_seq`,
        # then `forward_done_visit`) is in place before the mark can fire,
        # so when the writeback hears it the slot already says whose layer
        # end it was; the writes above were registered before either.
        # Nothing here waits: the slot is released only once those writes
        # have landed, which is the writeback's to see.
        module._elastic_active_slot = None
        pool.stamp_forward_done(slot, done)
        router.mark(done)

    return cb


def _capture_ln_save(gate_state: dict, save_ctx: dict, layer_idx: int,
                     hidden_in: torch.Tensor, ln) -> torch.Tensor:
    """The capture LayerNorm + the plain prefill's ACT save.

    In a prefill the LN is computed straight into the layer's own activation
    buffer via aten.native_layer_norm.out (bit-equal to nn.LayerNorm). It has
    to land there: a plain device allocation's virtual address is not
    readable by the host-side DMA path, while registered UMA memory is, so
    only the activation buffer can be a write source. Attention consumes the
    returned view; the write router DMAs whole-block views of the same
    bytes. Nothing is gathered, packed or copied. ACT bytes on disk are the
    post-LN activation y, and the reload runs kv_gen on y directly (no LN on
    read).

    Where the rows go, and what leaves from here:
      * prefill (the plain path, not the zig-zag prefill — the x-wheel owns
        that): the activation buffer is the whole-batch hidden matrix and
        this call's rows are its rows [0, n). The rows' immutable blocks'
        numbers k from `_event_ks`, the activation blocks among them from
        buffer rows `row_start + pos·bs` to `act_offset(l, k)`, are enqueued
        right away, one request per block region via `_event_writes`;
        each row's mutable prompt block is the layer end's (`_make_post_cb`),
        with every other sequence's.
      * decode: the activation buffer holds the layer's ACT block region as
        whole blocks by rank (a block's place is `act_rank·kb`, where the
        reload landed it). A block is stored whole, empty positions included,
        so a token grows into its own block — and that is where the row
        goes, with the step's own slot map: `mem_block·bs + so` is vLLM's
        KV slot and `mem_block = 1 + act_rank`, so the buffer row is that
        number minus `bs`. Same shape as `reshape_and_cache` on the KV side,
        and no index of its own. It goes in when the block's ACT load has
        completed (`arrival_decode`, from `ctx.act_rows` left here), not at
        this LN. Nothing is enqueued here; the layer-end post cb writes each
        block as one page-aligned run.

    Activation-buffer reuse safety comes from the slot rotation: a slot is
    released only after the layer's writes have landed. The forward pass
    never waits."""
    router = save_ctx["write_router"]
    act_pool = gate_state["block_load_state"]["act_pool"]
    L = router.layout
    tok_b = L.token_bytes
    n = int(hidden_in.shape[0])
    d = int(hidden_in.shape[-1])
    _prefill = gate_state["in_batch_prefill"]
    # The LN's statistics are write-only targets; the activation buffer's
    # own pair, cut to exactly `n` rows (an oversized `out` is resized in
    # place).
    mean, rstd = act_pool.stats(layer_idx, n)
    mean, rstd = mean[:n], rstd[:n]
    if _prefill:
        # The activation buffer is the whole-batch hidden matrix; the save
        # cuts activation blocks out of it without moving a byte.
        y = (act_pool.hidden_span(layer_idx, n * tok_b)
             .view(hidden_in.dtype).view(n, d))
        torch.ops.aten.native_layer_norm.out(
            hidden_in.reshape(n, d), ln.normalized_shape, ln.weight, ln.bias,
            float(ln.eps), out0=y, out1=mean, out2=rstd)
    else:
        # Decode: the step appends one row to each sequence's current
        # block. `y` is the dense activation attention consumes; each row
        # whose current block is an activation block goes into its place in
        # that block in the activation buffer, where the reload puts the rest
        # of that block, so the layer end writes each block whole. It goes in
        # once that block's ACT load has completed — `arrival_decode`
        # appends it right after taking in the item that carries the block;
        # the loader reloads every block of the step, this one included.
        # Writing it here, at the LN, would race the reload, which rewrites
        # the whole block and can land after the LN.
        #
        # No table and no sequence coordinate: the row's place comes from
        # the step's own slot map — the one vLLM's `reshape_and_cache` uses
        # for the new token's K/V. `mem_block = 1 + act_rank` for an activation
        # block, so the buffer row is that vLLM slot minus `bs`.
        y = torch.empty((n, d), device=hidden_in.device,
                        dtype=hidden_in.dtype)
        torch.ops.aten.native_layer_norm.out(
            hidden_in.reshape(n, d), ln.normalized_shape, ln.weight, ln.bias,
            float(ln.eps), out0=y, out1=mean, out2=rstd)
        _dev = gate_state["_pl_cur"]
        from .arrival_attention import get_arrival_ctx
        _actx = get_arrival_ctx()
        _rb = act_pool.act_buffer(layer_idx).t_buf
        _nr = int(_rb.numel()) // tok_b
        _rows = _rb[:_nr * tok_b].view(hidden_in.dtype).view(_nr, d)
        _actx.act_rows = (_rows, _dev[layer_idx, :n] - int(L.block_size), y)
    if _prefill:
        # Row r of this call is activation-buffer row r, so a step row's
        # `row_start` names its buffer row and block j of it is
        # `row_start + j·bs`. The immutable activation blocks go to their
        # offsets as one request.
        step_rows = gate_state["latest_step_rows"]
        act_e, _k = _event_writes(
            router, layer_idx,
            _prefill_blocks(L, layer_idx, step_rows,
                            act_pool.act_buffer(layer_idx).raw_buf, False),
            kv=False)
        if act_e:
            router.enqueue_marked(layer_idx, "act_batch", act_e, path="cap")
    return y


def _make_capture_ln(layer_idx: int, gate_state: dict, save_ctx: dict):
    """The decoder layer's pre-attention LayerNorm seam
    (`module._elastic_capture_ln`, called by the patched
    OPTDecoderLayer / BloomBlock in place of its own LN call — vllm
    .../models/{opt,bloom}.py). When the layer's α is > 0 it runs
    `_capture_ln_save`: in a plain prefill the LN lands in the layer's
    DMA-registered activation buffer (no gather copy) and the full
    activation blocks go to the write router straight from it; in decode
    the LN writes a device tensor and leaves the rows in `ctx.act_rows`,
    which `arrival_decode` appends into their blocks once each block's ACT
    load is in. When the layer's α is 0 (no activation block can exist)
    this is the stock LayerNorm call, bit-identical."""

    def cap(module: nn.Module, hidden_in: torch.Tensor) -> torch.Tensor:
        ln = getattr(module, "self_attn_layer_norm", None)      # OPT
        if ln is None:
            ln = module.input_layernorm                         # BLOOM
        if float(gate_state["alpha_curve"][layer_idx]) > 0.0:
            # The layer can hold an activation block. A prefill's LN writes
            # the activation buffer, and the save reads those bytes from it,
            # at the LN for the complete blocks and at the layer end for the
            # mutable prompt blocks; a decode row goes into its block once
            # that block's ACT load is in. Rows with no activation block
            # among them write nothing out (`_event_writes` selects by form,
            # and `arrival_decode` appends only rows whose current block is
            # an activation block), and the LN's bytes are the stock LN's
            # either way.
            return _capture_ln_save(gate_state, save_ctx, layer_idx,
                                    hidden_in, ln)
        return ln(hidden_in)

    return cap


class _XWheel:
    """Prefill x-wheel: the zig-zag prefill's residual stream lives in the
    0.5·kv activation buffers themselves and is carried in pre-LN form. The
    wheel carries x.

    Per span [t0, t1) of layer p (`prep_span`):
      1. act_buffer(p)[t0:t1] holds x_p (written by layer p-1's `out=`
         landing, or by the embed fill for p = 0);
      2. the span's x is copied to a transient, which is the exact
         skip-connection operand;
      3. layer p's own pre-attn LN (its weights are bound — the pre cb ran
         before the GPU-batch loop) reads the copy and writes y straight
         into the activation-buffer rows, which thereby become the ACT the
         save DMA-reads;
      4. `save_span` enqueues the span's activation blocks right there,
         so the write overlaps the span's own attention and FFN. Nothing
         mutates those bytes afterwards — MHA/FFN write only the transient
         and act_buffer(p+1), the last layer only the transient — so the
         descriptor's one deadline is the slot rotation (the slot-release
         condition), not a mid-layer wait.

    On-disk ACT is post-LN, which is what kv_gen on the reload expects.
    Decode does not use the wheel: a decode ACT row is appended into its
    block by `arrival_decode` and the block is saved by the post callback.
    The capture seam (_make_capture_ln) owns the plain (non-zig-zag)
    prefill's ACT save."""

    def __init__(self, gate_state: dict, save_ctx: dict, layers, pool):
        self._gs = gate_state
        self._sc = save_ctx
        self._layers = list(layers)
        self._pool = pool
        # Span transient: one device buffer, grown to the largest span seen
        # (the GPU batches bound it), holding the exact pre-LN x for the skip
        # connections — and at the last layer, once the attention's skip is
        # taken, the layer's output, out of which the decoder gathers every
        # sequence's last row. Dies logically at span end; physically reused.
        self._tmp = None

    def _act_pool(self):
        return self._gs["block_load_state"]["act_pool"]

    def _ln(self, layer_idx: int):
        return self._layers[layer_idx].self_attn_layer_norm

    def prefill(self) -> bool:
        """Is this forward a batch's prefill step? Only that forward runs the
        zig-zag prefill; the gate says which step this is."""
        return self._gs["in_batch_prefill"]

    def gpu_batches(self):
        """The prefill step's GPU batches, (g0, g1, t0, t1) each, computed
        by the gate on the host (`_gpu_batches`); None when the batch is one
        GPU batch and runs the plain layer loop."""
        return self._gs["prefill_mb_groups"]

    def x_view(self, layer_idx: int, t0: int, t1: int, d: int,
               dtype) -> torch.Tensor:
        st = self._act_pool()
        elem = dtype.itemsize
        return st.act_buffer_rows(layer_idx, t0, t1, d * elem, dtype)

    def land_view(self, layer_idx: int, t0: int, t1: int, d: int,
                  dtype) -> torch.Tensor:
        """Where layer `layer_idx`'s span output lands (`out=`): rows
        [t0, t1) of the next visit's activation buffer. That slot may still
        be its previous tenant's, whose saves may still be DMA-reading the
        bytes this output overwrites, so the landing waits for the writeback
        to have released it (`await_released`). Only the zig-zag prefill
        lands in an activation buffer that is not its own layer's; the plain
        prefill and decode never wait."""
        _slot = self._layers[layer_idx]._elastic_active_slot
        self._pool.await_released(_slot.acquired_visit + 1)
        return self.x_view(layer_idx + 1, t0, t1, d, dtype)

    def fill_embed(self, x: torch.Tensor, t0: int, t1: int) -> None:
        """A GPU batch's embedding → act_buffer(0)[t0:t1], plain pre-LN.
        Layer 0's LN happens at its own spans like every other layer's."""
        n = int(x.shape[0])
        d = int(x.shape[-1])
        self.x_view(0, t0, t1, d, x.dtype).copy_(x.reshape(n, d))

    def prep_span(self, layer_idx: int, t0: int, t1: int, d: int, dtype):
        """Open layer `layer_idx`'s span: copy the span's pre-LN x to the
        transient, LN that copy into the activation-buffer rows (reader and
        writer are different tensors — no in-place aliasing), enqueue the
        span's ACT, and hand back (y_view, exact_residual)."""
        st = self._act_pool()
        n = t1 - t0
        x = self.x_view(layer_idx, t0, t1, d, dtype)
        if (self._tmp is None or self._tmp.dtype != dtype
                or self._tmp.shape[1] != d or self._tmp.shape[0] < n):
            self._tmp = torch.empty((n, d), device="cuda", dtype=dtype)
        res = self._tmp[:n]
        res.copy_(x)
        ln = self._ln(layer_idx)
        # out1/out2 are write-only: `native_layer_norm.out` needs targets
        # for mean/rstd and nothing ever reads them back. They are the
        # span's, not the activation buffer's — this call norms `n` rows —
        # and the views are cut to exactly `n` because an oversized `out` is
        # resized in place by aten (deprecated, and it would shrink the
        # cached pair under the next longer span).
        mean, rstd = st.stats(layer_idx, n)
        torch.ops.aten.native_layer_norm.out(
            res, ln.normalized_shape, ln.weight, ln.bias,
            float(ln.eps), out0=x, out1=mean[:n], out2=rstd[:n])
        self.save_span(layer_idx, t0, t1)
        return x, res

    def save_span(self, layer_idx: int, t0: int, t1: int) -> None:
        """Enqueue the span's full activation blocks straight from the
        activation-buffer bytes the LN just wrote (ticket published right
        here, so the write overlaps the span's own attention and FFN).

        The span's immutable blocks' numbers k from `_event_ks`, the
        activation blocks among them from buffer rows `row_start + pos·bs`
        (kb bytes each) to `act_offset(l, k)`, one request per block region
        via `_event_writes`. The rows' mutable prompt blocks are not this
        span's: the layer end saves them all together (`_make_post_cb`)."""
        if float(self._gs["alpha_curve"][layer_idx]) <= 0.0:
            return
        st = self._act_pool()
        router = self._sc["write_router"]
        step_rows = self._gs["latest_step_rows"]
        L = router.layout
        # The wheel's y lives at the absolute rows [t0:t1] of the activation
        # buffer — a step row's `row_start` is that row, so block j of a row
        # is buffer row `row_start + j·bs`. The span's complete activation
        # blocks go to their offsets as one request, from the buffer rows
        # as-is (LOADER_TIMING's `srcmis` reports a non-page-aligned
        # source phase).
        act_e, _k = _event_writes(
            router, layer_idx,
            _prefill_blocks(L, layer_idx, step_rows,
                            st.act_buffer(layer_idx).raw_buf, False, t0, t1),
            kv=False)
        if not act_e:
            return
        router.enqueue_marked(layer_idx, "act_batch", act_e, path="xw")


_WORKER_STREAM_TRIED = False


def _use_worker_priority_stream() -> None:
    """Move the forward thread's kernels onto a high-priority CUDA stream.

    kv_gen runs on the kv_gen stream at priority 0, the least priority CUDA
    offers, so it cannot be pushed any lower. The forward pass runs on the
    default stream, also priority 0. Equal priority gives the forward pass
    no room: neither side preempts the other, so the forward pass's thread
    blocks queue behind whatever kv_gen thread blocks are already resident
    on a saturated device. The forward pass is the critical path — the
    loader can only refill the slot the forward pass vacates — so the
    forward pass is the side raised.

    Called on the forward thread before its first kernels — from the arrival
    gate, once per forward pass. `set_stream` is thread-scoped and sticky, so
    one call covers every kernel the forward thread launches from then on.
    The one-shot hand-off event orders the new stream after whatever the
    default stream was still running at the switch."""
    global _WORKER_STREAM_TRIED
    if _WORKER_STREAM_TRIED:
        return
    _WORKER_STREAM_TRIED = True
    # priority_range() is (least, greatest); the forward pass takes -1.
    s = torch.cuda.Stream(
        priority=max(-1, torch.cuda.Stream.priority_range()[1]))
    # Hand off: everything already queued on the stream we are leaving
    # must complete before the new stream's first kernel.
    _ev = torch.cuda.Event()
    _ev.record(torch.cuda.current_stream())
    s.wait_event(_ev)
    torch.cuda.set_stream(s)


def _per_layer_slot_mappings(state: dict):
    """This step's slot mapping for every layer, from the layout's closed
    form: the token at position pos of sequence seq goes to block
    `mem_block(l, ks_of_tokens(seq, pos))` of layer l, offset `pos % bs`.

    Returns {layer_name: (T,) int64 view}. A decode step carries one token
    per request and runs unpadded (eager mode), so T ≤ max_num_reqs. The
    views live in one registered UMA buffer, (2, L, max_num_reqs) int64,
    filled from the host and read by the device without a copy; two halves
    alternate per step so a step's fill never overwrites what the previous
    step's kernels may still read."""
    step_rows = state["latest_step_rows"]
    lay = state["block_load_state"]["storage_layout"]
    runner = state["runner"]
    bs = int(state["block_size"])
    # Per-token (sid, pos), expanded from the step rows: the rows tile
    # [0, T) in flat-row order, so a token's flat index minus its row's
    # start is its offset into the row and its position is base plus that.
    sr = np.asarray(step_rows, dtype=np.int64).reshape(-1, 4)
    nt = sr[:, 3]
    T = int(nt.sum())
    ent = state.get("_pl_sm")
    if ent is None:
        from vllm.model_executor.models.utils import extract_layer_index
        names = list(runner.kv_cache_config.kv_cache_groups[0].layer_names)
        idx = [extract_layer_index(n) for n in names]
        L = len(runner.kv_caches)
        cap = int(runner.input_batch.max_num_reqs)
        keep = state.setdefault("_pl_sm_keep", [])
        t = _alloc_uma_buffer(2 * L * cap * 8, keep)
        arr = keep[-1][0]
        ent = {"L": L, "names": names, "idx": idx,
               "dev": t.view(torch.int64).view(2, L, cap),
               "host": np.frombuffer(arr, dtype=np.int64).reshape(2, L, cap),
               "flip": 0}
        state["_pl_sm"] = ent
    f = ent["flip"] ^ 1
    ent["flip"] = f
    host = ent["host"][f]
    dev = ent["dev"][f]
    sid = np.repeat(sr[:, 1], nt)
    pos = np.repeat(sr[:, 2] - sr[:, 0], nt) + np.arange(T, dtype=np.int64)
    so = pos % bs
    k = lay.ks_of_tokens(sid, pos)
    for li in range(ent["L"]):
        host[li, :T] = lay.mem_blocks_np(li, k) * bs + so
    # The decode ACT capture puts its row at the same slot this map names
    # (`slot - bs`), so the map is published rather than rebuilt.
    state["_pl_cur"] = dev
    return {name: dev[i, :T] for name, i in zip(ent["names"], ent["idx"])}


def _gpu_batches(q_start: list[int], gpu_batch_tokens: int):
    """The zig-zag prefill's GPU batches, on the host: WHOLE sequences
    (each self-contained under causal attention) grouped while a GPU batch's
    token count stays <= `gpu_batch_tokens`, as (g0, g1, t0, t1) — sequence
    range [g0, g1) spanning token range [t0, t1). None when the batch is one
    GPU batch: it then runs the plain batch-outer loop."""
    n_seq = len(q_start) - 1
    groups = []
    g0 = 0
    while g0 < n_seq:
        g1 = g0 + 1
        while g1 < n_seq and (q_start[g1 + 1] - q_start[g0]) <= gpu_batch_tokens:
            g1 += 1
        groups.append((g0, g1, q_start[g0], q_start[g1]))
        g0 = g1
    return groups if len(groups) > 1 else None


# The latest batch's prefill/decode boundary: a timing event the gate records
# on the forward pass's stream at the batch's first decode step, so on the
# device it sits right behind the prefill step's forward and sampler. The
# benchmark CLI clears it before a generate() and times the decode steps
# from it.
decode_boundary = None


def _make_arrival_gate(state: dict):
    """Build the `_kv_arrival_gate` callable.

    `slot_mappings` is a `dict[int, torch.Tensor]` from the runner
    (kv_cache_group_id → slot mapping tensor, slot id per token, -1 for
    PAD). The gate runs once per forward pass, before any layer. It tells a
    batch's prefill step from a decode step, builds the step rows, and
    returns the per-layer slot mappings and block tables the runner seam
    installs on each layer's metadata. At a prefill step it also opens the
    batch: its seq ids, the layout's declaration, the zig-zag prefill's GPU
    batches and the blocks' global numbers k the pre callbacks map per
    layer. At a decode step it builds the arrival context."""

    def gate(slot_mappings):
        global decode_boundary
        # One-shot, on the forward thread, before any of this pass's kernels.
        _use_worker_priority_stream()
        block_size = state["block_size"]

        # No per-step D2H of slot_mapping: a token's slot offset is
        # `pos % block_size`, and its block's memory position is the
        # layout's closed form per layer — nothing is read off vLLM's block
        # table but the per-row token counts.
        runner = state["runner"]
        input_batch = runner.input_batch
        num_reqs = int(input_batch.num_reqs)
        # The loader reloads `n_seq · c` blocks per step, so every sequence
        # of the layout must be a row of every step: a smaller batch, or a
        # row that finishes early, would leave delivered blocks with no row.
        _n_seq = state["block_load_state"]["storage_layout"].n_seq
        if num_reqs != _n_seq:
            raise ValueError(
                f"elastic_streaming needs every batch to have exactly "
                f"max_num_seqs={_n_seq} prompts, each running to its "
                f"max_tokens (the same for all, with ignore_eos=True); this "
                f"step has {num_reqs} rows")
        # Which step this is. A batch's prefill is its first step: every row
        # is a new request with nothing computed yet, where a decode row has
        # at least its prompt behind it. The caller adds the whole batch
        # before the first step and the vLLM engine config admits all of it in
        # that step, so every row of a step is in the same world.
        prefill = not input_batch.num_computed_tokens_cpu[:num_reqs].any()
        was = state["in_batch_prefill"]
        state["in_batch_prefill"] = prefill

        seq_id_alloc = state["seq_id_alloc"]
        # The batch rows, condensed by vLLM before the forward: row r is
        # request `req_ids[r]`.
        req_ids = input_batch.req_ids[:num_reqs]
        if prefill:
            # A new batch: its sequences take ids 0..N_seq-1 in row order,
            # which is vLLM's flat packing order, so a GPU batch of consecutive
            # rows owns one range of global numbers k.
            seq_id_alloc.assign(req_ids)
        # Scheduled tokens per req (CPU-side, no D2H — vLLM keeps
        # num_scheduled_tokens.np up to date via copy_to_gpu's
        # corresponding host write).
        n_sched_np = runner.num_scheduled_tokens.np[:num_reqs]
        # The step rows: one per request row in vLLM's flat packing
        # order (req 0's scheduled tokens, then req 1's, …), as
        # (row_start, seq_id, base_pos, n_tokens). `row_start` is the
        # row's first flat hidden row, so a save indexes the hidden
        # without touching query_start_loc; the blocks a row's tokens
        # touch are the layout's closed form.
        step_rows: list[tuple[int, int, int, int]] = []
        _flat = 0
        # Arrival context: per-row (seq_id, seq_len incl. this step's
        # tokens) in the attention query-row order (flat packing is
        # req-index order), so the forward pass's arrival_decode owns
        # what/how-much it attends.
        _arr_sids: list[int] = []
        _arr_seqlens: list[int] = []
        _arr_seq_to_row: dict[int, int] = {}
        for r, rid in enumerate(req_ids):
            sid = seq_id_alloc.get(rid)
            base_pos = int(input_batch.num_computed_tokens_cpu[r])
            n_sched = int(n_sched_np[r])
            step_rows.append((_flat, sid, base_pos, n_sched))
            _flat += n_sched
            _arr_seq_to_row[sid] = len(_arr_sids)
            _arr_sids.append(sid)
            _arr_seqlens.append(base_pos + n_sched)
        state["latest_step_rows"] = step_rows
        state["latest_arrival_ctx"] = {
            "row_sids": _arr_sids,
            "seq_lens": _arr_seqlens,
            "seq_to_row": _arr_seq_to_row,
            "block_size": block_size,
        }

        if prefill:
            # The batch's declaration, made before anything takes an offset
            # from it: its prompt length and its decode steps (every request
            # runs to its max_tokens — an offline batch ignores EOS — so the
            # longest is the batch's; the loader takes it, in batch order,
            # when it leaves the batch's prefill step), and the prompt split
            # `n_immutable` (n in Eqs. 10–11) every global number k depends
            # on. Every prompt of a batch has the same length.
            lay = state["block_load_state"]["storage_layout"]
            plen = n_sched_np.astype(np.int64)
            if not bool((plen == plen[0]).all()):
                raise ValueError(
                    f"elastic_streaming needs every prompt of a batch to "
                    f"have the same length; got lengths "
                    f"{sorted(set(plen.tolist()))}")
            lay.declare_batch(int(plen.max()), max(
                int(runner.requests[rid].sampling_params.max_tokens)
                for rid in req_ids))
            lay.set_prompt_len(int(plen[0]))
            state["prefill_mb_groups"] = _gpu_batches(
                [0] + np.cumsum(plen).tolist(), state["prefill_gpu_batch_tokens"])
            # Every sequence's blocks' global numbers k, (N_seq, G): block j of
            # sequence seq is the block holding position j·bs. Each layer's pre
            # callback maps them to that layer's buffer blocks (its block
            # table), then runs vLLM's slot-mapping kernel over the table
            # into the step's one slot-mapping buffer — so every layer's
            # metadata names that buffer, and the queries' starts and
            # positions the kernel reads are the step's own.
            _g = -(-int(plen.max()) // int(block_size))
            state["prefill_ranks"] = lay.ks_of_tokens(
                np.arange(num_reqs, dtype=np.int64)[:, None],
                np.arange(_g, dtype=np.int64)[None, :] * int(block_size))
            state["prefill_sm"] = (runner.query_start_loc.gpu[:num_reqs + 1],
                                   runner.positions[:int(plen.sum())],
                                   slot_mappings[0])
            _pl = {name: slot_mappings[0] for name, _i in state["layer_names"]}
            # Every layer's block table, its first N_seq rows at full width —
            # the batch's, for all of its steps. The prefill's attention reads
            # the K/V back through it; the decode attends in arrival order
            # and builds its own.
            _tables = state["block_tables"]["dev"]
            state["batch_block_tables"] = {
                name: _tables[i, :num_reqs]
                for name, i in state["layer_names"]}
        else:
            # This step's per-layer slot mappings: a block's position is per
            # layer (forms differ by α), so reshape_and_cache needs one per
            # layer.
            _pl = _per_layer_slot_mappings(state)
            if was:
                # The batch's first decode step. Mark the boundary on the
                # forward pass's stream, then return the allocator blocks the
                # prefill's activations claimed, so they are not held
                # through the decode. That synchronises the device (the
                # cached blocks' stream-use events, then cudaFree), once per
                # batch.
                decode_boundary = torch.cuda.Event(enable_timing=True)
                decode_boundary.record()
                torch.cuda.empty_cache()
        return _pl, state["batch_block_tables"]

    return gate


def _log_meta_state(model: nn.Module) -> None:
    n_meta = n_real = 0
    n_meta_bytes = 0
    for _, p in model.named_parameters():
        if p.is_meta:
            n_meta += 1
            n_meta_bytes += p.numel() * p.element_size()
        else:
            n_real += 1
    logger.info(
        "StreamingModelLoader meta-init: %d meta params "
        "(%.2f GiB virtual), %d real params",
        n_meta, n_meta_bytes / (1 << 30), n_real,
    )
