"""WeightTable — maps a `stage_opt`-produced manifest to per-layer parameter
slice descriptors, and binds those slices onto a real PyTorch model so its
`.forward(...)` reads weights out of a streaming buffer instead of memory
allocated once at init.

That indirection is what allows a model larger than memory to run: only the
layers in the pool's slots are resident at any moment, and the model object
never owns its own weight storage.

Lifecycle, per layer i:

  1) At setup, `WeightTable.from_manifest(manifest)` parses the JSON into
     `LayerSpec[i]`, giving each parameter an `inner_offset` relative to the
     start of that layer's contiguous on-disk range (Weight_i). Offsets are
     relative, not absolute, precisely because a layer can be loaded into any
     buffer.

  2) At runtime, the loader acquires a layer slot of the slot pool (a fixed
     number of persistent host-registered buffers, sized from the slot memory
     budget) and the NVMe engine reads layer i's bytes into its weight buffer;
     before the layer's forward the forward pass calls `bind_layer(i,
     slot_buffer, model_layer)`. That rewrites every `Parameter.data` to a
     view into the slot — a storage pointer swap, no copy.

Slots are anonymous and reused across layers — there is no fixed
layer-to-slot mapping. Ordering between the forward pass and the refill of a
slot belongs to the slot pool, not here: binding a slot before the pool
reports the layer ready reads bytes that are still landing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


# The blob's only dtype (stage_opt writes float16; the model loader checks
# the manifest says so). Every view of a slot's weight bytes uses it.
DTYPE = torch.float16


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _replace_param(module: torch.nn.Module, dotted_name: str,
                   real_tensor: torch.Tensor) -> None:
    """Swap the `nn.Parameter` at `dotted_name` for one wrapping
    `real_tensor`. PyTorch's set_data type check rejects assigning a real
    tensor to a meta parameter's `.data`, so a meta Parameter has to be
    replaced outright. Preserves `requires_grad` from the existing parameter.
    """
    parent_path, _, attr = dotted_name.rpartition(".")
    parent = (module.get_submodule(parent_path)
              if parent_path else module)
    old = getattr(parent, attr)
    new_p = torch.nn.Parameter(
        real_tensor,
        requires_grad=bool(getattr(old, "requires_grad", False)))
    setattr(parent, attr, new_p)


@dataclass(frozen=True)
class ParamSpec:
    name: str                  # e.g. 'self_attn.qkv_proj.weight'
    inner_offset: int          # bytes from the layer's file_start
    nbytes: int
    shape: tuple


@dataclass(frozen=True)
class LayerSpec:
    file_start: int            # absolute file offset of param 0
    bytes_padded: int          # layer bytes padded to the alignment (what is read)
    params: tuple              # tuple[ParamSpec, ...]


class WeightTable:
    def __init__(self, layers: List[LayerSpec]):
        self.layers = layers
        self.max_layer_bytes = (max(L.bytes_padded for L in layers)
                                if layers else 0)
        # `bind_layer` runs once per layer per forward step, so anything it
        # does per parameter happens tens of thousands of times per run.
        # Resolving each name through `model_layer.get_parameter(name)` walks
        # the module tree every time, which is pure overhead for a mapping
        # that never changes. Cache the resolved `Parameter` objects keyed by
        # (id(model_layer), param_name): the first bind populates the cache,
        # later binds only assign `param.data = view`.
        self._param_ref_cache: dict[tuple[int, str], torch.nn.Parameter] = {}

    # ----------- construction from manifest --------------------------

    @classmethod
    def from_manifest(cls, manifest: dict) -> "WeightTable":
        """Parse the JSON manifest produced by `stage_opt`."""
        alignment = int(manifest["alignment"])

        layers: List[LayerSpec] = []
        for li in manifest["layers"]:
            raw_params = li["params"]
            file_start = int(raw_params[0]["offset"])
            params: List[ParamSpec] = []
            for pm in raw_params:
                off = int(pm["offset"])
                params.append(ParamSpec(
                    name=str(pm["name"]),
                    inner_offset=off - file_start,
                    nbytes=int(pm["nbytes"]),
                    shape=tuple(pm["shape"]),
                ))
            last = params[-1]
            bytes_padded = _align_up(last.inner_offset + last.nbytes,
                                     alignment)
            layers.append(LayerSpec(
                file_start=file_start,
                bytes_padded=bytes_padded,
                params=tuple(params),
            ))

        return cls(layers=layers)

    # ----------- per-layer binding -----------------------------------

    def bind_layer(self,
                   layer_idx: int,
                   buf: torch.Tensor,
                   model_layer: torch.nn.Module) -> None:
        """Re-point every `Parameter.data` in `model_layer` at a view into
        `buf` (contiguous uint8, on the target device).

        The first bind of a layer resolves its parameters and checks them
        against the manifest: it raises if `model_layer` lacks a parameter
        the manifest expected, or if a dtype disagrees. Either means the
        staged blob and the live model disagree about the layer's layout,
        and continuing would bind parameters to the wrong bytes — which
        produces plausible-looking garbage output rather than an error.
        Later binds only re-point the cached Parameters.
        """
        spec = self.layers[layer_idx]
        layer_id = id(model_layer)
        for p in spec.params:
            cache_key = (layer_id, p.name)
            view = (buf[p.inner_offset:p.inner_offset + p.nbytes]
                    .view(DTYPE).view(p.shape))
            param = self._param_ref_cache.get(cache_key)
            if param is not None:
                param.data = view
                continue
            try:
                param = model_layer.get_parameter(p.name)
            except AttributeError as e:
                raise KeyError(
                    f"model_layer has no parameter {p.name!r} "
                    f"(layer {layer_idx}): {e}") from e
            if view.dtype != param.dtype:
                raise ValueError(
                    f"dtype mismatch for {p.name} (layer {layer_idx}): "
                    f"blob {view.dtype} vs model {param.dtype}")
            param.data = view
            self._param_ref_cache[cache_key] = param
