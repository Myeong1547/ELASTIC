"""Stage a HuggingFace checkpoint into a packed NVMe blob + manifest.

The elastic runtime reads model weights off NVMe through the io_uring +
O_DIRECT NVMe engine, which needs (offset, size) pairs it can DMA directly
into a buffer. A HuggingFace checkpoint is a dict of tensors spread over
shards in no particular order, which the loader cannot use. This tool
performs the conversion once per model, offline.

Packing per layer is the point: one layer's parameters are laid out
contiguously, in the order the runtime binds them, so a decode step reads
each layer with a single sequential request rather than one request per
parameter.

Every tensor is written as float16 and padded to a 4 KiB boundary.

Layout of the output blob (OPT names; BLOOM has the same structure):

    [extras]
      embed_tokens, embed_positions, final_layer_norm, lm_head
                                       ← packed in this order, each
                                         aligned up to 4096 bytes

    [layer 0 weights]
      self_attn_layer_norm.{weight,bias}
      self_attn.qkv_proj.{weight,bias}   ← q ++ k ++ v, fused (vLLM layout)
      self_attn.out_proj.{weight,bias}
      final_layer_norm.{weight,bias}
      fc1.{weight,bias}, fc2.{weight,bias}

    [layer 1 weights]
      … same per-param order …

    [layer L-1 weights]
      …

Manifest schema (manifest.json):

    {
      "model": "facebook/opt-125m",
      "config": {model_type, hidden_size, num_hidden_layers,
                 num_attention_heads, vocab_size, ffn_dim, ...},
      "alignment": 4096,
      "dtype": "float16",
      "extras":  [{"name", "offset", "nbytes", "shape", "dtype"}, ...],
      "layers":  [{"layer": i, "params": [{...}, ...]}, ...],
      "total_bytes": N,
      "dummy": true                  (a --dummy staging only)
    }

`name` in extras is the full HF state-dict key (e.g.
``model.decoder.embed_tokens.weight``); in layer params it is the
suffix after ``model.decoder.layers.<i>.`` (e.g. ``fc1.weight``).

Implementation note — the models this project targets do not fit in host
memory, so the full model is never materialized. Each tensor is read
individually from whichever shard holds it (via a lazy shard cache),
written into the blob, and dropped. Peak host memory is therefore about
the size of the single largest tensor, not the size of the model, which
is what lets a machine stage a checkpoint many times its own memory.

Usage:

    python -m elastic.cli.stage_opt \\
        --model facebook/opt-125m \\
        --output /tmp/staged/opt-125m

    # validate the staged blob byte-matches the source tensors:
    python -m elastic.cli.stage_opt ... --verify

    # stage a model whose weights cannot be obtained (e.g. opt-175b):
    # --model is a local directory holding a hand-written config.json, no
    # hub access happens, and every tensor is synthesized at source-init
    # scale. The blob is a full-size, fully-written stand-in that goes
    # through the same write path; only the numeric content differs, so it
    # is a throughput-only artifact whose outputs are meaningless.
    python -m elastic.cli.stage_opt \\
        --model configs/opt-175b --dummy \\
        --output staged/opt-175b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# O_DIRECT needs every offset on a 4 KiB boundary.
_ALIGNMENT = 4096
_DTYPE = torch.float16
_DTYPE_NAME = "float16"

# OPT decoder layer parameter ordering, grouped as the forward pass uses
# them (LN, QKV, OUT, LN, FFN). Each entry is the suffix after
# ``model.decoder.layers.<i>.``. This order defines the on-disk layout, so
# changing it invalidates every staged blob.
#
# vLLM's OPT attention fuses q/k/v into a single fused projection whose
# `.weight` is `cat([q, k, v], dim=0)` and `.bias` the matching bias
# concatenation. Staging the fused form means the runtime can read one
# layer's bytes in a single request and bind them straight to the model's
# parameters — doing the concatenation offline here instead of once per
# layer per step at run time.
OPT_LAYER_PARAMS_VLLM: list[str] = [
    "self_attn_layer_norm.weight",
    "self_attn_layer_norm.bias",
    "self_attn.qkv_proj.weight",   # fused: q ++ k ++ v on dim 0
    "self_attn.qkv_proj.bias",     # fused
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "final_layer_norm.weight",
    "final_layer_norm.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
]


def _build_vllm_layer_tensor(reader: "_TensorReader", layer_idx: int,
                              suffix: str) -> torch.Tensor:
    """Resolve one vLLM-named layer parameter to its (possibly fused) tensor.

    For the fused qkv projection this concatenates the source q/k/v tensors
    on dim 0, reproducing the target parameter's memory layout exactly, so a
    single read of the staged bytes fills it with no further copy. All other
    parameters pass through unchanged."""
    base = f"model.decoder.layers.{layer_idx}"
    if suffix == "self_attn.qkv_proj.weight":
        q = reader.get(f"{base}.self_attn.q_proj.weight")
        k = reader.get(f"{base}.self_attn.k_proj.weight")
        v = reader.get(f"{base}.self_attn.v_proj.weight")
        return torch.cat([q, k, v], dim=0)
    if suffix == "self_attn.qkv_proj.bias":
        q = reader.get(f"{base}.self_attn.q_proj.bias")
        k = reader.get(f"{base}.self_attn.k_proj.bias")
        v = reader.get(f"{base}.self_attn.v_proj.bias")
        return torch.cat([q, k, v], dim=0)
    return reader.get(f"{base}.{suffix}")


# Extras: model-level tensors that are not part of any decoder layer. They
# are packed ahead of the layers' weights, in this order, so that they too are
# read sequentially. Every entry must be present except OPT's lm_head.weight,
# which is tied to embed_tokens and which some checkpoints do not ship.
OPT_EXTRAS_ORDER: list[str] = [
    "model.decoder.embed_tokens.weight",
    "model.decoder.embed_positions.weight",
    "model.decoder.final_layer_norm.weight",
    "model.decoder.final_layer_norm.bias",
    "lm_head.weight",                         # tied to embed_tokens
]

# ---------------------------------------------------------------- BLOOM
# BLOOM's model lives under `transformer`, with decoder layers at
# `transformer.h.<i>`. Its attention also fuses qkv into one parameter, but
# the two sides disagree on the order: the checkpoint interleaves per head
# (num_heads · 3 · head_dim) while vLLM expects the three projections
# concatenated (3 · num_heads · head_dim). Staging rebuilds the target
# order, so again a single read fills the parameter and no conversion
# happens per step. lm_head is tied to the word embeddings and therefore has
# no separate entry. Layer-parameter suffixes are relative to the block
# module.
BLOOM_LAYER_PARAMS_VLLM: list[str] = [
    "input_layernorm.weight",
    "input_layernorm.bias",
    "self_attention.query_key_value.weight",   # rebuilt → [q;k;v] on dim0
    "self_attention.query_key_value.bias",     # rebuilt → [q;k;v]
    "self_attention.dense.weight",
    "self_attention.dense.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]

BLOOM_EXTRAS_ORDER: list[str] = [
    "transformer.word_embeddings.weight",      # lm_head ties to this
    "transformer.word_embeddings_layernorm.weight",
    "transformer.word_embeddings_layernorm.bias",
    "transformer.ln_f.weight",
    "transformer.ln_f.bias",
]


def _bloom_qkv_to_vllm(t: torch.Tensor, num_heads: int,
                       hidden: int) -> torch.Tensor:
    """Convert a fused BLOOM `query_key_value` tensor from the checkpoint's
    per-head interleaving (num_heads · 3 · head_dim) to the [q;k;v]
    concatenation vLLM expects (3 · num_heads · head_dim).

    This must stay equivalent to the conversion vLLM's own loader performs;
    a mismatch here permutes attention heads and produces plausible but
    wrong output rather than an error. Handles both the 2-D weight (dim 0 =
    output) and the 1-D bias."""
    head_dim = hidden // num_heads
    if t.dim() == 2:
        return (t.view(num_heads, 3, head_dim, t.shape[1])
                 .transpose(0, 1).reshape(3 * hidden, t.shape[1]))
    return t.view(num_heads, 3, head_dim).transpose(0, 1).reshape(3 * hidden)


def _build_bloom_layer_tensor(reader: "_TensorReader", layer_idx: int,
                              suffix: str, num_heads: int,
                              hidden: int) -> torch.Tensor:
    base = f"transformer.h.{layer_idx}"
    t = reader.get(f"{base}.{suffix}")
    if suffix in ("self_attention.query_key_value.weight",
                  "self_attention.query_key_value.bias"):
        return _bloom_qkv_to_vllm(t, num_heads, hidden)
    return t


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


# --------------------------------------------------------------------
# Lazy per-shard tensor reader. Two formats × two layouts:
#   safetensors        single  /  sharded (model.safetensors.index.json)
#   pytorch pickle     single  /  sharded (pytorch_model.bin.index.json)
# The HF nn.Module is never instantiated — only the individual tensors that
# are about to be written to the blob. That is what keeps peak host memory
# at one tensor rather than one model.
# --------------------------------------------------------------------

class _LazyShard:
    def __init__(self, path: Path, fmt: str) -> None:
        self.path = Path(path)
        self.fmt = fmt          # 'st' or 'bin'
        self._st = None         # safetensors safe_open context
        self._sd = None         # torch.load(mmap=True) dict

    def open(self) -> "_LazyShard":
        if self.fmt == "st":
            from safetensors import safe_open
            self._st = safe_open(str(self.path), framework="pt")
            self._st.__enter__()
        else:
            self._sd = torch.load(str(self.path), map_location="cpu",
                                  mmap=True, weights_only=True)
        return self

    def close(self) -> None:
        if self._st is not None:
            self._st.__exit__(None, None, None)
            self._st = None
        self._sd = None

    def get(self, key: str) -> torch.Tensor:
        if self._st is not None:
            return self._st.get_tensor(key)
        assert self._sd is not None
        return self._sd[key]


class _TensorReader:
    """Resolve full HF state-dict keys to their backing shard file,
    open shards lazily, materialize one tensor at a time."""

    def __init__(self, model_id: str) -> None:
        from huggingface_hub import snapshot_download
        # Step 1 — fetch only the metadata first, to learn which weight format
        # the repository ships before pulling hundreds of gigabytes of shards.
        meta_root = Path(snapshot_download(
            model_id,
            allow_patterns=["config.json", "*.index.json"]))
        # Step 2 — download exactly one weight format, preferring safetensors
        # to match the resolution order in `_build_weight_map`. Some
        # repositories ship a complete set in both formats; downloading both
        # doubles the transfer for a set of files that is then never read.
        if (meta_root / "model.safetensors.index.json").exists():
            weight_pat = ["*.safetensors", "model.safetensors.index.json"]
        elif (meta_root / "pytorch_model.bin.index.json").exists():
            weight_pat = ["*.bin", "pytorch_model.bin.index.json"]
        else:  # unsharded single-file checkpoint (no index json)
            weight_pat = ["model.safetensors", "pytorch_model.bin"]
        self.root = Path(snapshot_download(
            model_id,
            allow_patterns=["config.json", *weight_pat]))
        self.fmt, self.weight_map = self._build_weight_map()
        self._open_shards: dict[Path, _LazyShard] = {}

    def _build_weight_map(self) -> tuple[str, dict[str, tuple[Path, str]]]:
        """Return (fmt, {canonical_key: (shard_path, key_in_shard)}).

        Checkpoints of the same architecture do not agree on key prefixes —
        some drop the top-level module wrapper and store keys like
        `decoder.layers.0.…` where others use `model.decoder.layers.0.…`.
        Both forms are normalised to one canonical spelling for lookup,
        while the shard-native key is kept alongside it because that is what
        the actual tensor read must use.
        """
        st_idx = self.root / "model.safetensors.index.json"
        if st_idx.exists():
            data = json.loads(st_idx.read_text())
            return "st", self._canonicalise(
                {k: self.root / v for k, v in data["weight_map"].items()})
        st_single = self.root / "model.safetensors"
        if st_single.exists():
            from safetensors import safe_open
            with safe_open(str(st_single), framework="pt") as f:
                raw = {k: st_single for k in f.keys()}
            return "st", self._canonicalise(raw)
        bin_idx = self.root / "pytorch_model.bin.index.json"
        if bin_idx.exists():
            data = json.loads(bin_idx.read_text())
            return "bin", self._canonicalise(
                {k: self.root / v for k, v in data["weight_map"].items()})
        bin_single = self.root / "pytorch_model.bin"
        if bin_single.exists():
            sd = torch.load(str(bin_single), map_location="cpu",
                            mmap=True, weights_only=True)
            raw = {k: bin_single for k in sd.keys()}
            del sd
            return "bin", self._canonicalise(raw)
        raise FileNotFoundError(
            f"no model weights found under {self.root} "
            "(looked for *.safetensors and *.bin)")

    @staticmethod
    def _canonicalise(raw_map: dict) -> dict:
        """raw_map maps shard-native keys → shard path. Reshape to
        canonical_key → (shard_path, shard_native_key)."""
        out: dict[str, tuple[Path, str]] = {}
        for k, path in raw_map.items():
            canon = k
            # Restore the `model.` wrapper when the checkpoint omitted it.
            # `lm_head.weight` legitimately has no wrapper, so it must not be
            # caught by this rule.
            if k.startswith("decoder."):
                canon = "model." + k
            # BLOOM checkpoints often drop the `transformer.` wrapper, giving
            # keys like `h.0.…`. The model's parameters live under
            # `transformer.`, so normalise to that.
            elif k.startswith(("h.", "word_embeddings", "ln_f")):
                canon = "transformer." + k
            out[canon] = (path, k)
        return out

    def _shard(self, path: Path) -> _LazyShard:
        if path not in self._open_shards:
            self._open_shards[path] = _LazyShard(path, self.fmt).open()
        return self._open_shards[path]

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(key)
        path, native = self.weight_map[key]
        return self._shard(path).get(native)

    def close(self) -> None:
        for sh in self._open_shards.values():
            sh.close()
        self._open_shards.clear()

    def __enter__(self) -> "_TensorReader":
        return self

    def __exit__(self, *a) -> None:
        self.close()


class _DummyReader:
    """Synthesize an OPT checkpoint from a local ``config.json``, for models
    whose weight artifacts cannot be used (e.g. opt-175b). Drop-in for
    `_TensorReader`: same `.root/.fmt/.weight_map/.has/.get` surface, so
    `stage()` and `_verify()` run unchanged and the blob takes the same
    write path as a real staging. Only the numeric content differs, and
    blob read-back speed does not depend on content.

    Values are source-init scale so the forward pass stays numerically
    bounded (no NaN/Inf): LayerNorm gamma = 1, every bias = 0, matrices =
    N(0, init_std). Each tensor is seeded from its key name, so a fresh
    reader rebuilds byte-identical tensors and ``--verify`` still proves
    the blob/manifest bookkeeping.
    """

    def __init__(self, config_dir: Path) -> None:
        self.root = Path(config_dir)
        cfg_path = self.root / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"--dummy expects --model to be a local directory holding "
                f"config.json; {cfg_path} does not exist")
        cfg = json.loads(cfg_path.read_text())
        model_type = str(cfg["model_type"]).lower()
        if model_type != "opt":
            raise ValueError(
                f"--dummy supports only OPT configs (got model_type="
                f"{model_type!r})")
        # The synthesized key set below is the large-OPT shape: model-level
        # final_layer_norm present, every linear biased.
        hidden = int(cfg["hidden_size"])
        n_layers = int(cfg["num_hidden_layers"])
        ffn = int(cfg["ffn_dim"])
        vocab = int(cfg["vocab_size"])
        max_pos = int(cfg["max_position_embeddings"])
        self._init_std = float(cfg["init_std"])

        shapes: dict[str, tuple[int, ...]] = {}
        shapes["model.decoder.embed_tokens.weight"] = (vocab, hidden)
        # HF's OPTLearnedPositionalEmbedding carries a fixed offset of 2
        # rows, so real checkpoints store max_position_embeddings + 2.
        shapes["model.decoder.embed_positions.weight"] = (max_pos + 2, hidden)
        shapes["model.decoder.final_layer_norm.weight"] = (hidden,)
        shapes["model.decoder.final_layer_norm.bias"] = (hidden,)
        # OPT ties lm_head to embed_tokens, and real checkpoints differ on
        # whether they also ship the tied copy. This one ships it; `get()`
        # seeds it from embed_tokens so the two stay byte-identical.
        shapes["lm_head.weight"] = (vocab, hidden)
        for i in range(n_layers):
            base = f"model.decoder.layers.{i}"
            shapes[f"{base}.self_attn_layer_norm.weight"] = (hidden,)
            shapes[f"{base}.self_attn_layer_norm.bias"] = (hidden,)
            for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
                shapes[f"{base}.self_attn.{p}.weight"] = (hidden, hidden)
                shapes[f"{base}.self_attn.{p}.bias"] = (hidden,)
            shapes[f"{base}.final_layer_norm.weight"] = (hidden,)
            shapes[f"{base}.final_layer_norm.bias"] = (hidden,)
            shapes[f"{base}.fc1.weight"] = (ffn, hidden)
            shapes[f"{base}.fc1.bias"] = (ffn,)
            shapes[f"{base}.fc2.weight"] = (hidden, ffn)
            shapes[f"{base}.fc2.bias"] = (hidden,)
        self._shapes = shapes
        self.fmt = "dummy"
        # Same value shape as _TensorReader.weight_map, so stage()'s
        # shard-count print works on either reader.
        self.weight_map = {k: (cfg_path, k) for k in shapes}

    def has(self, key: str) -> bool:
        return key in self._shapes

    def get(self, key: str) -> torch.Tensor:
        import zlib
        shape = self._shapes[key]
        if key.endswith("layer_norm.weight"):
            return torch.ones(shape, dtype=torch.float16)
        if key.endswith(".bias"):
            return torch.zeros(shape, dtype=torch.float16)
        # Real checkpoints that ship lm_head.weight carry a byte-identical
        # tied copy of embed_tokens, so seed it from the embed_tokens key.
        seed_key = ("model.decoder.embed_tokens.weight"
                    if key == "lm_head.weight" else key)
        g = torch.Generator().manual_seed(zlib.crc32(seed_key.encode()))
        return (torch.randn(shape, generator=g, dtype=torch.float32)
                .mul_(self._init_std).to(torch.float16))

    def close(self) -> None:
        pass

    def __enter__(self) -> "_DummyReader":
        return self

    def __exit__(self, *a) -> None:
        self.close()


def _write_aligned(fh, tensor: torch.Tensor) -> tuple[int, int]:
    """Write a tensor as contiguous float16 to `fh` at the current position,
    then pad to the next `_ALIGNMENT` boundary. Returns (offset, nbytes).

    The padding is what makes every parameter's offset O_DIRECT-addressable,
    so the loader can read any one of them without a bounce buffer. `nbytes`
    excludes the padding, so the manifest describes the real tensor."""
    t = tensor
    if t.dtype != _DTYPE:
        t = t.to(_DTYPE)
    t = t.detach().contiguous().cpu()
    offset = fh.tell()
    fh.write(t.numpy().tobytes())
    nbytes = fh.tell() - offset
    pad = _align_up(fh.tell(), _ALIGNMENT) - fh.tell()
    if pad:
        fh.write(b"\0" * pad)
    return offset, nbytes


def stage(model_id: str, output_dir: Path, verify: bool,
          dummy: bool = False) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    blob_path = output_dir / "model.bin"
    manifest_path = output_dir / "manifest.json"

    print(f"[stage] resolving {model_id} (dtype={_DTYPE_NAME}"
          f"{', DUMMY' if dummy else ''}) …", flush=True)
    reader = (_DummyReader(Path(model_id)) if dummy
              else _TensorReader(model_id))
    cfg = json.loads((reader.root / "config.json").read_text())
    model_type = str(cfg["model_type"]).lower()
    # BLOOM configs name the width and depth n_embed / n_layer.
    hidden = int(cfg["hidden_size"] if "hidden_size" in cfg
                 else cfg["n_embed"])
    num_heads = int(cfg["num_attention_heads"])
    num_layers = int(cfg["num_hidden_layers"] if "num_hidden_layers" in cfg
                     else cfg["n_layer"])
    # The runtime binds pre-LayerNorm OPT only; a post-LayerNorm config (the
    # opt-350m family, which also carries project_in/project_out) is refused.
    if model_type == "opt" and not cfg["do_layer_norm_before"]:
        raise ValueError(
            f"{model_id}: post-LayerNorm OPT (do_layer_norm_before=false) "
            "is not supported")

    # Architecture dispatch: which extras and layer parameters to pack, and
    # how to resolve each one to its (possibly fused) source tensor. Both
    # architectures stage the vLLM layout the streaming runtime binds. The
    # param list defines the on-disk order and `layer_get` produces the bytes.
    if model_type == "bloom":
        extras_order = BLOOM_EXTRAS_ORDER
        layer_params = BLOOM_LAYER_PARAMS_VLLM
        def layer_get(reader, layer_idx, suffix):
            return _build_bloom_layer_tensor(reader, layer_idx, suffix,
                                             num_heads, hidden)
    else:
        extras_order = OPT_EXTRAS_ORDER
        layer_params = OPT_LAYER_PARAMS_VLLM
        layer_get = _build_vllm_layer_tensor

    print(f"[stage] arch={model_type}  format={reader.fmt}  "
          f"shards={len({p for p, _ in reader.weight_map.values()})}  "
          f"keys={len(reader.weight_map)}  layers={num_layers}", flush=True)

    extras_meta: list[dict] = []
    layers_meta: list[dict] = []

    with reader, blob_path.open("wb") as blob:
        # ---- extras ----
        for key in extras_order:
            if key == "lm_head.weight" and not reader.has(key):
                continue        # tied to embed_tokens and not shipped
            t = reader.get(key)
            offset, nbytes = _write_aligned(blob, t)
            extras_meta.append({
                "name": key,
                "offset": offset,
                "nbytes": nbytes,
                "shape": list(t.shape),
                "dtype": _DTYPE_NAME,
            })
            del t
        extras_end = blob.tell()
        print(f"[stage] extras: {len(extras_meta)} tensors, "
              f"{extras_end / 2**20:.2f} MiB", flush=True)

        # ---- per-layer ----
        layer_bytes_first = None
        for layer_idx in range(num_layers):
            params: list[dict] = []
            layer_start = blob.tell()
            for suffix in layer_params:
                t = layer_get(reader, layer_idx, suffix)
                offset, nbytes = _write_aligned(blob, t)
                params.append({
                    "name": suffix,
                    "offset": offset,
                    "nbytes": nbytes,
                    "shape": list(t.shape),
                    "dtype": _DTYPE_NAME,
                })
                del t
            layers_meta.append({"layer": layer_idx, "params": params})
            if layer_idx == 0:
                layer_bytes_first = blob.tell() - layer_start
                print(f"[stage] layer 0: {len(params)} params, "
                      f"{layer_bytes_first / 2**20:.2f} MiB "
                      "(every layer will be similar size)", flush=True)
            elif (layer_idx + 1) % 16 == 0 or layer_idx == num_layers - 1:
                print(f"[stage] layer {layer_idx + 1}/{num_layers} "
                      f"({blob.tell() / 2**30:.2f} GiB written)",
                      flush=True)
        total_bytes = blob.tell()

    # The `config` block records the model's shape; run_bench sizes the KV
    # pool from its hidden_size and num_attention_heads. The `extras`/`layers`
    # offset tables are what the runtime binds.
    config_meta = {
        "model_type":          model_type,
        "hidden_size":         hidden,
        "num_hidden_layers":   num_layers,
        "num_attention_heads": num_heads,
        "vocab_size":          int(cfg["vocab_size"]),
        # Absent from a config when it holds HF's default, True.
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", True)),
    }
    if model_type == "bloom":
        config_meta["ffn_dim"] = 4 * hidden          # BLOOM MLP is 4·hidden
        config_meta["layer_norm_epsilon"] = float(cfg["layer_norm_epsilon"])
    else:
        config_meta.update({
            "ffn_dim":                 int(cfg["ffn_dim"]),
            "max_position_embeddings": int(cfg["max_position_embeddings"]),
            "activation_function":     str(cfg["activation_function"]),
        })

    manifest = {
        "model": model_id,
        "config": config_meta,
        "alignment":   _ALIGNMENT,
        "dtype":       _DTYPE_NAME,
        "extras":      extras_meta,
        "layers":      layers_meta,
        "total_bytes": total_bytes,
        # Synthesized weights: a throughput-only artifact.
        **({"dummy": True} if dummy else {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[stage] wrote {blob_path} ({total_bytes / 2**30:.2f} GiB)")
    print(f"[stage] wrote {manifest_path}")

    if verify:
        _verify(blob_path, manifest, model_id, dummy=dummy)

    return manifest


def _verify(blob_path: Path, manifest: dict, model_id: str,
            dummy: bool = False) -> None:
    """Re-read every staged tensor from the blob and byte-compare it to the
    source checkpoint.

    Fused parameters are rebuilt here by the same helpers the writer used, so
    the comparison checks that the bytes landed at the recorded offsets, not
    that the fusion logic agrees with itself. A mismatch means the blob and
    the manifest disagree, which the runtime cannot detect at load time."""
    print(f"[verify] re-reading {blob_path} and comparing to source …",
          flush=True)
    n_checked = 0
    mcfg = manifest["config"]
    is_bloom = mcfg["model_type"] == "bloom"
    num_heads = int(mcfg["num_attention_heads"])
    hidden = int(mcfg["hidden_size"])
    with (_DummyReader(Path(model_id)) if dummy
          else _TensorReader(model_id)) as reader, \
         blob_path.open("rb") as fh:

        def _src_tensor(full_key: str, layer_idx: Optional[int] = None,
                         suffix: Optional[str] = None) -> torch.Tensor:
            if suffix is not None:
                if is_bloom:
                    return _build_bloom_layer_tensor(reader, layer_idx, suffix,
                                                     num_heads, hidden)
                return _build_vllm_layer_tensor(reader, layer_idx, suffix)
            return reader.get(full_key)

        def _check(name: str, meta: dict, layer_idx: Optional[int] = None,
                    suffix: Optional[str] = None) -> None:
            nonlocal n_checked
            fh.seek(meta["offset"])
            raw = fh.read(meta["nbytes"])
            src = _src_tensor(name, layer_idx=layer_idx, suffix=suffix)
            if src.dtype != _DTYPE:
                src = src.to(_DTYPE)
            src = src.detach().contiguous().cpu()
            if raw != src.numpy().tobytes():
                raise AssertionError(
                    f"byte mismatch for {name!r} "
                    f"(offset={meta['offset']}, nbytes={meta['nbytes']})")
            n_checked += 1

        layer_base = ("transformer.h" if is_bloom
                      else "model.decoder.layers")
        for em in manifest["extras"]:
            _check(em["name"], em)
        for li in manifest["layers"]:
            for pm in li["params"]:
                full_key = f"{layer_base}.{li['layer']}.{pm['name']}"
                _check(full_key, pm,
                        layer_idx=li["layer"], suffix=pm["name"])
    print(f"[verify] OK — {n_checked} tensors byte-identical")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. facebook/opt-125m (with --dummy, "
                         "a local directory holding config.json)")
    ap.add_argument("--output", type=Path, required=True,
                    help="output directory, empty or absent (will contain "
                         "model.bin + manifest.json)")
    ap.add_argument("--verify", action="store_true",
                    help="re-read every staged tensor and assert byte-equality")
    ap.add_argument("--dummy", action="store_true",
                    help="synthesize the checkpoint instead of downloading "
                         "it: --model must be a LOCAL directory holding "
                         "config.json (no hub access). Tensors are "
                         "deterministic source-init-scale values; the blob "
                         "goes through the identical write path, so it is a "
                         "faithful throughput stand-in with meaningless "
                         "outputs. For models whose weights cannot be used "
                         "(e.g. opt-175b).")
    args = ap.parse_args(argv)
    stage(args.model, args.output, args.verify, dummy=args.dummy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
