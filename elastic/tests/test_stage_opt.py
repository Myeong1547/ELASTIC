"""Smoke test for `elastic.cli.stage_opt`.

Skips when `transformers` is unavailable or the checkpoint cannot be
fetched, so an offline environment reports a skip rather than a failure.
When it does run it stages a small model end to end and asserts the packed
blob is byte-identical to the source tensors, plus the two structural
properties the loader depends on: every layer carries its full parameter
set, and every recorded offset is alignment-aligned so an O_DIRECT read can
address it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_stage_opt_125m_byte_identical(tmp_path: Path) -> None:
    pytest.importorskip("transformers",
                        reason="transformers not installed")
    try:
        from elastic.cli.stage_opt import stage
    except ImportError as e:
        pytest.skip(f"stage_opt not importable: {e}")

    out = tmp_path / "opt-125m"
    try:
        manifest = stage(model_id="facebook/opt-125m", output_dir=out,
                         verify=True)
    except OSError as e:
        # No network or the model hub is unreachable — not a code failure.
        pytest.skip(f"could not download opt-125m: {e}")

    assert (out / "model.bin").exists()
    assert (out / "manifest.json").exists()
    assert manifest["config"]["num_hidden_layers"] == 12
    assert len(manifest["extras"]) >= 4
    assert len(manifest["layers"]) == 12
    for li in manifest["layers"]:
        # Every layer must carry the full parameter set — a layer silently
        # missing one would bind against the wrong bytes at run time.
        names = {p["name"] for p in li["params"]}
        assert "self_attn.qkv_proj.weight" in names
        assert "fc2.weight" in names

    # Every offset must be alignment-aligned, or O_DIRECT cannot read it.
    for em in manifest["extras"]:
        assert em["offset"] % manifest["alignment"] == 0
    for li in manifest["layers"]:
        for pm in li["params"]:
            assert pm["offset"] % manifest["alignment"] == 0


_MINI_OPT_CONFIG = {
    # Shape of a miniature OPT — enough to exercise every dummy-reader
    # branch (fused qkv, extras, per-layer params) without any download.
    "model_type": "opt",
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "ffn_dim": 256,
    "max_position_embeddings": 32,
    "vocab_size": 100,
    "word_embed_proj_dim": 64,
    "do_layer_norm_before": True,
    "activation_function": "relu",
    "init_std": 0.02,
}


def test_stage_opt_dummy_offline(tmp_path: Path) -> None:
    """--dummy stages a full blob from a local config.json with no network,
    through the identical write path; --verify proves the manifest offsets
    describe the bytes that landed (the dummy reader is deterministic, so
    the re-read rebuilds identical tensors)."""
    from elastic.cli.stage_opt import stage

    cfg_dir = tmp_path / "configs" / "mini-opt"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps(_MINI_OPT_CONFIG))
    out = tmp_path / "staged-mini"

    manifest = stage(model_id=str(cfg_dir), output_dir=out,
                     verify=True, dummy=True)

    assert manifest["dummy"] is True
    assert manifest["config"]["num_hidden_layers"] == 3
    # Extras must mirror a real large-OPT staging (66b): embeddings, final
    # LN, and the tied lm_head copy. embed_positions carries HF's +2 rows.
    extra_names = [e["name"] for e in manifest["extras"]]
    assert extra_names == [
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
        "model.decoder.final_layer_norm.weight",
        "model.decoder.final_layer_norm.bias",
        "lm_head.weight",
    ]
    pos = next(e for e in manifest["extras"]
               if e["name"].endswith("embed_positions.weight"))
    assert pos["shape"] == [32 + 2, 64]

    for li in manifest["layers"]:
        names = [p["name"] for p in li["params"]]
        # Full parameter set, in on-disk order, every layer.
        assert names == [
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "self_attn.qkv_proj.weight", "self_attn.qkv_proj.bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "final_layer_norm.weight", "final_layer_norm.bias",
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
        ]
        qkv = next(p for p in li["params"]
                   if p["name"] == "self_attn.qkv_proj.weight")
        assert qkv["shape"] == [3 * 64, 64]
        for pm in li["params"]:
            assert pm["offset"] % manifest["alignment"] == 0

    blob = out / "model.bin"
    assert blob.stat().st_size == manifest["total_bytes"]

    # The synthetic content must be numerically sane: LN gamma = 1, biases
    # = 0, matrices small — that is what keeps a dummy forward pass free of
    # NaN/Inf.
    import numpy as np
    raw = blob.read_bytes()

    def _tensor(meta):
        return np.frombuffer(
            raw[meta["offset"]:meta["offset"] + meta["nbytes"]],
            dtype=np.float16)

    l0 = {p["name"]: p for p in manifest["layers"][0]["params"]}
    assert (_tensor(l0["self_attn_layer_norm.weight"]) == 1.0).all()
    assert (_tensor(l0["self_attn.qkv_proj.bias"]) == 0.0).all()
    qkv_vals = _tensor(l0["self_attn.qkv_proj.weight"])
    assert np.isfinite(qkv_vals.astype(np.float32)).all()
    assert 0.001 < float(np.abs(qkv_vals.astype(np.float32)).std()) < 0.1
    # lm_head must be the tied byte-identical copy of embed_tokens, as in
    # every real OPT checkpoint that ships it.
    ex = {e["name"]: e for e in manifest["extras"]}
    assert (_tensor(ex["lm_head.weight"])
            == _tensor(ex["model.decoder.embed_tokens.weight"])).all()


def test_stage_opt_dummy_rejects_350m_shape(tmp_path: Path) -> None:
    """A post-LayerNorm OPT config (do_layer_norm_before=false, the opt-350m
    family) must fail loudly instead of staging a blob the runtime cannot
    bind."""
    from elastic.cli.stage_opt import stage

    cfg = dict(_MINI_OPT_CONFIG, do_layer_norm_before=False)
    cfg_dir = tmp_path / "cfg350"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        stage(model_id=str(cfg_dir), output_dir=tmp_path / "o",
              verify=False, dummy=True)


def test_stage_opt_dummy_rejects_remote_id(tmp_path: Path) -> None:
    """--dummy must never fall through to a hub download: a model id that
    is not a local config directory is a hard error."""
    from elastic.cli.stage_opt import stage

    with pytest.raises(FileNotFoundError):
        stage(model_id="facebook/opt-175b", output_dir=tmp_path / "o",
              verify=False, dummy=True)
