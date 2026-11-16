"""Parameters of one decoder layer, for the unit-cost measurement.

Random weights — this measures layer compute, not generation quality, so
checkpoint fidelity is out of scope. The module holds the pre-LN block's
submodules at the model's shapes; `measure_units` calls the submodules
directly, so the module has no forward of its own.
"""

from __future__ import annotations

import torch.nn as nn

from .config import LayerConfig


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LayerConfig):
        super().__init__()
        H, FFN = cfg.hidden_size, cfg.ffn_hidden_size

        self.ln1 = nn.LayerNorm(H, dtype=cfg.dtype)
        self.q_proj = nn.Linear(H, H, bias=True, dtype=cfg.dtype)
        self.k_proj = nn.Linear(H, H, bias=True, dtype=cfg.dtype)
        self.v_proj = nn.Linear(H, H, bias=True, dtype=cfg.dtype)
        self.out_proj = nn.Linear(H, H, bias=True, dtype=cfg.dtype)

        self.ln2 = nn.LayerNorm(H, dtype=cfg.dtype)
        self.fc1 = nn.Linear(H, FFN, bias=True, dtype=cfg.dtype)
        self.fc2 = nn.Linear(FFN, H, bias=True, dtype=cfg.dtype)


def make_random_layer(cfg: LayerConfig, device: str = "cuda") -> DecoderLayer:
    """Construct a layer on the target device with random fp16 weights."""
    layer = DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
    layer.eval()
    return layer
