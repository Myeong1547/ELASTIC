"""Per-model layer shape constants.

The cost-model measurements depend only on shape and dtype — never on the
actual checkpoint weights. Adding a new homogeneous-decoder model means
appending one LayerConfig entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class LayerConfig:
    name: str
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    ffn_hidden_size: int
    activation: Literal["relu", "gelu"]
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        assert self.hidden_size == self.num_attention_heads * self.head_dim, (
            f"hidden_size {self.hidden_size} != heads {self.num_attention_heads} "
            f"* head_dim {self.head_dim}"
        )

    @property
    def dtype_bytes(self) -> int:
        return torch.empty([], dtype=self.dtype).element_size()

    @property
    def weight_bytes(self) -> int:
        # q, k, v, out projections (each H x H) + 2 FFN (H x F and F x H) +
        # 2 LayerNorms (each 2H scalars: gamma, beta) + linear biases.
        H, F = self.hidden_size, self.ffn_hidden_size
        attn = 4 * H * H            # 4 projection weights
        ffn = 2 * H * F             # fc1, fc2 weights
        ln = 4 * H                  # 2 LN * (gamma, beta)
        bias = 4 * H + 2 * H + F    # q/k/v/out + fc2 + fc1
        return (attn + ffn + ln + bias) * self.dtype_bytes


# OPT family (Meta).  Shapes per https://huggingface.co/facebook/opt-* configs.
OPT_125M = LayerConfig(
    name="opt-125m",
    hidden_size=768,   num_attention_heads=12,  head_dim=64,
    ffn_hidden_size=3072,  activation="relu",
)
OPT_1_3B = LayerConfig(
    name="opt-1.3b",
    hidden_size=2048,  num_attention_heads=32,  head_dim=64,
    ffn_hidden_size=8192,  activation="relu",
)
OPT_6_7B = LayerConfig(
    name="opt-6.7b",
    hidden_size=4096,  num_attention_heads=32,  head_dim=128,
    ffn_hidden_size=16384, activation="relu",
)
OPT_13B = LayerConfig(
    name="opt-13b",
    hidden_size=5120,  num_attention_heads=40,  head_dim=128,
    ffn_hidden_size=20480, activation="relu",
)
OPT_30B = LayerConfig(
    name="opt-30b",
    hidden_size=7168,  num_attention_heads=56,  head_dim=128,
    ffn_hidden_size=28672, activation="relu",
)
OPT_66B = LayerConfig(
    name="opt-66b",
    hidden_size=9216,  num_attention_heads=72,  head_dim=128,
    ffn_hidden_size=36864, activation="relu",
)
OPT_175B = LayerConfig(
    name="opt-175b",
    hidden_size=12288, num_attention_heads=96,  head_dim=128,
    ffn_hidden_size=49152, activation="relu",
)

# BLOOM (BigScience). 176B: 70 homogeneous decoder layers, GELU FFN, bias-ful
# like OPT (so the weight_bytes formula applies). bf16 upstream — element size
# is 2 B, identical to the fp16 default the byte accounting assumes. ALiBi
# positions don't change the per-layer matmul/KV shapes the cost model times.
BLOOM_176B = LayerConfig(
    name="bloom-176b",
    hidden_size=14336, num_attention_heads=112, head_dim=128,
    ffn_hidden_size=57344, activation="gelu",
)

ALL_MODELS = {
    m.name: m for m in [
        OPT_125M, OPT_1_3B, OPT_6_7B, OPT_13B, OPT_30B, OPT_66B, OPT_175B,
        BLOOM_176B,
    ]
}
