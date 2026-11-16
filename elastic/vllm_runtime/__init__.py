"""vLLM integration runtime for elastic.

This package wires elastic's NVMe streaming loader, the per-layer α-driven
KV/ACT pipeline and arrival-based decode attention into a stock vLLM build.
`elastic/runtime/` holds the vLLM-agnostic core it builds on (block forms,
block storage split into block regions, weight table).

Importing it registers the load_format "elastic_streaming" with vLLM
(`streaming_loader.StreamingModelLoader`). That loader meta-initialises the
model so a large model's full weight set never has to land on device, then
hands per-layer weight materialisation to the loader thread that reads from a
`stage_opt` blob via io_uring.

Runtime defaults are pinned here at import time, before the vLLM engine is
built, so every entry point runs the same configuration. An explicitly set
environment variable always wins over a default below.
"""

from __future__ import annotations

import os as _os

# kv_gen launch size (tokens): one kv_gen launch covers this many tokens of
# loaded activations, cut apart from the 16 MiB read segments. A launch that
# is not a multiple of 1024 tokens costs more per token (the kv_gen kernel's
# tile group covers 1024 rows). The value lives in `elastic.cost.markers`; the
# units check rejects a record measured at a different launch size.
from elastic.cost.markers import LOAD_CHUNK_TOKENS_DEFAULT as _KV_GEN_TOKENS
_os.environ.setdefault("ELASTIC_LOAD_CHUNK_TOKENS", str(_KV_GEN_TOKENS))

# KV cache layout: NHD (token-major). The on-disk K/V block byte order is
# this cache block order (runtime/block_kv_storage), so a save or reload is
# one zero-copy whole-block DMA.
_os.environ.setdefault("VLLM_KV_CACHE_LAYOUT", "NHD")
if _os.environ["VLLM_KV_CACHE_LAYOUT"] != "NHD":
    raise RuntimeError(
        "elastic requires VLLM_KV_CACHE_LAYOUT=NHD (the on-disk block byte "
        "order is the NHD cache block order), got "
        f"{_os.environ['VLLM_KV_CACHE_LAYOUT']!r}")

# Weight read segment size in the unified read stream. A layer's weight is
# split into segments of this size so every read segment has the same size
# as the KV/activation reads; one whole-layer segment would put thousands of
# block-layer requests in flight at once and fall off this drive's tag-pool
# cliff.
_os.environ.setdefault("ELASTIC_UNIFIED_WEIGHT_MB", "16")

# Read segment size and read queue depth. Their product is the bytes in
# flight (8 × 16 MiB = 128 MiB = one whole tag pool of 128 KiB requests),
# which is the quantity the drive responds to.
_os.environ.setdefault("ELASTIC_READ_SPAN_MB", "16")
_os.environ.setdefault("ELASTIC_READ_QD", "8")


# Write unit (NVMe engine side). The NVMe engine composes each write SQE
# from up to this many file-contiguous admitted bytes, joining abutting
# requests and cutting longer ones at unit boundaries. The same figure bounds
# the bytes in flight: a new unit is composed only while less than one is in
# flight, so the device holds one to two units. The bound gates the next fill
# and never rejects an item, so it cannot deadlock and does not change the
# request order. How deep a write lands sets the blob's read-back speed on
# this drive, so the unit is 32 MiB.
_os.environ.setdefault("ELASTIC_WRITE_INFLIGHT_MB", "32")

# Zig-zag prefill GPU batch size (tokens). A batch's prefill is one forward,
# the vLLM engine's first step for the batch; whole-sequence GPU batches of at
# most this many tokens loop through each layer while the slot holding its
# weights is held once, which bounds the transient activation.
_os.environ.setdefault("ELASTIC_PREFILL_MB_TOKENS", "16384")

# Partial attention call volume (tokens) of arrival-based attention: pending
# reloaded KV accumulates to this many tokens before one partial attention
# call is issued; the remainder is flushed when the layer's feed closes.
from elastic.cost.markers import ARRIVAL_DRAIN_TOKENS_DEFAULT as _PARTIAL_ATTN
_os.environ.setdefault("ELASTIC_ARRIVAL_DRAIN_TOKENS", str(_PARTIAL_ATTN))

# The vLLM engine runs in this process (vLLM's in-process client):
# `LLM.generate` adds every request before the vLLM engine takes its first
# step, so that step holds the whole batch and is its prefill.
_os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Side-effect: registers "elastic_streaming" load_format with vLLM.
from . import streaming_loader as _streaming_loader   # noqa: F401
