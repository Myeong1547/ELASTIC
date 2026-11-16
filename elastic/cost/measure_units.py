"""Micro-benchmarks for the unit costs of the α-linear cost model.

    T_NVMe(α)    = t_w + N_seq·(t_kv + t_save_kv)·(1 − α/2)
    T_comp(α)    = N_seq·T_fwd + α·N_seq·T_kv_gen

Every micro-benchmark reproduces the runtime's conditions for the primitive
it measures — access shape, queue depth, buffer residency and kernel. A
primitive measured under different conditions is a different number and
mis-aims the solver, so each measurement below states the condition it
matches.

Unit costs (one record per model, prompt S_op and gen_len; no batch):

    read_GBps              NVMe read rate: the staged weight blob read in
                           ELASTIC_UNIFIED_WEIGHT_MB segments with
                           ELASTIC_READ_QD segments in flight, through
                           the same NVMe engine the loader uses
    write_GBps             NVMe write rate, priced at the post-cache-
                           exhaustion floor of the drive (a drive constant)
    T_kv_gen_per_token_s   kv_gen (the projection with its store fused in,
                           run once for K and once for V) of one token's
                           activation into K and V, in the runtime's kv_gen
                           launches, over ELASTIC_ARRIVAL_DRAIN_TOKENS
                           tokens, all operands host-registered
    T_attn_per_token_s     one LSE partial attention call of
                           ELASTIC_ARRIVAL_DRAIN_TOKENS tokens, run through
                           `arrival_decode` as is, per token
    T_dense_per_token_s    LN, fused Q/K/V projection, output projection, LN
                           and FFN over the same ELASTIC_ARRIVAL_DRAIN_TOKENS
                           tokens, weights host-registered, per token

The solver prices one sequence's forward per layer and step as
T_dense_per_token + (S+gen)·T_attn_per_token (`policy.from_units`).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from elastic.loader.nvme_loader import NvmeEngine

from .markers import (KVGEN_STORE_SHAPE, LAYER_STEP_SHAPE, run_partial_attn_tokens,
                      run_kv_gen_tokens)
from .config import LayerConfig
from .decoder_layer import DecoderLayer

_BLOCK = 16                 # runtime KV block size (tokens)
_ALIGN = 4096               # O_DIRECT alignment
_REPEATS = 5                # timed repeats per measurement (median taken)
_REPEATS_FORWARD = 8        # timed repeats of the attention and dense micros
_WARMUP = 2                 # untimed runs before them

# A read rate above this is a broken measurement (page-cache read, wrong
# pattern), never a fast drive: it is above the drive's rated sequential read.
_READ_CEILING_BPS = 16e9


@dataclass
class UnitTimes:
    read_GBps: float
    write_GBps: float
    T_kv_gen_per_token_s: float
    T_attn_per_token_s: float
    T_dense_per_token_s: float
    S_op: int
    gen_len: int
    weight_bytes: int
    provenance: dict


def _align_up(x: int, a: int = _ALIGN) -> int:
    return ((x + a - 1) // a) * a


def weight_bytes_per_layer(cfg: LayerConfig) -> int:
    """Per-layer weight bytes as the staged blob stores them (4 KiB aligned)."""
    return _align_up(int(cfg.weight_bytes))


def _median_event_ms(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def _median_wall_s(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ------------------------------------------------------------------ read


def measure_read_GBps(weight_blob: str, span: int, repeats: int,
                      warmup: int, n_offsets: int = 4) -> tuple[float, list]:
    """Read `span` bytes (one layer's weight) at `n_offsets` well-separated
    offsets of the staged weight blob, in the loader's shape: segments of
    ELASTIC_UNIFIED_WEIGHT_MB enqueued up front on an nvme_engine read port
    that keeps ELASTIC_READ_QD segments in flight one-out-one-in. NVMe
    read rate varies by flash region, so the median over offsets is taken.
    Returns (bytes/s, offsets)."""
    piece = max(1, int(os.environ.get("ELASTIC_UNIFIED_WEIGHT_MB", "16")
                       or 16)) << 20
    qd = max(2, int(os.environ.get("ELASTIC_READ_QD", "8") or 8))
    if not os.path.exists(weight_blob):
        raise RuntimeError(f"weight blob {weight_blob} does not exist")
    with NvmeEngine(weight_blob, read_ring=256, write_ring=0,
                    read_qd=qd) as eng:
        port = eng.read_port
        if port.size < span:
            raise RuntimeError(
                f"weight blob {weight_blob} smaller than one layer "
                f"({port.size} < {span})")
        dst = port.alloc_aligned(span)
        step = (port.size - span) // max(1, n_offsets - 1)
        offsets = [(i * step) & ~(_ALIGN - 1) for i in range(n_offsets)]
        pieces_of = lambda base: [(base + o, min(piece, span - o))
                                  for o in range(0, span, piece)]

        def read_once(base: int) -> float:
            pieces = pieces_of(base)
            t0 = time.perf_counter()
            for i, (po, pl) in enumerate(pieces):
                rel = po - base
                port.submit_read(po, pl,
                                 dst[rel:rel + pl], user_data=i)
            for _ in pieces:
                _ud, res = port.wait_one()
                if res < 0:
                    raise OSError(-res, os.strerror(-res))
            return time.perf_counter() - t0

        per_off = [_median_wall_s(lambda b=b: read_once(b), repeats, warmup)
                   for b in offsets]
    t = float(np.median(per_off))
    return span / t, offsets


# --------------------------------------------------------------- forward


def _dense_weights_uma(layer: DecoderLayer, keep: list):
    """The layer's weights where the runtime keeps them: in host-registered
    memory (the slot's weight buffer), Q/K/V fused into one projection as
    vLLM's QKVParallelLinear stores them."""
    from elastic.vllm_runtime.streaming_loader import _alloc_uma_buffer

    def uma(t):
        u = _alloc_uma_buffer(t.numel() * t.element_size(), keep) \
            .view(t.dtype).view_as(t)
        u.copy_(t)
        return u

    d = {n: p.detach() for n, p in layer.named_parameters()}
    W = {n: uma(d[n]) for n in ("ln1.weight", "ln1.bias", "out_proj.weight",
                                "out_proj.bias", "ln2.weight", "ln2.bias",
                                "fc1.weight", "fc1.bias", "fc2.weight",
                                "fc2.bias")}
    W["qkv.weight"] = uma(torch.cat([d["q_proj.weight"], d["k_proj.weight"],
                                     d["v_proj.weight"]], 0))
    W["qkv.bias"] = uma(torch.cat([d["q_proj.bias"], d["k_proj.bias"],
                                   d["v_proj.bias"]], 0))
    return W


class _IndexScratch:
    """`ActBufferPool.arrival_index` over one registered buffer: bumped
    within a visit, rewound at the visit start."""

    def __init__(self, nbytes: int, keep: list):
        from elastic.vllm_runtime.streaming_loader import _alloc_uma_buffer
        t = _alloc_uma_buffer(int(nbytes), keep)
        self._np = keep[-1][0]
        self._t = t.view(torch.uint8)
        self._cap = int(nbytes)
        self._used = 0

    def arrival_index_reset(self, layer_idx):
        self._used = 0

    def arrival_index(self, layer_idx, nbytes):
        n = (int(nbytes) + 255) & ~255
        end = self._used + n
        if end > self._cap:
            raise RuntimeError(f"index scratch {self._cap} B exceeded")
        self._used = end
        return self._np[end - n:end], self._t[end - n:end]


def _measure_attn_per_token(cfg: LayerConfig, S_op: int, gen_len: int,
                            repeats: int, warmup: int):
    """One LSE partial attention call of ELASTIC_ARRIVAL_DRAIN_TOKENS tokens,
    run through `arrival_decode` as is, per token.

    The call holds the rows one partial attention call holds at the last
    decode step's context (S_op + gen_len − 1 keys, the new token
    included): ELASTIC_ARRIVAL_DRAIN_TOKENS // context rows. Their K/V
    blocks are pushed as the loader pushes them (items of one K/V segment
    pair, ELASTIC_READ_SPAN_MB each) and the feed is closed, so the consumer
    attends to them in one partial attention call; the KV cache is
    host-registered in the NHD layout, and the new token's K/V is written
    through `append_cb` with `reshape_and_cache_flash`. Each visit is timed
    on the device, median of `repeats`.
    Returns (seconds/token, rows, context, calls per visit)."""
    from elastic.runtime.block_kv_storage import BlockKVLayout
    from elastic.vllm_runtime import arrival_attention as aa
    from elastic.vllm_runtime.layer_arrival import ArrivalItem, LayerArrival
    from elastic.vllm_runtime.streaming_loader import _alloc_uma_buffer
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

    dev = torch.device("cuda", torch.cuda.current_device())
    nH, D = cfg.num_attention_heads, cfg.head_dim
    drain = run_partial_attn_tokens()
    ctx_len = S_op + gen_len - 1
    rows = max(1, drain // ctx_len)
    lay = BlockKVLayout(n_seq=rows, num_layers=1,
                        max_blocks_per_seq=-(-(ctx_len + 1) // _BLOCK),
                        block_size=_BLOCK, num_heads=nH, head_dim=D,
                        dtype=cfg.dtype, alpha=[0.0])
    lay.set_prompt_len(S_op)
    K = rows * lay.block_split(ctx_len)[1]
    kb = lay.block_bytes
    span = max(1, int(os.environ["ELASTIC_READ_SPAN_MB"])) << 20
    pair = max(1, span // kb)
    items = [ArrivalItem(1, r0, min(pair, K - r0), None)
             for r0 in range(0, K, pair)]
    keep: list = []
    nblk = int(lay.mem_blocks_needed)
    buf = _alloc_uma_buffer(2 * nblk * _BLOCK * nH * D * 2, keep).view(cfg.dtype)
    kv = buf.view(2, nblk, _BLOCK, nH, D)
    key_cache, value_cache = kv.unbind(0)
    key_cache.normal_(); value_cache.normal_()
    q = torch.randn(rows, nH, D, dtype=cfg.dtype, device=dev)
    out = torch.empty(rows, nH, D, dtype=cfg.dtype, device=dev)
    k_new = torch.randn(rows, nH, D, dtype=cfg.dtype, device=dev)
    v_new = torch.randn(rows, nH, D, dtype=cfg.dtype, device=dev)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    ranks = lay.ks_of_tokens(np.arange(rows, dtype=np.int64),
                                np.full(rows, ctx_len - 1, dtype=np.int64))
    slot_map = torch.from_numpy(
        (lay.mem_blocks_np(0, ranks) * _BLOCK
         + (ctx_len - 1) % _BLOCK).astype(np.int64)).to(dev)

    def append(r):
        reshape_and_cache_flash(k_new.index_select(0, r),
                                v_new.index_select(0, r), key_cache,
                                value_cache, slot_map.index_select(0, r),
                                "auto", one, one)

    stub = _IndexScratch(16 << 20, keep)
    row_sids = list(range(rows))
    calls = []

    def visit():
        arr = LayerArrival(1)
        arr.begin(0)
        for it in items:
            arr.push(0, it)
        arr.close(0)
        c = aa.ArrivalCtx(arr, 0, row_sids, lay, [ctx_len] * rows,
                          {s_: s_ for s_ in row_sids}, _BLOCK, act_pool=stub)
        f0 = aa.STATS["flash_calls"]
        aa.arrival_decode(q, out, key_cache, value_cache, scale=D ** -0.5,
                          ctx=c, append_cb=append)
        calls.append(aa.STATS["flash_calls"] - f0)

    try:
        ms = _median_event_ms(visit, repeats=repeats, warmup=warmup)
    finally:
        del kv, key_cache, value_cache, buf
        keep.clear()
        torch.cuda.empty_cache()
    return (ms / 1e3) / (rows * ctx_len), rows, ctx_len, sorted(set(calls))


def _measure_dense_per_token(layer: DecoderLayer, cfg: LayerConfig,
                             tokens: int, repeats: int, warmup: int) -> float:
    """LN, fused Q/K/V projection, output projection, LN and FFN over
    `tokens` rows, every weight in host-registered memory as at runtime,
    median of `repeats`, per token."""
    dev = next(layer.parameters()).device
    H = cfg.hidden_size
    keep: list = []
    W = _dense_weights_uma(layer, keep)
    x = torch.randn(tokens, H, dtype=cfg.dtype, device=dev)
    attn_in = torch.randn(tokens, H, dtype=cfg.dtype, device=dev)
    act = (torch.nn.functional.relu if cfg.activation == "relu"
           else torch.nn.functional.gelu)
    F = torch.nn.functional

    def dense():
        with torch.no_grad():
            h = F.layer_norm(x, (H,), W["ln1.weight"], W["ln1.bias"])
            F.linear(h, W["qkv.weight"], W["qkv.bias"])
            y = F.linear(attn_in, W["out_proj.weight"], W["out_proj.bias"]) + x
            h = F.layer_norm(y, (H,), W["ln2.weight"], W["ln2.bias"])
            h = F.linear(act(F.linear(h, W["fc1.weight"], W["fc1.bias"])),
                         W["fc2.weight"], W["fc2.bias"])
            return y + h

    try:
        ms = _median_event_ms(dense, repeats=repeats, warmup=warmup)
    finally:
        del x, attn_in
        keep.clear()
        torch.cuda.empty_cache()
    return (ms / 1e3) / tokens


# ---------------------------------------------------------------- kv_gen


def _measure_kv_gen(layer: DecoderLayer, cfg: LayerConfig, tokens: int,
                    repeats: int, warmup: int) -> tuple[float, list, int]:
    """Per-token cost of kv_gen (the projection with its store fused in) over
    `tokens` tokens of activations.

    The kv_gen launches are the runtime's: `act_segments_and_launches` cuts
    the activation block region into launches of ELASTIC_LOAD_CHUNK_TOKENS
    tokens apart from the read segments, as the loader does, and each launch
    runs kv_gen twice, K then V, on a priority-0 stream (the kv_gen
    stream's), its hidden a dense slice of a host-registered activation
    buffer, its destination a consecutive run of host-registered cache
    blocks, the K/V weights and biases host-registered (the slot's weight
    buffer). The whole visit's launches are run `repeats` times and the
    median taken. Returns (seconds/token, launch sizes in blocks, launch
    tokens)."""
    from elastic.kernels.kv_gen import kv_gen
    from elastic.vllm_runtime.loader_thread import act_segments_and_launches
    from elastic.vllm_runtime.streaming_loader import _alloc_uma_buffer

    elem = torch.empty(0, dtype=cfg.dtype).element_size()
    keep: list = []

    def to_uma(t):
        u = _alloc_uma_buffer(t.numel() * elem, keep).view(cfg.dtype).view_as(t)
        u.copy_(t)
        return u

    k_w = to_uma(layer.k_proj.weight.detach())
    v_w = to_uma(layer.v_proj.weight.detach())
    k_b = to_uma(layer.k_proj.bias.detach())
    v_b = to_uma(layer.v_proj.bias.detach())
    kv_dim = k_w.shape[0]
    hd = cfg.head_dim
    nkvh = kv_dim // hd
    n_act = -(-int(tokens) // _BLOCK)
    kb = _BLOCK * cfg.hidden_size * elem
    launch_tokens = run_kv_gen_tokens()
    span = max(1, int(os.environ["ELASTIC_READ_SPAN_MB"])) << 20
    _d, launches = act_segments_and_launches(n_act, 0, kb, span // kb,
                               max(1, launch_tokens // _BLOCK))
    stage = _alloc_uma_buffer(n_act * kb, keep).view(cfg.dtype) \
        .view(n_act * _BLOCK, cfg.hidden_size)
    stage.normal_()

    def mk_cache():
        buf = _alloc_uma_buffer((1 + n_act) * _BLOCK * kv_dim * elem, keep) \
            .view(cfg.dtype)
        return buf.view(1 + n_act, _BLOCK, nkvh, hd)

    cache_k, cache_v = mk_cache(), mk_cache()
    stream = torch.cuda.Stream(priority=0)

    def kvgen():
        with torch.cuda.stream(stream):
            for w0, n, _ids in launches:
                h_in = stage[w0 * _BLOCK:(w0 + n) * _BLOCK]
                kv_gen(h_in, k_w, k_b, cache_k, 1 + w0, _BLOCK)
                kv_gen(h_in, v_w, v_b, cache_v, 1 + w0, _BLOCK)
        torch.cuda.current_stream().wait_stream(stream)

    try:
        ms = _median_event_ms(kvgen, repeats=repeats, warmup=warmup)
    finally:
        del stage, cache_k, cache_v
        keep.clear()
        torch.cuda.empty_cache()
    return (ms / 1000.0) / (n_act * _BLOCK), [int(w[1]) for w in launches], \
        launch_tokens


# ------------------------------------------------------------------ main


def measure_unit_times(cfg: LayerConfig, weight_blob: str, layer: DecoderLayer,
                       S_op: int, gen_len: int) -> UnitTimes:
    """Measure the unit costs for one model at prompt S_op and gen_len."""
    repeats, warmup = _REPEATS, _WARMUP
    wbytes = weight_bytes_per_layer(cfg)
    read_GBps, offsets = measure_read_GBps(weight_blob, wbytes, repeats, warmup)
    print(f"[measure.read] {weight_blob}: {read_GBps/1e9:.2f} GB/s "
          f"({len(offsets)} offsets × median-of-{repeats})", flush=True)

    # Write rate: the drive's post-write-cache-exhaustion floor. A run writes
    # far more than the write cache holds and recovery needs an idle device,
    # so the exhausted regime is the one every decode step runs in. A pwrite
    # micro-benchmark measures the in-cache regime and is not a price. A drive
    # constant, entered as observed; re-derive it on another drive with an
    # isolated pure-write ramp past the cache.
    write_GBps = float(os.environ.get("ELASTIC_UNITS_WRITE_GBPS", "1.82") or 1.82) * 1e9
    if not 0.05e9 <= write_GBps <= 20e9:
        raise ValueError(f"ELASTIC_UNITS_WRITE_GBPS={write_GBps/1e9} out of range")

    drain = run_partial_attn_tokens()
    T_attn, rows, ctx_len, n_calls = _measure_attn_per_token(
        cfg, S_op, gen_len, _REPEATS_FORWARD, warmup)
    T_dense = _measure_dense_per_token(layer, cfg, drain, _REPEATS_FORWARD,
                                       warmup)
    T_kv_gen, launches, kv_gen_tokens = _measure_kv_gen(
        layer, cfg, drain, repeats, warmup)
    print(f"[measure.forward] LSE call {rows} rows × {ctx_len} keys "
          f"({n_calls} call/visit): {T_attn*1e9:.1f} ns/token · dense "
          f"{T_dense*1e6:.3f} µs/token over {drain} tokens · kv_gen "
          f"{T_kv_gen*1e6:.3f} µs/token", flush=True)

    # Physical plausibility: the measurement is rejected, never adjusted.
    if read_GBps > _READ_CEILING_BPS:
        raise RuntimeError(
            f"read rate {read_GBps/1e9:.1f} GB/s exceeds the drive ceiling "
            f"{_READ_CEILING_BPS/1e9:.1f} GB/s — page-cache contamination or "
            f"broken pattern")

    return UnitTimes(
        read_GBps=read_GBps / 1e9,
        write_GBps=write_GBps / 1e9,
        T_kv_gen_per_token_s=T_kv_gen,
        T_attn_per_token_s=T_attn,
        T_dense_per_token_s=T_dense,
        S_op=int(S_op),
        gen_len=int(gen_len),
        weight_bytes=int(wbytes),
        provenance=dict(weight_blob=weight_blob, read_offsets=offsets,
                        attn_rows=rows, attn_context=ctx_len,
                        attn_calls_per_visit=n_calls,
                        drain_tokens=drain, repeats=repeats,
                        layer_step_shape=LAYER_STEP_SHAPE,
                        kvgen_waves=launches,
                        kvgen_wave_tokens=int(kv_gen_tokens),
                        kvgen_store_shape=KVGEN_STORE_SHAPE),
    )
