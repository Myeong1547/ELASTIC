"""End-to-end throughput measurement for a single (model, batch, prompt_len,
gen_len) cell, via vLLM plus the elastic streaming loader.

One process runs exactly one cell and appends one CSV row per repeat. A
sweep spawns this as a subprocess per cell, which keeps each vLLM engine
instance isolated: no GPU memory carried across models, and one cell that
dies cannot corrupt the rest of the sweep.

Usage:
    python -m elastic.cli.run_bench \\
        --model facebook/opt-66b --blob-dir staged/opt-66b \\
        --alpha-curve policy_out/opt-66b_B64_S1024_g128/alpha_curve.csv \\
        --kv-save-blob /path/to/kv.blob --slots-mem-gib 96 \\
        --n-prompts 64 --prompt-len 1024 --gen-len 128 --block-size 16 \\
        --csv-out /tmp/bench.csv --tag opt-66b_B64_S1024_g128

The solve writes summary.json next to alpha_curve.csv; the cell it names
(N_seq, S, gen_len; keys "B", "S", "gen_len") must equal --n-prompts /
--prompt-len / --gen-len.
`--alpha` runs one uniform α instead of a curve.

CSV columns (header written on first append):
    tag, model, blob_dir, n_prompts, prompt_len, gen_len, alpha_source,
    slots_mem_gib, repeat, init_s, wall_s, prefill_wall_s, decode_toks_s,
    prompt_toks, gen_toks, toks_s

Importing `elastic.vllm_runtime` is what registers the custom load_format
and pins the runtime's env defaults, so it must happen before the vLLM engine
is constructed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


# vLLM's gpu_memory_utilization. On the elastic path it does not change the
# footprint (num_gpu_blocks_override sizes the KV pool exactly and the
# elastic model skips the profiled "available memory"); it only sets vLLM's
# startup check that free memory ≥ this fraction of total. On the
# --reference path it sizes the KV pool. Memory safety comes from the slot
# budget (--slots-mem-gib), not from this value.
_GPU_MEMORY_UTILIZATION = 0.85


def _build_alpha_curve(csv_path: str | None, scalar: float | None) -> list[float] | float:
    """Return the alpha_curve config the streaming loader expects.
    The streaming loader accepts either a list[float] or a scalar."""
    if csv_path:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f'alpha-curve csv not found: {csv_path}')
        import pandas as pd
        df = pd.read_csv(p)
        if 'alpha' not in df.columns:
            raise ValueError(f'{csv_path}: no `alpha` column')
        return [float(a) for a in df['alpha'].tolist()]
    if scalar is None:
        raise ValueError('either --alpha-curve or --alpha required')
    return float(scalar)


def _kv_pool_override(manifest: dict, max_num_seqs: int, max_model_len: int,
                       block_size: int) -> tuple[int, float, str]:
    """Compute the exact `num_gpu_blocks_override` for the real working set:
    max_num_seqs sequences, each up to max_model_len tokens, plus vLLM's
    reserved null block.

    Under the elastic streaming loader vLLM's block count is the size of one
    layer's K and V buffers, and only n_slots such pairs exist: the patched
    runner allocates n_slots of them and the loader (ReadRouter) re-points
    each layer at its visit's buffers. The buffers are filled in load order
    (`BlockKVLayout.mem_block`) and address every block of the batch, so
    they need exactly this many blocks — fewer and the streaming loader
    refuses to start (`mem_blocks_needed`), more is memory the slot budget
    does not count. The budget (`StreamingModelLoader._compute_n_slots`,
    the solver's `_m_kv`, Eq. 7) charges the max_num_seqs·blocks_per_seq
    data blocks; the +1 null block of each slot's K and V buffers is the one
    block it does not count. vLLM's own figure cannot give it: the elastic
    model skips the profiling forward and reports the gpu_memory_utilization
    share of total memory as a placeholder, which vLLM divides across all
    layers — a function of the machine, not of the batch.

    Returns (num_gpu_blocks_override, buffer_gib, detail); buffer_gib is one
    layer's K and V buffers together, of which the run allocates n_slots."""
    cfg = manifest['config']
    n_heads = int(cfg['num_attention_heads'])  # OPT/BLOOM: no GQA → n_kv == n_heads
    head_dim = int(cfg['hidden_size']) // n_heads
    dtype_bytes = 2  # fp16, the only dtype stage_opt writes
    blocks_per_seq = (max_model_len + block_size - 1) // block_size
    num_blocks = max_num_seqs * blocks_per_seq + 1  # +1: vLLM null block
    per_block_bytes = block_size * 2 * n_heads * head_dim * dtype_bytes
    buffer_gib = (num_blocks * per_block_bytes) / 2**30
    detail = (f'{num_blocks} blocks ({blocks_per_seq}/seq × {max_num_seqs} +1), '
              f'{n_heads}h×{head_dim}d×{dtype_bytes}B, '
              f'block_size={block_size}')
    return num_blocks, buffer_gib, detail


_DISTINCT_BASES = [
    ("Paris is the capital city of France. It is a beautiful "
     "place to visit with many museums and historic sites. "),
    ("The Pacific ocean is the largest and deepest body of water "
     "on Earth, covering more than thirty percent of the surface. "),
    ("Modern computers process enormous amounts of data every "
     "second using billions of tiny transistors etched on silicon. "),
    ("Classical music has shaped composers across many centuries, "
     "from baroque counterpoint to romantic symphonic grandeur. "),
    ("Photosynthesis lets green plants convert sunlight, water and "
     "carbon dioxide into the sugars that feed almost every ecosystem. "),
    ("The history of mathematics stretches from ancient counting "
     "tablets to abstract modern theories of numbers and shapes. "),
    ("Volcanoes form where molten rock from deep inside the planet "
     "rises through cracks and erupts onto the surface as lava. "),
    ("A healthy diet balances proteins, carbohydrates and fats with "
     "vitamins and minerals drawn from a wide variety of foods. "),
]


def _build_prompts(model: str, prompt_len: int, n_prompts: int) -> list[dict]:
    """`n_prompts` prompts of exactly `prompt_len` tokens each, as token ids
    (`{"prompt_token_ids": ids}`): an English paragraph repeated and cut to
    `prompt_len` ids of the tokenizer's own encoding, BOS included where the
    tokenizer adds one. Handing vLLM the ids, not text, is what keeps the
    count exact: decoding the cut ids back to text and letting vLLM
    re-encode it would add a second BOS.

    Sequence i takes paragraph i mod 8, so neighbouring sequences generate
    different text and a token diff against the reference reports the first
    diverging position and sequence.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    def _fill(base: str) -> dict:
        text = base
        while len(tok(text).input_ids) < prompt_len:
            text = text + base
        ids = tok(text).input_ids[:prompt_len]
        return {"prompt_token_ids": ids}

    bases = [_fill(b) for b in _DISTINCT_BASES]
    return [bases[i % len(bases)] for i in range(n_prompts)]


def _csv_fieldnames() -> list[str]:
    return [
        'tag', 'model', 'blob_dir', 'n_prompts', 'prompt_len', 'gen_len',
        'alpha_source', 'slots_mem_gib', 'repeat',
        'init_s', 'wall_s', 'prefill_wall_s', 'decode_toks_s',
        'prompt_toks', 'gen_toks',
        'toks_s',
    ]


def _append_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        if first:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _run_reference(args) -> int:
    """Stock-vLLM reference run: plain in-memory KV, no elastic anywhere.

    This is the external correctness oracle, so it must not import
    `elastic.vllm_runtime`: that import registers the `elastic_streaming`
    load_format and pins the ELASTIC_* defaults (the in-process vLLM engine
    client among them), and the arrival decode path only engages under that
    loader.
    Without the import this really is vanilla vLLM rather than a
    configuration of the system under test, so it cannot share a bug with it.

    It uses the same deterministic prompts and greedy (temperature=0)
    sampling as the elastic path, so `--dump-tokens` output from the two is
    directly comparable.
    """
    from vllm import LLM, SamplingParams

    prompts = _build_prompts(args.model, args.prompt_len, args.n_prompts)
    print(f'[run_bench:ref] model={args.model} (vanilla vLLM, in-memory KV)',
          file=sys.stderr)
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.prompt_len + args.gen_len,
        max_num_seqs=args.n_prompts,
        block_size=args.block_size,
        gpu_memory_utilization=_GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False,
    )
    init_s = time.perf_counter() - t0
    # Same EOS handling as the elastic path: if one side stopped at EOS and
    # the other generated past it, the two token streams would differ in
    # length and every position after the first EOS would report a spurious
    # mismatch.
    sp = SamplingParams(max_tokens=args.gen_len, temperature=0,
                        ignore_eos=True)
    out = llm.generate(prompts, sp)
    if args.dump_tokens:
        toks = [list(o.outputs[0].token_ids) for o in out]
        Path(args.dump_tokens).write_text(json.dumps(toks))
        print(f'[run_bench:ref] dumped {len(toks)} token seqs -> '
              f'{args.dump_tokens}', file=sys.stderr)
    row = dict(tag=args.tag, model=args.model, mode='reference',
               n_prompts=args.n_prompts, prompt_len=args.prompt_len,
               gen_len=args.gen_len, init_s=round(init_s, 2), status='ok')
    print(json.dumps(row))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True,
                      help='HF model name (e.g. facebook/opt-66b)')
    ap.add_argument('--blob-dir', default=None,
                      help='stage_opt output directory (contains blob + '
                            'manifest.json). Required unless --reference.')

    g_alpha = ap.add_mutually_exclusive_group()
    g_alpha.add_argument('--alpha-curve', default=None,
                          help='CSV path with columns (layer, alpha); the '
                               'solve\'s summary.json must sit next to it')
    g_alpha.add_argument('--alpha', type=float, default=None,
                          help='scalar α (uniform across all layers)')

    _sm_default = float(os.environ.get('ELASTIC_SLOTS_MEM_GIB', '96'))
    ap.add_argument('--slots-mem-gib', dest='slots_mem_gib', type=float,
                      default=_sm_default,
                      help='memory reserved for the rotating layer slots, in '
                            'GiB (or $ELASTIC_SLOTS_MEM_GIB). It sizes n_slots '
                            'only: the loader fits n_slots·(weight + KV + '
                            '0.5·KV staging) inside it and lowers n_slots '
                            'until that holds. Transient allocations '
                            '(activations, framework scratch) are not bounded '
                            'by it, so keep it below free system RAM on '
                            'unified-memory hardware.')
    ap.add_argument('--kv-save-blob', default=None,
                      help='NVMe blob path for KV/ACT save (BlockKVStorage). '
                            'Required unless --reference: every KV/ACT block '
                            'is saved to and reloaded from it.')
    ap.add_argument('--n-prompts', type=int, default=1,
                      help='number of sequences N_seq (the batch size); the '
                            'whole batch is admitted as one prefill step')
    ap.add_argument('--block-size', type=int, default=16,
                      help='vLLM KV block size. Fixed explicitly so the '
                            'num_gpu_blocks_override computation is exact.')
    ap.add_argument('--prompt-len', type=int, default=256)
    ap.add_argument('--gen-len', type=int, default=32)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--csv-out', default=None,
                      help='throughput CSV the run appends its rows to '
                            '(one per repeat). Required unless --reference.')
    ap.add_argument('--tag', default='')
    ap.add_argument('--dump-tokens', default=None,
                      help='Write per-sequence generated token_ids (last '
                            'repeat) to this JSON path, for a token-for-token '
                            'comparison against a --reference dump.')
    ap.add_argument('--reference', action='store_true',
                      help='Vanilla-vLLM reference run: plain in-memory KV from '
                            'the HF model, with no elastic streaming, slots or '
                            'ACT. Same deterministic prompts and greedy decode, '
                            'so its --dump-tokens output is directly comparable '
                            'to an elastic run\'s: the slot-rotating KV/ACT-reproject '
                            'decode must emit the same tokens as stock vLLM. '
                            'Ignores --blob-dir, --alpha/--alpha-curve, '
                            '--slots-mem-gib, --kv-save-blob, --repeats and '
                            '--csv-out.')
    args = ap.parse_args(argv)

    if args.reference:
        return _run_reference(args)

    if args.blob_dir is None:
        ap.error('--blob-dir is required; only --reference runs without it')
    if args.csv_out is None:
        ap.error('--csv-out is required; only --reference runs without it')
    if args.kv_save_blob is None:
        ap.error('--kv-save-blob is required (every KV/ACT block is saved '
                 'to and reloaded from it); only --reference runs without it')

    # Import order matters: importing elastic.vllm_runtime registers the
    # custom load_format and pins the runtime's env defaults (the in-process
    # vLLM engine among them), and vLLM reads the environment as it builds
    # the vLLM engine.
    import torch
    import elastic.vllm_runtime  # noqa: F401  (registers load_format)
    from elastic.vllm_runtime import streaming_loader as _sl
    from vllm import LLM, SamplingParams

    alpha_curve = _build_alpha_curve(args.alpha_curve, args.alpha)
    alpha_source = ('csv:' + args.alpha_curve) if args.alpha_curve \
                     else f'scalar:{args.alpha}'

    # Cell cross-check. A curve is solved for one specific (N_seq, S, gen_len):
    # those determine n_slots as well as α itself, and running a different
    # cell with it mis-sizes the slots and shows up only as throughput, never
    # as an error. The solve's summary.json next to alpha_curve.csv names the
    # cell; it must equal this run's.
    if args.alpha_curve:
        _summ = Path(args.alpha_curve).parent / 'summary.json'
        if not _summ.exists():
            raise FileNotFoundError(
                f'{_summ} not found: the solve writes summary.json next to '
                f'alpha_curve.csv, and run_bench checks the solved cell '
                f'against its own arguments')
        _cell = json.loads(_summ.read_text())['cell']
        _want = {'B': args.n_prompts, 'S': args.prompt_len,
                  'gen_len': args.gen_len}
        _bad = {k: (int(_cell[k]), v) for k, v in _want.items()
                if int(_cell[k]) != int(v)}
        if _bad:
            raise RuntimeError(
                f'[run_bench] policy/bench cell mismatch {_summ}: '
                + ', '.join(f'{k}: solved={a} vs bench={b}'
                            for k, (a, b) in _bad.items())
                + '  — re-solve for this cell')

    print(f'[run_bench] model={args.model}', file=sys.stderr)
    print(f'[run_bench] blob-dir={args.blob_dir}', file=sys.stderr)
    print(f'[run_bench] alpha_source={alpha_source}', file=sys.stderr)
    print(f'[run_bench] slots_mem={args.slots_mem_gib} GiB', file=sys.stderr)
    print(f'[run_bench] n_prompts={args.n_prompts} prompt_len={args.prompt_len} '
            f'gen_len={args.gen_len}  repeats={args.repeats}', file=sys.stderr)

    # A sequence holds at most its prompt and gen tokens.
    max_model_len = args.prompt_len + args.gen_len
    # The whole batch is admitted in one scheduler step, which is its
    # prefill: max_num_seqs is the batch, and the step's token budget holds
    # every admitted prompt at up to max_model_len tokens (chunked prefill
    # off, below), whatever the prompts tokenize to.
    max_num_seqs = args.n_prompts
    max_num_batched_tokens = max_num_seqs * max_model_len
    # The staged blob's manifest describes the bytes on disk. The model is
    # built in its dtype, not the HF config's: the streaming loader binds
    # parameters directly to raw blob bytes, so a model built in another
    # dtype fails its first layer bind.
    manifest = json.loads((Path(args.blob_dir) / 'manifest.json').read_text())
    # Replace vLLM's KV-pool auto-sizing with the exact working-set block
    # count (see _kv_pool_override for why the automatic figure is unusable
    # here). `num_gpu_blocks_override` alone sizes each layer's K and V
    # buffers: the elastic model bypasses the memory-profiling forward, so
    # no profiled figure feeds into the block count.
    num_blocks_override, kv_buffer_gib, kv_detail = _kv_pool_override(
        manifest, max_num_seqs, max_model_len, args.block_size)
    print(f'[run_bench] max_num_seqs={max_num_seqs} '
            f'max_model_len={max_model_len} '
            f'max_num_batched_tokens={max_num_batched_tokens}',
            file=sys.stderr)
    print(f'[run_bench] KV pool override: {kv_buffer_gib:.2f} GiB per '
            f'KV buffer, n_slots buffers ({kv_detail})', file=sys.stderr)
    print(f'[run_bench] dtype={manifest["dtype"]} (from manifest)',
            file=sys.stderr)

    t0 = time.perf_counter()
    # FLASH_ATTN is the prefill kernel and the backend that holds the decode
    # arrival seam (vLLM's patched FlashAttentionImpl.forward); another
    # backend would attend the paged cache while its reloads are landing.
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    _attn_backend = AttentionBackendEnum["FLASH_ATTN"]

    llm = LLM(
        model=args.model,
        load_format='elastic_streaming',
        dtype=manifest['dtype'],
        attention_backend=_attn_backend,
        model_loader_extra_config={
            'blob_dir': args.blob_dir,
            'alpha_curve': alpha_curve,
            'slots_mem_bytes': int(args.slots_mem_gib * 2**30),
            'kv_save_blob': args.kv_save_blob,
        },
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        block_size=args.block_size,
        num_gpu_blocks_override=num_blocks_override,
        gpu_memory_utilization=_GPU_MEMORY_UTILIZATION,
        # The batch's prefill is one scheduler step: the in-process vLLM
        # engine has every prompt before its first step, and the scheduler
        # admits each one whole (no chunking) under a budget that holds them
        # all — so the zig-zag prefill streams each layer's weight once for
        # the whole batch, and the decode steps that follow are decode only.
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=False,
        # Rotating resident KV requires prefix caching off. Prefix caching lets
        # sequences share physical blocks and skip recomputing them, so
        # those blocks are never saved for the sharing sequences, while the
        # reload path assumes every sequence owns and reloads its own
        # blocks. The resulting holes are silent: the slot's K and V buffers
        # are reused across layers, so reading a hole yields another layer's KV
        # instead of an error. Stock vLLM tolerates sharing because it keeps
        # those blocks resident.
        enable_prefix_caching=False,
    )
    init_s = time.perf_counter() - t0
    print(f'[run_bench] init_s={init_s:.2f}', file=sys.stderr)

    prompts = _build_prompts(args.model, args.prompt_len, args.n_prompts)
    # Every sequence runs the full gen_len. If sequences finished at EOS at
    # different times the batch would shrink part-way through, and the
    # measured throughput would be an average over several batch sizes
    # rather than a figure for N_seq.
    sp = SamplingParams(max_tokens=args.gen_len, temperature=0,
                          ignore_eos=True)

    def _sync():
        # The vLLM engine runs in this process, so its kernels are this
        # process's to wait for: a repeat's clock starts on an idle device.
        torch.cuda.synchronize()

    rows: list[dict] = []
    for r in range(args.repeats):
        # No torch.cuda peak-memory tracking here: on unified memory the
        # footprint that matters is system RAM, registered host buffers
        # included, which the caching allocator does not see. Memory is
        # measured externally.
        #
        # Steady-state decode throughput is reported separately from the
        # end-to-end figure, which folds in the prefill forward over N_seq·S
        # tokens; α governs the decode reload, so a headline number
        # containing prefill cannot be attributed to the policy.
        #
        # The split is the batch's prefill/decode boundary, which the
        # arrival gate marks on the forward pass's stream at the batch's
        # first decode step (`streaming_loader.decode_boundary`, cleared here
        # first): the decode is the device time from that mark to the end of
        # the generate(), decode steps t = 1..gen−1 producing (gen−1)·N_seq
        # tokens, and the prefill wall is the rest of the wall — input
        # processing, admission, and the prefill step's forward and sampler.
        # A gen-1 batch has no decode step, so no mark: its whole wall is
        # prefill.
        _sl.decode_boundary = None
        _sync()
        t0 = time.perf_counter()
        out = llm.generate(prompts, sp)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        wall_s = time.perf_counter() - t0

        prompt_toks = sum(len(o.prompt_token_ids) for o in out)
        gen_toks    = sum(len(o.outputs[0].token_ids) for o in out)
        toks_s      = gen_toks / wall_s if wall_s > 0 else 0.0
        prefill_wall = wall_s
        decode_toks_s = None
        if _sl.decode_boundary is not None:
            decode_s = _sl.decode_boundary.elapsed_time(end) / 1e3
            prefill_wall = wall_s - decode_s
            decode_toks_s = round(
                sum(len(o.outputs[0].token_ids) - 1 for o in out)
                / decode_s, 3)

        rows.append(dict(
            tag=args.tag, model=args.model, blob_dir=args.blob_dir,
            n_prompts=args.n_prompts, prompt_len=args.prompt_len,
            gen_len=args.gen_len, alpha_source=alpha_source,
            slots_mem_gib=args.slots_mem_gib,
            repeat=r, init_s=round(init_s, 4) if r == 0 else 0.0,
            wall_s=round(wall_s, 4),
            prefill_wall_s=round(prefill_wall, 4),
            decode_toks_s=decode_toks_s,
            prompt_toks=prompt_toks, gen_toks=gen_toks,
            toks_s=round(toks_s, 3),
        ))
        print(f'[run_bench] r{r}: wall={wall_s:.3f}s  '
                f'toks/s={toks_s:.2f}'
                + (f'  decode_toks/s={decode_toks_s}' if decode_toks_s else ''),
                file=sys.stderr)

    # The runtime's end: the runner's shutdown calls the streaming loader's
    # `_elastic_shutdown` first — the loader stops, the writeback lands every
    # write still owed, the NVMe engine closes — while the KV caches those
    # threads touch are still alive. The vLLM engine runs in this process, so
    # this call is the only one that reaches it.
    llm.llm_engine.engine_core.shutdown()

    _append_csv(Path(args.csv_out), rows)

    if args.dump_tokens:
        toks = [list(o.outputs[0].token_ids) for o in out]
        Path(args.dump_tokens).write_text(json.dumps(toks))
        print(f'[run_bench] dumped {len(toks)} token streams → '
                f'{args.dump_tokens}', file=sys.stderr)
    print(json.dumps(rows[-1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
