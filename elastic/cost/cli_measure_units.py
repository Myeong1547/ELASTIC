"""CLI: measure the unit costs for one model at one operational (S, gen).

Usage:
    python -m elastic.cost.cli_measure_units \\
        --model opt-66b --s-op 1024 --gen-len 128 \\
        --output benchmarks/results/units_cache/units_opt-66b_S1024.json
        [--remeasure]

The read rate is measured on staged/<model>/model.bin, the blob the bench
streams.

Output JSON:
    {
      "timestamp": "<YYYYmmdd_HHMMSS>",
      "env": {gpu, torch, cuda, elastic: {ELASTIC_* as set}},
      "model": "opt-66b",
      "units": {
        "read_GBps", "write_GBps",
        "T_kv_gen_per_token_s", "T_attn_per_token_s", "T_dense_per_token_s",
        "S_op", "gen_len", "weight_bytes", "provenance": {...}
      }
    }

The record has no batch: every priced primitive is a per-byte device rate
or a per-token kernel cost, so one record per (model, S, gen) prices every
batch size through `elastic.policy.from_units`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .config import ALL_MODELS
from .decoder_layer import make_random_layer
from .measure_units import measure_unit_times


def _default_weight_blob(model: str) -> Path | None:
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        p = root / "staged" / model / "model.bin"
        if p.exists():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ALL_MODELS.keys()))
    ap.add_argument("--s-op", type=int, default=1024,
                    help="operational sequence length the layer step is measured at")
    ap.add_argument("--gen-len", type=int, default=128,
                    help="decode tokens per sequence: the LSE call is "
                         "measured at the last decode step's context, "
                         "s-op + gen-len − 1 keys")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--remeasure", action="store_true",
                    help="measure even if --output already exists")
    args = ap.parse_args(argv)

    if args.output.exists() and not args.remeasure:
        print(f"[skip] {args.output} already exists; pass --remeasure to redo")
        return 0

    blob = _default_weight_blob(args.model)
    if blob is None:
        sys.exit(f"error: no staged weight blob staged/{args.model}/model.bin; "
                 f"stage the model first")

    cfg = ALL_MODELS[args.model]
    layer = make_random_layer(cfg, device="cuda")
    try:
        units = measure_unit_times(cfg, str(blob), layer, S_op=args.s_op,
                                   gen_len=args.gen_len)
    finally:
        del layer
        torch.cuda.empty_cache()

    out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "elastic": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith("ELASTIC_")},
        },
        "model": args.model,
        "units": asdict(units),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print(f"\nunits {args.model} S_op={units.S_op} gen_len={units.gen_len}:")
    print(f"  read_GBps             {units.read_GBps:10.2f}")
    print(f"  write_GBps            {units.write_GBps:10.2f}")
    print(f"  T_kv_gen_per_token_s  {units.T_kv_gen_per_token_s*1e6:10.3f} µs")
    print(f"  T_attn_per_token_s    {units.T_attn_per_token_s*1e9:10.3f} ns")
    print(f"  T_dense_per_token_s   {units.T_dense_per_token_s*1e6:10.3f} µs")
    print(f"  weight_bytes          {units.weight_bytes}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
