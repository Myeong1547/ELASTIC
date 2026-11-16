"""Policy entry point — the hybrid policy (Algorithm 3) over
α^0..α^{L−1}.

Prices one (N_seq, S, gen_len) cell from a units record (`cli_measure_units`
output) and solves its per-layer α curve:

    python -m elastic.policy.run_solve \\
        --from-units benchmarks/results/units_cache/units_opt-66b_S1024.json \\
        --B 64 --S 1024 --gen-len 128 --d-model 9216 --num-layers 64 \\
        --out-dir policy_out/opt-66b_B64_S1024_g128

`--num-layers` is L, the model's total decoder-layer count. t_act is not
an input: the model encodes t_act = 0.5·t_kv directly into C2.

Outputs (under --out-dir):
    alpha_curve.csv     per-layer α^i  (i = 0..L−1)
    lead_profile.csv    per-layer T_ahead^i + T_NVMe^i + T_comp^i [s]
                        (columns T_ahead_s, T_load_s, T_compute_s)
    summary.json        feasibility/regime + cost constants + n_slots + the
                        solved cell (run_bench cross-checks it)
    alpha_plot.png      α + ahead-time trajectory + memory headroom
    trajectory_plot.png stacked per-layer T_ahead^i / ΔT^i / α^i (shared x)
    sanity.txt          plaintext sanity report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


from .from_units import load_units, units_to_solver_inputs
from .solver import (
    SolverInputs, _cost_constants, _c_mem, _num_slots, solve,
)
from .plot_policy import plot_solution, plot_trajectory
from .sanity import check


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    g_in = ap.add_argument_group("units record and pricing")
    g_in.add_argument("--from-units",   type=Path, required=True,
                      help="JSON file from `cli_measure_units --output`; "
                            "`from_units.units_to_solver_inputs` prices the "
                            "cell from it")
    g_in.add_argument("--read-gbps", type=float, default=None,
                          help="the read rate (GB/s) that prices t_w and "
                                "t_kv, a hand-entered drive constant; "
                                "omitted → the record's measured rate. "
                                "BW_NVMe keeps the measured rate")
    g_in.add_argument("--write-gbps", type=float, default=None,
                          help="the write rate (GB/s) that prices t_save_kv; "
                                "omitted → the record's write rate")
    g_in.add_argument("--block-size", type=int, default=16,
                          help="runtime KV block size (tokens); must equal "
                                "run_bench --block-size. Scales the write "
                                "price (t_save_kv = block_size · 4·d_model / "
                                "write) and rounds M_KV up to whole blocks of "
                                "S+gen_len (Eq. 7), which sets n_slots and "
                                "C_mem")

    g_shape = ap.add_argument_group("model + workload sizing")
    g_shape.add_argument("--B",           type=int, required=True,
                          help="N_seq, the number of sequences")
    g_shape.add_argument("--S",           type=int, required=True,
                          help="prompt length; must equal the record's S_op")
    g_shape.add_argument("--d-model",     type=int, required=True,
                          help="hidden_size (12288 for OPT-175B)")
    g_shape.add_argument("--num-layers",  type=int, required=True,
                          help="L — 40/48/64/96 for opt-13b/30b/66b/175b")
    g_shape.add_argument("--gen-len",     type=int, required=True,
                          help="decode tokens per seq; must equal the "
                               "record's gen_len. The cell is priced at the "
                               "endpoint context S+gen_len, and the resident "
                               "KV grows to S+gen_len, so n_slots and the "
                               "ahead-time bound C_mem price that peak.")

    g_sys = ap.add_argument_group("system constants")
    g_sys.add_argument("--slots-mem-bytes", type=float, default=96 * 2**30,
                        help="byte budget of the slot set (weights + KV + ACT "
                              "staging). On coherent unified-memory hardware "
                              "a GPU allocation is system RAM, so the real "
                              "limit is free system RAM; keep a conservative "
                              "margin below it.")
    g_sys.add_argument("--beta",        type=float, default=0.5,
                        help="band position for the midline tracker. "
                              "Each layer aims the lead at β of the way from the "
                              "lower bound to the upper bound: 0.5=midline "
                              "(default), 0=hug floor, 1=hug OOM/burn ceiling.")

    ap.add_argument("--out-dir",  default=None,
                      help="if set, write alpha_curve.csv, lead_profile.csv, "
                            "summary.json, alpha_plot.png, trajectory_plot.png "
                            "and sanity.txt under this directory")
    return ap


def _write_outputs(res, inp: SolverInputs, out: Path, report) -> None:
    out.mkdir(parents=True, exist_ok=True)
    C1, C2, C3, C4 = _cost_constants(inp)
    n = inp.num_layers                       # n == L

    # per-layer α value (layers 0..L−1)
    with (out / "alpha_curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "alpha"])
        if res.feasible:
            for i, a in enumerate(res.alpha):
                w.writerow([i, f"{a:.6f}"])

    # T_ahead profile (T_ahead^0..T_ahead^{L−1}) + per-layer T_NVMe^i /
    # T_comp^i
    with (out / "lead_profile.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "T_ahead_s", "T_load_s", "T_compute_s"])
        if res.feasible:
            for i in range(n):
                a = res.alpha[i]
                tl = f"{(C1 - C2 * a):.6e}"
                tc = f"{(C3 + C4 * a):.6e}"
                w.writerow([i, f"{res.T_ahead[i]:.6e}", tl, tc])

    if res.feasible:
        plot_solution(res, out / "alpha_plot.png")
        plot_trajectory(res, out / "trajectory_plot.png")

    # sanity + machine-readable summary
    with (out / "sanity.txt").open("w") as f:
        f.write(f"feasible: {res.feasible}\n")
        f.write(f"status:   {res.status}\n")
        for line in report.lines:
            f.write(line + "\n")
    with (out / "summary.json").open("w") as f:
        json.dump({
            "feasible":       res.feasible,
            "status":         res.status,
            "alpha_sum":      float(np.nansum(res.alpha)),
            "num_layers":     inp.num_layers,
            # Solved cell identity — run_bench cross-checks these against its
            # own --n-prompts/--prompt-len/--gen-len and refuses a curve solved
            # for another cell: n_slots and α are cell-specific.
            "cell":           {"B": int(inp.n_seq), "S": int(inp.S),
                                "gen_len": int(inp.gen_len)},
            "slots":              int(_num_slots(inp)),
            "cost":           {"C1": C1, "C2": C2, "C3": C3, "C4": C4,
                                "P": C2 + C4, "Q": C1 - C3},
            "verify":         report.verify,
        }, f, indent=2)


def main() -> None:
    args = _cli().parse_args()

    units_blob = load_units(args.from_units)
    inp = units_to_solver_inputs(
        units_blob["units"],
        n_seq=args.B, S=args.S, d_model=args.d_model,
        num_layers=args.num_layers, gen_len=args.gen_len,
        slots_mem_bytes=args.slots_mem_bytes, beta=args.beta,
        block_size=args.block_size,
        read_GBps=args.read_gbps, write_GBps=args.write_gbps,
    )
    print(f"[policy] BW_NVMe derived from units: "
            f"{inp.BW_NVMe:.3e} B/s (the record's read rate)",
            file=sys.stderr)
    print(f"[policy] loaded units from {args.from_units}",
            file=sys.stderr)

    C1, C2, C3, C4 = _cost_constants(inp)
    print(f"[policy] L={inp.num_layers}  N_seq={inp.n_seq} S={inp.S} "
            f"d_model={inp.d_model}", file=sys.stderr)
    print(f"[policy] C1={C1:.4e}  C2={C2:.4e}  C3={C3:.4e}  C4={C4:.4e}  "
            f"P={C2+C4:.4e}  Q={C1-C3:.4e}", file=sys.stderr)
    _n_slots = _num_slots(inp)
    _cmem = _c_mem(inp)
    print(f"[policy] n_slots={_n_slots} from the byte budget; "
            f"ahead-time bound C_mem=n_slots·(M_W+M_KV)/BW "
            f"={_cmem:.3e} s; "
            f"slots_mem={args.slots_mem_bytes:.3e} B  BW_NVMe={inp.BW_NVMe:.3e} B/s",
            file=sys.stderr)

    res = solve(inp)

    print(f"[policy] feasible={res.feasible}  "
            f"Σα={float(np.nansum(res.alpha)):.4f}  "
            f"status={res.status}", file=sys.stderr)

    report = check(res)
    for line in report.lines:
        print("  " + line, file=sys.stderr)

    if args.out_dir:
        _write_outputs(res, inp, Path(args.out_dir), report)
        print(f"[policy] wrote {args.out_dir}/", file=sys.stderr)

    if not res.feasible:
        sys.exit(3)


if __name__ == "__main__":
    main()
