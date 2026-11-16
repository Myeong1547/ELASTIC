"""Plots of a solve: per-layer α value, ahead time T_ahead, memory headroom."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .solver import (
    SolveResult, _cost_constants, _c_mem,
    ahead_time_bounds, compute_delta_T, compute_T_target,
)


def plot_solution(result: SolveResult, out_path: Path) -> None:
    """Three-panel summary: per-layer α value, ahead-time trajectory, memory
    headroom.

    Memory headroom (panel 3) is `C_mem − T_ahead^i` in seconds — positive
    everywhere means the policy respected the upper bound C_mem.
    """
    inp = result.inputs
    n = inp.num_layers
    C1, C2, C3, C4 = _cost_constants(inp)
    P = C2 + C4
    Q = C1 - C3
    c_mem_s = _c_mem(inp)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # per-layer α value — layers 0..L−1
    layers = np.arange(n)
    ax1.bar(layers, result.alpha, width=1.0)
    ax1.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
    mean_target = Q / P
    ax1.axhline(mean_target, color="red", linewidth=0.8, linestyle="--",
                  label=f"mean target = Q/P = {mean_target:.3f}")
    ax1.legend(loc="best")
    ax1.set_xlabel("layer i (0..L−1)")
    ax1.set_ylabel("α^i")
    ax1.set_title("per-layer α value")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # ahead time T_ahead^0..T_ahead^{L−1}
    tahead_ms = result.T_ahead * 1000.0
    ax2.plot(np.arange(n), tahead_ms, marker="o", markersize=2,
              label="T_ahead^i")
    ax2.axhline(c_mem_s * 1000.0, color="red", linewidth=0.8,
                  linestyle="--",
                  label=f"C_mem={c_mem_s*1000:.1f} ms")
    ax2.axhline(0.0, color="grey", linewidth=0.6, linestyle=":")
    ax2.set_xlabel("layer i (0..L−1)")
    ax2.set_ylabel("T_ahead^i [ms]")
    ax2.set_title("loader lead T_ahead")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # memory headroom = (C_mem − T_ahead^i)
    headroom_ms = (c_mem_s - result.T_ahead) * 1000.0
    ax3.plot(np.arange(n), headroom_ms, marker="o", markersize=2)
    ax3.axhline(0.0, color="red", linewidth=0.8, linestyle="--",
                  label="ceiling exceeded")
    ax3.set_xlabel("i (lead index 0..L−1)")
    ax3.set_ylabel("(C_mem − T_ahead^i) [ms]")
    ax3.set_title("memory headroom")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"L={n}  N_seq={inp.n_seq} S={inp.S} d_model={inp.d_model}  "
        f"t_w={inp.t_w*1000:.2f}ms  "
        f"t_kv={inp.t_kv*1e6:.1f}µs  "
        f"T_fwd={inp.T_fwd*1e6:.1f}µs  T_kv_gen={inp.T_kv_gen*1e6:.1f}µs",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectory(result: SolveResult, out_path: Path) -> None:
    """Three stacked panels sharing the layer axis (x = layer i, 0..L−1):

        1. T_ahead^i  — the ahead time, inside the ahead-time bounds the
                      policy keeps it in: upper = C_burn^i = min(C_mem,
                      remaining_burn^i) (the C_mem-clamped landing cap — flat
                      at C_mem early, declining to 0 at the end) and
                      lower = C_floor^i.
        2. ΔT^i     — per-layer ΔT actually applied (T_comp − T_NVMe
                      with the layer's launch credit); the sign shows banking
                      (>0) vs burning (<0). This is what T_ahead integrates,
                      not T_ahead[i]−T_ahead[i−1].
        3. α^i      — per-layer α value.

    Infeasible results (NaN curve) are skipped.
    """
    if not result.feasible:
        return
    inp = result.inputs
    n = inp.num_layers
    layers = np.arange(n)
    tahead_ms = result.T_ahead * 1000.0
    dT_ms = compute_delta_T(result.alpha, inp) * 1000.0
    lower, upper = ahead_time_bounds(inp)
    target = compute_T_target(inp)
    lower_ms, upper_ms = lower * 1000.0, upper * 1000.0
    target_ms = target * 1000.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # 1) T_ahead^i with the upper bound C_burn^i and the lower bound C_floor^i
    ax1.fill_between(layers, lower_ms, upper_ms, color="tab:blue", alpha=0.08)
    ax1.plot(layers, upper_ms, color="red", linewidth=1.0, linestyle="--",
             label="upper = C_burn^i = min(C_mem, −(L−i)·ΔT|α=0)")
    ax1.plot(layers, lower_ms, color="tab:purple", linewidth=1.0,
             linestyle="--", label="lower = C_floor^i")
    ax1.plot(layers, target_ms, color="tab:gray", linewidth=1.0,
             linestyle="-.", label=f"T_target = C_floor^i + β·(C_burn^i − C_floor^i), β={inp.beta:.2f}")
    ax1.plot(layers, tahead_ms, marker="o", markersize=2, color="tab:blue",
             label="T_ahead^i")
    ax1.axhline(0.0, color="grey", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("T_ahead^i  [ms]")
    ax1.set_title("loader lead T_ahead (bounded by the OOM-clamped band)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2) ΔT^i (the time gained or lost per layer)
    ax2.plot(layers, dT_ms, marker="o", markersize=2, color="tab:green")
    ax2.axhline(0.0, color="red", linewidth=0.8, linestyle="--",
                label="ΔT=0 (bank ↑ / burn ↓)")
    ax2.set_ylabel("ΔT^i  [ms]")
    ax2.set_title("per-layer surplus")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) α^i
    ax3.bar(layers, result.alpha, width=1.0, color="tab:orange")
    ax3.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
    ax3.set_ylabel("α^i")
    ax3.set_title("per-layer α value")
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("layer i (0..L−1)")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"{result.status}  |  L={n}  N_seq={inp.n_seq} S={inp.S} "
        f"d_model={inp.d_model}",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
