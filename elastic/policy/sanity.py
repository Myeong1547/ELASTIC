"""Human-readable sanity report on the policy's α solution.

If `solve` returns `feasible=True` the policy emitted a curve; `verify()`
recomputes the ahead-time trajectory and checks the bounds, the upper bound
C_mem and the near-0 landing. This module wraps `verify()` for printing and
surfaces the *shape* of the per-layer α values (cold-start ramp + plateau).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .solver import SolveResult, verify


@dataclass
class SanityReport:
    lines: List[str]
    # The `verify()` dict the report was built from, so a caller that also
    # records it does not run the checks a second time.
    verify: dict


def check(result: SolveResult, head: int = 5, plateau_tol: float = 1e-3) -> SanityReport:
    lines: List[str] = []

    v = verify(result)
    if not result.feasible:
        return SanityReport([f"infeasible: {result.status}"], verify=v)

    # The C_mem bound and the near-0 landing are guarantees the
    # ok/load-bound regimes meet — but compute-bound provably cannot (Q<0 ⇒
    # the ahead time only grows, no α shrinks it). There the overflow is a
    # memory warning, not a policy failure: at runtime the loader acquires
    # only a slot the writeback has released, so the slot rotation bounds
    # residency.
    tier = "warn-memory" if v["memory_warning"] else "ok"

    alpha = result.alpha
    n = alpha.size
    lines.append(
        f"regime={v['regime']}  tier={tier}  "
        f"α in [{alpha.min():.4f}, {alpha.max():.4f}], "
        f"mean={alpha.mean():.4f}, Σα={float(alpha.sum()):.4f}"
    )
    lines.append(
        f"lead T_ahead^i ∈ [{v['ahead_min']:.4e}, {v['ahead_max']:.4e}] s "
        f"(C_mem ceiling {v['mem_cap']:.4e} s)"
    )

    if not v["bounds_ok"]:
        lines.append(f"  ! bounds violated: "
                       f"min={alpha.min()} max={alpha.max()}")
    if not v["traj_match"]:
        lines.append("  ! stored lead disagrees with recomputed trajectory")
    if v["memory_warning"]:
        lines.append("  note: compute-bound: lead exceeds C_mem and won't land "
                       "at 0 — expected; the slot rotation bounds residency "
                       "at runtime (not a policy failure)")
    elif not v["memory_ok"]:
        lines.append("  ! C_mem ceiling exceeded by some T_ahead^i")
    if not v["memory_warning"] and not v["end_ahead_ok"]:
        lines.append(f"  ! end T_ahead^{{L−1}} not near 0: {result.T_ahead[-1]:.4e} s "
                       f"(landing tol {v['landing_tol']:.4e} s)")

    # Shape: cold-start ramp + plateau (informational only).
    if n > head + 1:
        tail = alpha[head:]
        plateau_var = float(tail.var())
        plateau_mean = float(tail.mean())
        head_mean = float(alpha[:head].mean())
        if plateau_var > plateau_tol:
            lines.append(
                f"tail (i ≥ {head}) varies: var={plateau_var:.3e} "
                f"> tol {plateau_tol:.0e} — no clean plateau "
                f"(head mean {head_mean:.4f}, tail mean {plateau_mean:.4f})"
            )
        else:
            lines.append(
                f"plateau α ≈ {plateau_mean:.4f} (var {plateau_var:.2e}, "
                f"head mean {head_mean:.4f})"
            )

    return SanityReport(lines, verify=v)
