"""Per-layer α^i curve via the hybrid policy (Algorithm 3), a
non-linear rule applied layer by layer.

α^i, the activation ratio, is the fraction of layer i's blocks that are
stored as activation blocks (one hidden vector per token) instead of KV
blocks (key/value pairs). Activation blocks are half the bytes, so raising
α trades NVMe device time for GPU kv_gen time. The policy picks the α^i
that keep the loader and the forward pass in step.

Layers i = 0..L−1 (L total; the code counts from 0 where the paper counts
1..L), one α^i ∈ [0, 1] per layer. The loader runs ahead of the forward
pass; the carried ahead time is the accumulated ΔT *beyond the lower bound
C_floor*

    T_ahead^i = Σ_{m=0..i} ΔT(α^m)   (floored at 0 on an unavoidable stall)

where ΔT(α) = T_comp(α) − T_NVMe(α) is the time gained or lost at one
layer. The problem is sequential: each layer spends or banks ahead time,
and the ahead time must land back near 0 by the end so the slots are
emptied. The policy supplies the needed ΔT where it can and, where it
cannot, takes the stall and keeps going — a workload whose loader cannot
keep up even at α ≡ 1 is a legitimate regime to plan for, not an
infeasible one. The solver refuses to produce a curve only when one
layer's working set does not fit the slot memory budget.

Cost model — the measured primitives (Eqs. 15–16), with t_act = 0.5·t_kv
folded into C2, so there is no t_act input. C1..C4 and P are code names
with no counterpart in the paper:

    T_NVMe(α)    = C1 − C2·α          C1 = t_w + N_seq·(t_kv + t_save_kv)
    T_comp(α)    = C3 + C4·α          C2 = 0.5·N_seq·(t_kv + t_save_kv)
    ΔT(α)        = (C3 − C1) + P·α     C3 = N_seq·T_fwd
                                       C4 = N_seq·T_kv_gen
                                       P  = C2 + C4   (the ΔT slope)

T_fwd is the forward pass time of one sequence through one decoder layer
in a decode step, including the new token's own K/V projection; T_kv_gen is
the extra kv_gen of the context that was stored as activation blocks.

T_NVMe is device time, not read time. The one NVMe serves both the step's
reloads and the step's KV/activation saves, and a landing write takes the
drive away from the reads rather than merely queueing behind them, so the
save's device time belongs in T_NVMe. t_save_kv carries the same α
dependence as t_kv (an activation block is half a KV block going out as
well as coming in), so it folds into C1 and C2 and adds no structure. It
differs in scale: a step reloads the whole context (t_kv ∝ S + gen) but
writes only the current block (t_save_kv ∝ block_size), so neither is
derivable from the other. The write unit cost is the drive's
post-cache-exhaustion floor rate.

The predicted step time is T_total = Σ_i max(T_NVMe(α^i), T_comp(α^i)).

P is the ΔT gained per unit α (more activation blocks ⇒ less load + more
kv_gen compute ⇒ larger ΔT). The priced primitives are positive, so P > 0:
α = 1 banks ahead time and α = 0 burns it.

Launch credit and the lower bound C_floor (the `floor`). The minimum
amount that must be loaded before any layer can start computing is the
weight plus the first activations, taken as one sequence's worth (t_act),

    floor = C_floor = t_w + t_act = t_w + C2_read/N_seq

(t_act = C2_read/N_seq, C2_read = ½·N_seq·t_kv; the device-time C2 also
carries t_save_kv, see _c_floor). A layer is *credited* this amount —
ΔT drops the launch bytes from its load — when the launch was already paid
for off the critical path: at cold-start (layer 0's weight is resident
before the pass), and right after a stall (the forward pass idled while at
least the C_floor bytes streamed in). T_ahead therefore tracks only the
slack *above* C_floor, since those bytes are reloaded on every layer
regardless. Algorithm 3 has neither the launch credit nor the floor of
T_ahead at 0; both are code-only.

Per-layer α is chosen against the target line T_target (Eqs. 22–25). Each
layer aims the ahead time at β of the way from its lower bound C_floor^i
to its upper bound C_burn^i (β = inp.beta, 0.5 = halfway) and picks the
α that lands there:

  * target:     T_ahead → c_floor_i + β·(C_burn − c_floor_i)   (β=½ ⇒ midpoint)
  * clip range: ΔT ∈ [ΔT(α=0), ΔT(α=1)]          (α ∈ [0, 1])

The needed change (T_target − T_ahead) is clipped into this range. When
T_target is reachable the inverse is exact and α lands strictly inside
(0, 1); when it is not, the clip saturates α to the near edge — the gain
edge (α=1) when the whole range is below the target (can't bank enough),
the burn edge (α=0) when it is above (can't burn enough). The rule is
deterministic and lands itself: the exclusive remaining_burn drives
T_target → 0 over the final layers, so the policy walks the ahead time to
0 with no separate landing logic. In steady state it sits at ΔT ≈ 0 — the
T_NVMe == T_comp crossover α — which is the design balance point. The
load-bound regime is just the gain-edge saturation (bank all, stall next).

Memory (the static OOM bound C_mem on the ahead time, in time units):

    n_slots      = N_slots (Eq. 9), the number of rotating layer slots —
                   each holds a weight buffer, K and V buffers and an
                   activation buffer (a single coupled depth). A capacity in
                   bytes: n_slots = min(⌊slots_mem / (weight_bytes + 1.5·M_KV)⌋, L)
                   with M_KV = 4·N_seq·⌈(S+gen_len)/block_size⌉·block_size·d
                   (Eq. 7). Not time-derived — the slot set is a memory
                   capacity and all the bytes are known, so it needs no
                   bandwidth. Only C_mem below is a time quantity.
    C_mem        = n_slots · (M_W + M_KV) / BW   (Eq. 19) the slot set's
                   loaded bytes as load time — the upper bound on the ahead
                   time: what is loaded ahead sits in the slots, so the ahead
                   time cannot exceed the disk time they hold. A slot's
                   loaded bytes are at most M_W + M_KV (up to the blob's
                   per-tensor 4 KiB padding of the weights): a step reads
                   at most ⌈(S+gen_len)/block_size⌉ blocks per sequence,
                   and each activation block read into the activation
                   buffer stands in for a KV block of twice its size, so a
                   layer reads at most M_W + (1 − α/2)·M_KV. The one place
                   BW converts memory → time.

The ahead time may never exceed C_mem (C_mem caps the upper bound
C_burn^i). A backward look-ahead "landing cap" remaining_burn^i — the
code's form of the −(L−i)·ΔT|α=0 term of Eq. 20 (see `_remaining_burn`),
the most ahead time that the layers *after* i can still burn off down to 0
(exclusive, so remaining_burn^{L−1} = 0) — squeezes the ahead time toward
0 near the end, forcing an exact-0 landing on the last layer.

Regimes (reported via SolveResult.status):
  * neg-mem     n_slots < 1 — not even one layer's working set fits.
                Infeasible (NaN curve).
  * compute-bound / load-bound / ok — the policy always emits a curve;
    status just records which regime it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class SolverInputs:
    """All numeric inputs to the policy.

    Cost primitives (layer-uniform, seconds). t_act is not an input: an
    activation block is by construction half a KV block, so t_act =
    0.5·t_kv is folded into C2 = 0.5·N_seq·(t_kv + t_save_kv).
    """

    t_w:        float   # weight load time per layer visit
    t_kv:  float   # KV load per layer visit, per (per-sequence) item
    T_fwd:      float   # forward pass time per item (no kv_gen)
    T_kv_gen:   float   # kv_gen (projection) per item, activation blocks only
    t_save_kv: float   # KV save per layer visit, per (per-sequence) item —
                        #   the device time the step's write takes away from
                        #   the reads, not the write's own latency. Scales with
                        #   block_size (one block per sequence per layer per
                        #   step), not with S. An activation block halves it
                        #   exactly as it halves t_kv, so it folds into C1
                        #   and C2.

    # Sizing.
    n_seq:      int     # N_seq, the number of sequences
    S:          int     # prompt length (resident KV per seq at decode start)
    d_model:    int     # hidden_size
    num_layers: int     # L — the model's decoder layer count
    gen_len:    int     # decode tokens appended per seq. The resident KV grows
                        #   to S + gen_len over the decode, so the slot sizing
                        #   and C_mem use S + gen_len (the decode peak).
    block_size: int     # tokens per block; M_KV (Eq. 7) is reserved in whole
                        #   blocks of S + gen_len.

    # System constants. slots_mem_bytes is the slot memory budget M_budget
    # (weight, K, V and activation buffers). On a coherent unified-memory
    # machine the GPU addresses the whole system pool, so the ceiling is total
    # system RAM rather than a separate VRAM carve-out, and exceeding it shows
    # up as an ordinary system OOM. The budget must already leave headroom for
    # the OS and transient allocations; it is used directly with no further
    # derating. The runtime takes the same figure as `slots_mem_bytes` in
    # model_loader_extra_config (run_bench --slots-mem-gib).
    slots_mem_bytes: float
    # β — the target line's position within the ahead-time bounds. Each layer
    # aims the ahead time at β of the way from the lower bound (C_floor^i) to
    # the upper bound (C_burn^i):
    #     T_target = c_floor_i + β·(C_burn − c_floor_i)
    # 0.5 = halfway. 0.0 hugs C_floor (minimal ahead time, earliest α burn);
    # 1.0 hugs C_burn (maximal ahead time, most α banking). β does not scale
    # slots_mem — it is purely the fractional position between the ahead
    # time's lower and upper bound.
    beta:        float
    # Measured NVMe read rate in bytes/s (the units record's read_GBps), used
    # to convert slot bytes into ahead time (C_mem).
    BW_NVMe:     float
    # Per-layer weight bytes M_W: the units record's weight_bytes (the
    # layer's tensor bytes aligned to 4 KiB), which keeps _num_slots pure
    # byte arithmetic.
    weight_bytes: float


@dataclass
class SolveResult:
    """The policy outcome. `status` is the single source of truth for the
    regime / feasibility:

        "ok" | "compute-bound: …" | "load-bound: …" | "infeasible (neg-mem): …"

    `feasible` is derived from it (a curve was produced ⇔ not infeasible).
    """

    status:  str                          # regime / feasibility, see above
    alpha:   np.ndarray                   # shape (L,) — α^0..α^{L−1}
    T_ahead:   np.ndarray                   # shape (L,) — T_ahead^0..T_ahead^{L−1}
    inputs:  SolverInputs = field(repr=False)

    @property
    def feasible(self) -> bool:
        """A usable α curve was produced (not neg-mem)."""
        return not self.status.startswith("infeasible")


# ---------------------------------------------------------- cost constants


def _cost_constants(inp: SolverInputs) -> Tuple[float, float, float, float]:
    """(C1, C2, C3, C4) of the α-linear cost model.

    C1/C2 are device time per layer visit: the reload plus the save. See the
    module docstring for why the save belongs in T_NVMe. The four constants
    have no paper names; they expand Eqs. 15–16.
    """
    C1 = inp.t_w + inp.n_seq * (inp.t_kv + inp.t_save_kv)
    C2 = 0.5 * inp.n_seq * (inp.t_kv + inp.t_save_kv)
    C3 = inp.n_seq * inp.T_fwd
    C4 = inp.n_seq * inp.T_kv_gen
    return C1, C2, C3, C4


def _c_floor(inp: SolverInputs) -> float:
    """C_floor = t_w + t_act = t_w + C2_read/N_seq — the lower bound of the
    ahead time: the bytes (weight + one sequence's activations) that must be
    resident before a layer computes. Sourced from the read time alone
    (C2_read = ½·N_seq·t_kv = N_seq·t_act), deliberately not from the
    device-time C2: C_floor is about the bytes that must have landed before
    a layer can start, not about what the device is busy doing. A save
    occupies the drive but no slot, so folding it in would inflate a memory
    bound with device time."""
    C2r = 0.5 * inp.n_seq * inp.t_kv
    return inp.t_w + C2r / inp.n_seq


def _m_slot(inp: SolverInputs) -> float:
    """M_slot (Eq. 8), one slot in bytes: M_W, the layer's weights; M_KV, its
    whole-batch K/V at the decode-peak length S+gen_len; and M_A, its
    activation buffer at the α=1 worst case (half the KV — the hidden is
    d_model per token against KV's 2·d_model), so M_slot = weight + 1.5·kv.
    The activation buffer is sized for α=1 so it is a sufficient buffer for
    any curve and for the prefill's residual stream; the runtime allocates
    exactly this (`StreamingModelLoader._act_buffers_bytes`). M_KV is
    `_m_kv`."""
    return inp.weight_bytes + 1.5 * _m_kv(inp)


def _m_kv(inp: SolverInputs) -> float:
    """M_KV (Eq. 7), one slot's K and V buffers in bytes: the whole batch's
    K and V at the decode-peak length S+gen_len, the KV each sequence grows
    to over the decode, reserved in whole blocks:

        M_KV = 2 · bytes · N_seq · ⌈(S+gen_len)/block_size⌉ · block_size · d_model

    (fp16: 2·bytes = 4). vLLM's null block is not part of it."""
    blocks = -(-(inp.S + inp.gen_len) // inp.block_size)
    return 4.0 * inp.n_seq * blocks * inp.block_size * inp.d_model


def _num_slots(inp: SolverInputs) -> int:
    """n_slots = N_slots (Eq. 9) — the number of rotating layer slots (each
    with a weight, K, V and activation buffer, a single coupled depth) grown
    to the slot memory budget, floored to an integer layer count
    (≤ num_layers = L). Derived from the budget in bytes:

        M_slot  = M_W + M_KV + M_A = weight + 1.5·KV   (`_m_slot`)
        n_slots = min(⌊slots_mem / M_slot⌋, num_layers)

    The slot set is a memory capacity, so it is counted in bytes; neg-mem can
    only come from n_slots < 1 (one slot does not fit).

    A time budget (⌊(slots_mem/BW)/C1⌋) would be wrong here: C1 mixes the
    weight read with KV terms priced at other rates (and with write time), so
    C1·BW is not M_slot's bytes; where it understates them, n_slots
    exceeds what the byte budget holds and the slots do not fit in memory at
    runtime. t_kv is deliberately unused here — it is a time for the
    cost model, not a memory size.

    Returns 0 when the slot memory budget cannot hold one layer's working
    set.
    """
    chunk = _m_slot(inp)
    slots_mem = inp.slots_mem_bytes
    return min(int(slots_mem // chunk), inp.num_layers)


def _c_mem(inp: SolverInputs) -> float:
    """C_mem — the memory upper bound on the ahead time (Eq. 19), in
    load-time seconds:

        C_mem = n_slots · (M_W + M_KV) / BW_NVMe

    What is loaded ahead of the computation occupies the slots, so the ahead
    time can be at most the disk time the slot set holds. A slot's loaded
    bytes are at most M_W + M_KV, up to the blob's per-tensor 4 KiB padding
    of the weights: decode step t reads ⌈(S+t)/block_size⌉ ≤
    ⌈(S+gen_len)/block_size⌉ blocks per sequence, and each activation block
    read into the activation buffer stands in for a KV block of twice its
    size, whose K/V kv_gen recomputes, so a layer at ratio α reads at most
    M_W + (1 − α/2)·M_KV.
    0 when not even one slot fits — the neg-mem regime.

    The unit here is read seconds, deliberately. `BW_NVMe` is the measured
    read rate, while the ahead time this bound limits is accumulated in
    device time (a second of ahead time also buys write service), so a device
    second brings in fewer resident bytes than a read second and the true
    headroom is slightly larger than this returns. That direction is safe for
    a memory bound, and it avoids a second, differently-derived bandwidth.
    `_c_floor` (C_floor) is read-only for a different reason — a save
    occupies the drive but does not occupy a slot.
    """
    return _num_slots(inp) * (inp.weight_bytes + _m_kv(inp)) / inp.BW_NVMe


# -------------------------------------------------------- per-layer ΔT math


def _delta_T(C1: float, C2: float, C3: float, C4: float,
             alpha: float, launch_credit: float) -> float:
    """One layer's ΔT(α) = T_comp(α) − T_NVMe(α), the time gained or lost at
    the layer.

    `launch_credit` (= floor when the C_floor bytes are already resident,
    else 0; code-only, not in Algorithm 3) is dropped from T_NVMe: at
    cold-start or right after a stall the weight + one sequence's
    activations were streamed off the critical path, so ΔT does not pay for
    them.
    """
    t_nvme = (C1 - C2 * alpha) - launch_credit
    t_comp = C3 + C4 * alpha
    return t_comp - t_nvme


def _inverse_alpha(C1: float, C2: float, C3: float, C4: float,
                   dT_target: float, launch_credit: float) -> float:
    """Invert ΔT(α) = dT_target for α (Eq. 25).

    α = (dT_target − ΔT(α=0)) / P, P = ΔT(α=1) − ΔT(α=0), with ΔT(α=0)
    carrying the same `launch_credit` as `dT_target` was computed under.
    """
    P = C2 + C4
    return (dT_target - _delta_T(C1, C2, C3, C4, 0.0, launch_credit)) / P


def _burn_per_layer(C1: float, C2: float, C3: float, C4: float) -> float:
    """The most ahead time one layer can burn: |ΔT| at the burn extreme
    α = 0 (|ΔT|α=0|), with no launch credit."""
    return abs(_delta_T(C1, C2, C3, C4, 0.0, 0.0))


def _remaining_burn(C1: float, C2: float, C3: float, C4: float,
              n: int) -> np.ndarray:
    """Backward look-ahead landing cap remaining_burn^i, the −(L−i)·ΔT|α=0
    term of Eq. 20: the most ahead time the layers after i can still burn off
    down to 0, each at its burn extreme α = 0. With 0-based i it is
    (L−1−i)·|ΔT|α=0| — the absolute value keeps it non-negative also in the
    compute-bound regime, where the paper's term is negative. Exclusive of
    layer i, so remaining_burn^{L−1} = 0 forces an exact-0 landing on the
    last layer. The policy loop (`_per_layer_alpha`), the trajectory replay
    and the plotted ahead-time bounds all read this one array."""
    step = _burn_per_layer(C1, C2, C3, C4)
    remaining_burn = np.zeros(n, dtype=np.float64)
    for i in range(n - 2, -1, -1):
        remaining_burn[i] = remaining_burn[i + 1] + step
    return remaining_burn


# ------------------------------------------------------------ regime label


def _regime_label(inp: SolverInputs) -> Tuple[str, str | None, bool]:
    """Classify the workload. Returns (label, explanation-or-None, feasible).

    Only neg-mem is infeasible; the policy still emits a curve for
    compute-bound / load-bound / ok.
    """
    n = inp.num_layers
    C1, C2, C3, C4 = _cost_constants(inp)
    P = C2 + C4
    Q = C1 - C3
    if _num_slots(inp) < 1:
        return ("neg-mem",
                "one slot (weight + 1.5·KV at S+gen_len) does not fit the "
                "budget", False)
    if Q < 0.0:
        return ("compute-bound",
                f"Q=C1−C3={Q:.3e} < 0 — compute exceeds load even at α=0; "
                f"surplus only ever grows, greedy burns it down", True)
    if (n * Q / P) > float(n):
        return ("load-bound",
                f"balance Σα=L·Q/P={n * Q / P:.3f} > L={n} — loader "
                f"can't keep up even at α≡1; greedy takes the stalls", True)
    return ("ok", None, True)


# --------------------------------------------- hybrid policy (Algorithm 3)


def _per_layer_alpha(C1: float, C2: float, C3: float, C4: float,
                  n: int, C_mem: float, floor: float,
                  beta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """The hybrid policy (Algorithm 3) over the ahead time T_ahead.
    Pure in its scalar cost constants.

    Each layer places the target line at `beta` of the way up its ahead-time
    bounds [C_floor^i, C_burn^i] and picks the α that lands the ahead time
    on it:

        T_target^i = c_floor_i + beta·(C_burn − c_floor_i)   (Eq. 22)
        ΔT_target  = T_target^i − T_ahead     (ΔT needed to get there)

    (beta = ½ puts the target halfway between the two bounds.)

    ΔT_target is then clipped into the layer's range [ΔT(α=0), ΔT(α=1)]
    (= [dT_alpha0, dT_alpha1], Eq. 24). That clip is the saturation rule:
    when the whole range sits below the target (can't bank enough) α → the
    gain edge (max ΔT); when it sits above (can't burn enough) α → the burn
    edge (min ΔT). When the target is reachable the inverse (Eq. 25) is
    exact and α lands strictly inside (0, 1). There is no smoothing and no
    burn/gain blend. Unlike Algorithm 3, the loop also applies the launch
    credit and floors T_ahead at 0 after a stall.

    Landing is automatic: the exclusive backward look-ahead remaining_burn
    drives C_burn (hence T_target) → 0 over the final layers, so the policy
    walks the ahead time down to 0 with no separate end handling. In steady
    state it sits at ΔT_target ≈ 0, i.e. the T_NVMe == T_comp crossover α.

    α is continuous; the runtime quantizes it one-sidedly toward the
    activation form (`runtime.labels.block_is_act`).

    Returns (alpha, T_ahead) where T_ahead^i is the post-update, post-floor
    ahead time.
    """
    # Capping the ahead time at C_burn^i = min(C_mem, remaining_burn^i) keeps
    # the end reachable.
    remaining_burn = _remaining_burn(C1, C2, C3, C4, n)

    alpha = np.zeros(n, dtype=np.float64)
    tahead = np.zeros(n, dtype=np.float64)
    T_ahead = 0.0
    for i in range(n):
        C_burn = min(C_mem, remaining_burn[i])
        c_floor_i = min(floor, C_burn)

        # A normal layer whose ahead time dropped below C_floor stalls; the
        # C_floor bytes (weight + one sequence's activations) streamed in
        # during the wait, so credit them and restart the slack accounting
        # from 0.
        credit = (C_burn > floor) and (T_ahead < floor)
        if credit:
            T_ahead = 0.0
        launch_credit = floor if credit else 0.0

        # Clip range of this layer: ΔT at α = 0 (the min ΔT, burns ahead
        # time) and at α = 1 (the max, banks ahead time).
        dT_alpha0 = _delta_T(C1, C2, C3, C4, 0.0, launch_credit)
        dT_alpha1 = _delta_T(C1, C2, C3, C4, 1.0, launch_credit)

        # T_target = beta (β) up the ahead-time bounds [c_floor_i, C_burn].
        # The ΔT to reach it is (target − T_ahead), clipped into the clip
        # range — the clip saturates α to the burn/gain edge when the target
        # is out of reach this layer, and lands the ahead time exactly on it
        # when it is not.
        target = c_floor_i + beta * (C_burn - c_floor_i)
        dT = min(dT_alpha1, max(dT_alpha0, target - T_ahead))

        # Invert the (already clipped) ΔT_target to α ∈ [0, 1].
        # The outer clamp only wipes IEEE rounding dust.
        alpha[i] = min(1.0, max(0.0,
                       _inverse_alpha(C1, C2, C3, C4, dT, launch_credit)))
        T_ahead = max(0.0, T_ahead + dT)               # floor at 0 on a stall
        tahead[i] = T_ahead

    return alpha, tahead


# ---------------------------------------------------------- solve / verify


def solve(inp: SolverInputs) -> SolveResult:
    """α curve from the hybrid policy (Algorithm 3)."""
    n = inp.num_layers
    if n < 2:
        raise ValueError(f"num_layers must be ≥ 2, got {n}")
    C1, C2, C3, C4 = _cost_constants(inp)
    regime, reason, feasible = _regime_label(inp)

    if not feasible:                               # neg-mem
        return SolveResult(
            status=f"infeasible ({regime}): {reason}",
            alpha=np.full(n, np.nan),
            T_ahead=np.full(n, np.nan),
            inputs=inp,
        )

    C_mem = _c_mem(inp)
    floor = _c_floor(inp)
    alpha, T_ahead = _per_layer_alpha(C1, C2, C3, C4, n, C_mem, floor,
                                   beta=inp.beta)

    status = "ok" if regime == "ok" else f"{regime}: {reason}"
    return SolveResult(status=status, alpha=alpha, T_ahead=T_ahead, inputs=inp)


def _replay_trajectory(alpha: np.ndarray,
                       inp: SolverInputs) -> Tuple[np.ndarray, np.ndarray]:
    """Replay the policy's T_ahead accounting for a given α curve,
    independently of solve: the exclusive landing cap remaining_burn, the
    per-layer launch credit (cold-start / post-stall), and the floor at 0.
    Returns (dT, T_ahead) where dT^i is the credited per-layer ΔT actually
    applied and T_ahead^i is the post-floor ahead time. Shared by
    compute_T_ahead / compute_delta_T / verify.
    """
    n = inp.num_layers
    if alpha.shape != (n,):
        raise ValueError(f"alpha shape {alpha.shape} != (n={n},)")
    C1, C2, C3, C4 = _cost_constants(inp)
    floor = _c_floor(inp)
    C_mem = _c_mem(inp)
    remaining_burn = _remaining_burn(C1, C2, C3, C4, n)

    dT = np.empty(n, dtype=np.float64)
    tahead = np.empty(n, dtype=np.float64)
    T_ahead = 0.0
    for i in range(n):
        C_burn = min(C_mem, remaining_burn[i])
        credit = (C_burn > floor) and (T_ahead < floor)
        if credit:
            T_ahead = 0.0
        launch_credit = floor if credit else 0.0
        d = _delta_T(C1, C2, C3, C4, float(alpha[i]), launch_credit)
        dT[i] = d
        T_ahead = max(0.0, T_ahead + d)
        tahead[i] = T_ahead
    return dT, tahead


def compute_T_ahead(alpha: np.ndarray, inp: SolverInputs) -> np.ndarray:
    """Ahead-time (T_ahead) trajectory for a given α curve. Used by `verify` to
    cross-check the stored `T_ahead`."""
    return _replay_trajectory(alpha, inp)[1]


def compute_delta_T(alpha: np.ndarray, inp: SolverInputs) -> np.ndarray:
    """Per-layer ΔT^i (the time gained or lost at layer i) actually applied
    along the policy's T_ahead trajectory (with the same launch-credit
    decisions as T_ahead). This is what T_ahead integrates, not
    T_ahead[i] − T_ahead[i−1], which a credit reset or a 0-floor stall would
    distort. Handy for plotting the per-layer ΔT.
    """
    return _replay_trajectory(alpha, inp)[0]


def ahead_time_bounds(inp: SolverInputs) -> Tuple[np.ndarray, np.ndarray]:
    """Per-layer ahead-time bounds the policy keeps T_ahead^i inside:
    (lower, upper).

        upper^i = C_burn^i = min(C_mem, remaining_burn^i)   (Eq. 21) — the
                  C_mem-clamped landing cap. Early layers remaining_burn^i ≫
                  C_mem so upper^i = C_mem; near the end the exclusive
                  remaining_burn drops upper^i toward 0.
        lower^i = C_floor^i = min(floor, upper^i)   (Eq. 23) — it too drops
                  to 0 on the last layer so the curve lands at 0.

    These are the bound lines for the T_ahead plot (clamped at C_mem, so the
    huge early remaining_burn does not blow up the y-range).
    """
    C1, C2, C3, C4 = _cost_constants(inp)
    floor = _c_floor(inp)
    C_mem = _c_mem(inp)
    remaining_burn = _remaining_burn(C1, C2, C3, C4, inp.num_layers)
    upper = np.minimum(C_mem, remaining_burn)
    lower = np.minimum(floor, upper)
    return lower, upper


def compute_T_target(inp: SolverInputs) -> np.ndarray:
    """Per-layer target line T_target^i (Eq. 22) the policy aims T_ahead at:

        T_target^i = lower^i + β·(upper^i − lower^i)

    where (lower, upper) = ahead_time_bounds and β = inp.beta (0.5 = halfway).
    This is the line the policy lands the ahead time on when it is reachable
    within the clip range; it is drawn on the T_ahead plot as the aim between
    the two bounds.
    """
    lower, upper = ahead_time_bounds(inp)
    return lower + inp.beta * (upper - lower)


def verify(result: SolveResult, atol: float = 1e-6) -> dict:
    """Post-solve checks for the policy's curve. Recomputes the T_ahead
    trajectory independently and asserts: α ∈ [0, 1]; T_ahead never
    exceeds C_mem; T_ahead lands near 0 by the end (it may touch the 0
    floor mid-trajectory — that is an accepted stall, not a failure).
    """
    if not result.feasible:
        return {"feasible": False}
    inp = result.inputs
    C1, C2, C3, C4 = _cost_constants(inp)
    alpha = result.alpha
    T_ahead = result.T_ahead
    C_mem = _c_mem(inp)
    regime, _, _ = _regime_label(inp)

    bounds_ok = bool(alpha.min() >= -atol and alpha.max() <= 1.0 + atol)

    rec = compute_T_ahead(alpha, inp)
    traj_match = bool(np.allclose(rec, T_ahead, atol=1e-9))

    memory_ok = bool((T_ahead <= C_mem + atol).all()
                     and T_ahead.min() >= -atol)

    # "Near 0" = within one last-layer max-burn (no credit) above the floor.
    # The exclusive remaining_burn drives C_burn → 0 on the last layer; a
    # clean curve ends in [0, ~burn]. Only a curve under-burned over its last
    # layers ends higher than one burn.
    landing_tol = _burn_per_layer(C1, C2, C3, C4) + atol
    end_ahead_ok = bool(-atol <= T_ahead[-1] <= landing_tol)

    # `curve_valid` is the regime-independent "the policy produced a usable
    # α curve" signal (bounds + self-consistent trajectory). `memory_ok` /
    # `end_ahead_ok` are extra guarantees the ok/load-bound regimes meet but
    # compute-bound cannot: there the ahead time provably exceeds C_mem and
    # never lands (Q<0 ⇒ ΔT>0 for every α, no α can shrink it). That overflow
    # is a `memory_warning`, not a policy failure: at runtime the loader
    # acquires only a slot the writeback has released, so the slot rotation
    # bounds residency.
    curve_valid = bool(bounds_ok and traj_match)
    memory_warning = bool(regime == "compute-bound" and not memory_ok)

    return {
        "feasible":         True,
        "regime":           regime,
        "curve_valid":      curve_valid,
        "memory_warning":   memory_warning,
        "bounds_ok":        bounds_ok,
        "traj_match":       traj_match,
        "memory_ok":        memory_ok,
        "end_ahead_ok":       end_ahead_ok,
        "ahead_min":          float(T_ahead.min()),
        "ahead_max":          float(T_ahead.max()),
        "mem_cap":          float(C_mem),
        "landing_tol":      float(landing_tol),
    }
