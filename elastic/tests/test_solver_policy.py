"""Policy unit tests — the non-linear hybrid policy (Algorithm 3)
over α^0..α^{L−1}.

The cost primitives here are synthetic and chosen so that each test lands in
the regime it means to exercise. Nothing is mocked: the solver runs for real
on inputs picked to make a particular regime reachable, because the regimes
that matter (compute-bound, load-bound, memory-infeasible) are otherwise hard
to hit from measured numbers on demand.

What the policy must guarantee:

  * α^i ∈ [0, 1] for every layer
  * the ahead time T_ahead never exceeds C_mem
  * T_ahead lands near 0 by the last layer — it may touch the floor at 0
    part-way through, which is an accepted stall, not a failure
  * it always emits a curve, except when the slot set cannot hold one
    layer's working set

The distinction in that last point is the one to preserve: a hard workload
still produces a usable curve with a warning, and only a genuinely unsolvable
input reports infeasible. A caller filtering on the success flag has to keep
the warned cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from elastic.policy.solver import (
    SolverInputs, _cost_constants, _delta_T, _inverse_alpha,
    _c_floor, _c_mem, _num_slots,
    compute_T_ahead, solve, verify,
)
from elastic.policy.sanity import check
from elastic.runtime.labels import block_is_act


def _feasible_inputs(*, num_layers: int = 40) -> SolverInputs:
    """The comfortable regime: the loader keeps up with time to spare, and
    the memory budget is large relative to the model, so the slot count
    clamps at num_layers and C_mem is comfortably positive.
    Baseline for the tests that then perturb one quantity to reach another
    regime. The weight bytes are the ones t_w reads at BW_NVMe."""
    return SolverInputs(
        t_w=1e-4, t_kv=1e-6, T_fwd=1e-6, T_kv_gen=2e-5,
        t_save_kv=1e-7,
        n_seq=64, S=128, d_model=768, num_layers=num_layers, gen_len=0,
        block_size=16,
        slots_mem_bytes=96.0 * 2**30, beta=0.5,
        BW_NVMe=7.0e9, weight_bytes=1e-4 * 7.0e9,
    )


def _big_inputs() -> SolverInputs:
    """Large model and large batch, still in the comfortable regime.

    The primitives are kept self-consistent with the byte-based slot
    sizing: the KV-load time is about one layer's KV working set divided by
    bandwidth, so the slot set is genuinely memory-bound (fewer slots than
    layers) while C_mem stays positive, and compute is sized so
    the loader still keeps up. Inconsistent primitives would produce a slot
    count that no real cell could exhibit, and the tests below would be
    exercising an impossible configuration."""
    return SolverInputs(
        t_w=0.3, t_kv=3.9e-3, T_fwd=3.9e-3, T_kv_gen=1.95e-3,
        t_save_kv=6.1e-5,
        n_seq=256, S=1024, d_model=9216, num_layers=64, gen_len=0,
        block_size=16,
        slots_mem_bytes=96.0 * 2**30, beta=0.5,
        BW_NVMe=7.0e9, weight_bytes=0.3 * 7.0e9,
    )


# ---------------------------------------------------------- cost constants


def test_cost_constants_match_spec():
    inp = _feasible_inputs()
    C1, C2, C3, C4 = _cost_constants(inp)
    assert C1 == pytest.approx(
        inp.t_w + inp.n_seq * (inp.t_kv + inp.t_save_kv))
    assert C2 == pytest.approx(
        0.5 * inp.n_seq * (inp.t_kv + inp.t_save_kv))
    assert C3 == pytest.approx(inp.n_seq * inp.T_fwd)
    assert C4 == pytest.approx(inp.n_seq * inp.T_kv_gen)


def test_write_term_folds_in_with_the_read_term_and_only_there():
    """The save is device time in T_NVMe, and an activation block halves it
    exactly as it halves the reload — so it enters C1/C2 with the same α
    dependence and nowhere else.

    The quantity it must not enter is the lower bound C_floor: it is about
    what has to be resident, and a save occupies the drive but no slot, so
    `_c_floor` counts the read time alone — folding the write in would
    inflate a memory bound with device time.
    """
    inp = _feasible_inputs()
    dry = SolverInputs(**{**inp.__dict__, "t_save_kv": 0.0})
    C1, C2, _, _ = _cost_constants(inp)
    C1d, C2d, _, _ = _cost_constants(dry)
    assert C1 - C1d == pytest.approx(inp.n_seq * inp.t_save_kv)
    assert C2 - C2d == pytest.approx(0.5 * inp.n_seq * inp.t_save_kv)
    # α=1 saves half the write, same as it saves half the reload
    t_nvme = lambda c1, c2, a: c1 - c2 * a
    assert (t_nvme(C1, C2, 1.0) - t_nvme(C1d, C2d, 1.0)
            == pytest.approx(0.5 * inp.n_seq * inp.t_save_kv))
    # memory-derived quantities are untouched
    assert _c_floor(inp) == pytest.approx(_c_floor(dry))
    assert (_c_mem(inp)
            == pytest.approx(_c_mem(dry)))


def test_pricing_the_save_pushes_alpha_up():
    """The direction the write term exists to express: a save that costs
    device time makes T_NVMe dearer, and the only lever that shortens both
    the reads and the writes is α (an activation block is half a KV block on
    the way out as well as in). So a solve that prices the save must not
    choose fewer activation blocks than one that prices it at zero."""
    inp = _big_inputs()
    dry = SolverInputs(**{**inp.__dict__, "t_save_kv": 0.0})
    wet = SolverInputs(**{**inp.__dict__,
                          "t_save_kv": 4.0 * inp.t_save_kv})
    a_dry = solve(dry).alpha.mean()
    a_wet = solve(wet).alpha.mean()
    assert a_wet >= a_dry - 1e-12


# --------------------------------------- (a) feasible: curve + zero landing


@pytest.mark.parametrize("num_layers", [40, 48, 64, 96])
def test_feasible_yields_curve_and_zero_landing(num_layers):
    """A feasible workload at each common model depth: solve succeeds, the
    curve stays in [0, 1], T_ahead respects C_mem and lands near 0, and
    every invariant the verifier checks holds."""
    inp = _feasible_inputs(num_layers=num_layers)
    res = solve(inp)
    assert res.feasible, f"L={num_layers}: {res.status}"
    assert res.alpha.shape == (num_layers,)
    C_mem = _c_mem(inp)
    assert (res.T_ahead <= C_mem + 1e-9).all()
    assert res.T_ahead.min() >= -1e-9
    # lands near 0 (within one last-layer max-burn of the floor at 0)
    v = verify(res)
    assert res.T_ahead[-1] <= v["landing_tol"], res.T_ahead[-1]
    for k in ("bounds_ok", "memory_ok", "traj_match", "end_ahead_ok"):
        assert v[k], f"{k} failed: {v}"


def test_big_workload_feasible_and_lands():
    """Large model and batch with a genuinely memory-bound slot set: still
    solvable, T_ahead stays under C_mem and lands near 0."""
    inp = _big_inputs()
    assert 1 <= _num_slots(inp) < inp.num_layers   # genuinely memory-bound
    assert _c_mem(inp) > 0.0
    res = solve(inp)
    assert res.feasible, res.status
    assert res.status == "ok", res.status
    v = verify(res)
    assert all(v[k] for k in ("bounds_ok", "memory_ok", "traj_match",
                              "end_ahead_ok")), v


def test_alpha_bounds():
    res = solve(_feasible_inputs())
    assert res.feasible
    assert res.alpha.min() >= -1e-9 and res.alpha.max() <= 1.0 + 1e-9


def test_ahead_time_under_cmem():
    inp = _feasible_inputs()
    res = solve(inp)
    assert res.feasible
    assert (res.T_ahead <= _c_mem(inp) + 1e-9).all()


def test_compute_T_ahead_matches_solve():
    """The stored T_ahead trajectory must be reproducible from the α curve
    alone, under the policy's own conventions (layer-0 weight excluded, floor
    at 0). If it is not, the reported curve and the reported T_ahead
    describe different solutions."""
    inp = _feasible_inputs()
    res = solve(inp)
    assert res.feasible
    np.testing.assert_allclose(compute_T_ahead(res.alpha, inp), res.T_ahead,
                               atol=1e-12)


def test_verify_recomputes_trajectory():
    res = solve(_feasible_inputs())
    v = verify(res)
    assert v["traj_match"]


# ------ (b) neg-mem only from N_slots < 1; byte-sized slots never overflow


def _m_slot(inp):
    """M_slot, the per-layer bytes the slot set is sized from — a local
    mirror of the solver's own computation:

        weight (M_W)
      + one layer's decode-peak KV (M_KV), priced at S+gen_len
      + the activation buffer (M_A), 0.5·kv, which is the α=1 worst case

    Duplicated here on purpose: a test that called the solver's helper would
    pass even if that helper changed, so it would stop pinning anything."""
    kv = 4.0 * inp.n_seq * _whole_blocks(inp) * inp.d_model
    return inp.weight_bytes + kv + 0.5 * kv


def _whole_blocks(inp):
    """S+gen_len rounded up to whole blocks, in tokens (Eq. 7)."""
    return math.ceil((inp.S + inp.gen_len) / inp.block_size) * inp.block_size


def test_byte_slot_count_keeps_cmem_positive():
    """A byte-consistent slot count never drives C_mem to zero or below.

    C_mem = N_slots·(M_W + M_KV)/BW_NVMe is positive for every N_slots ≥ 1,
    and N_slots·M_slot ≤ slots_mem by construction, so an infeasible-memory
    verdict can only come from N_slots < 1 — never from the slot set
    overflowing the budget it was derived from.

    The byte derivation matters for the budget, not for C_mem's sign: a
    time-derived count mixes the weight bandwidth with the KV-load primitive,
    which runs at a different effective rate, and can inflate N_slots past
    what the byte budget allows, so the slots would not fit in memory at
    runtime."""
    inp = _big_inputs()                                  # genuinely memory-bound
    saw_slot = False
    for frac in (1.0, 0.5, 0.25, 0.12, 0.06, 0.03):
        inp.slots_mem_bytes = frac * (96.0 * 2**30)
        n_slots = _num_slots(inp)
        if n_slots >= 1:
            saw_slot = True
            assert _c_mem(inp) > 0.0, (frac, n_slots)
            assert solve(inp).feasible, (frac, n_slots)
    assert saw_slot                                    # the sweep exercised n_slots ≥ 1


def test_neg_mem_when_budget_holds_under_one_layer():
    """A budget below one M_slot gives zero slots, which is the
    only path to an infeasible-memory verdict (see the test above). The
    budget is set directly — nothing scales it."""
    inp = _feasible_inputs()
    # slots_mem = half an M_slot ⇒ ⌊slots_mem / M_slot⌋ = 0.
    inp.slots_mem_bytes = 0.5 * _m_slot(inp)
    assert _num_slots(inp) < 1
    res = solve(inp)
    assert not res.feasible
    assert "neg-mem" in res.status.lower(), res.status


# ----------------------------------- (c) launch credit (cold-start / stall)


def test_launch_credit_lowers_load_and_inverts_exactly():
    """A credited layer — one starting cold or resuming after a stall — drops
    C_floor from its T_NVMe, so its ΔT is exactly `floor` higher than the
    same α uncredited.

    `_inverse_alpha` is the exact inverse of `_delta_T` under the same credit.
    That exactness keeps the policy's α inside [0, 1] without clipping, and a
    clip would change the curve the solver claims to have produced."""
    inp = _feasible_inputs()
    C1, C2, C3, C4 = _cost_constants(inp)
    # = t_w + t_act + t_save_act (any credit value works here)
    floor = inp.t_w + C2 / inp.n_seq
    a = 0.4
    dT_plain = _delta_T(C1, C2, C3, C4, a, 0.0)
    dT_credit = _delta_T(C1, C2, C3, C4, a, floor)
    assert dT_credit == pytest.approx(dT_plain + floor)
    # exact inverse under each credit
    assert _inverse_alpha(C1, C2, C3, C4, dT_plain, 0.0) == pytest.approx(a)
    assert _inverse_alpha(C1, C2, C3, C4, dT_credit, floor) == pytest.approx(a)


def test_cold_start_layer_is_credited():
    """Layer 0 starts with T_ahead = 0 in a normal regime, so it is credited
    C_floor: its recomputed ΔT matches `_delta_T` with the credit and is
    distinguishable from the uncredited value."""
    inp = _feasible_inputs()
    C1, C2, C3, C4 = _cost_constants(inp)
    floor = _c_floor(inp)
    res = solve(inp)
    a0 = float(res.alpha[0])
    credited = _delta_T(C1, C2, C3, C4, a0, floor)
    uncredited = _delta_T(C1, C2, C3, C4, a0, 0.0)
    # res.T_ahead[0] == max(0, credited) and differs from the uncredited ΔT
    assert res.T_ahead[0] == pytest.approx(max(0.0, credited))
    assert not np.isclose(res.T_ahead[0], max(0.0, uncredited))


# ------------------------------------ compute-bound / load-bound regimes


def test_compute_bound_feasible_with_status():
    """T_comp exceeds T_NVMe even at α=0, so T_ahead only grows and no α
    can shrink it — α adds kv_gen compute, it never removes any.

    The policy does not fail here: it emits a valid [0,1] curve and flags the
    regime. The projected T_ahead does overflow C_mem, and that is honest
    rather than a bug — at run time the loader acquires only a slot the
    writeback has released, which the projection does not model — so
    memory_ok is expected to be False while the result stays usable."""
    inp = _feasible_inputs()
    inp.T_fwd = 0.01      # C3 ≫ C1 ⇒ Q < 0
    res = solve(inp)
    assert res.feasible
    assert "compute-bound" in res.status.lower(), res.status
    assert res.alpha.min() >= -1e-9 and res.alpha.max() <= 1.0 + 1e-9
    v = verify(res)
    assert v["curve_valid"]                 # bounds + trajectory still valid
    assert v["ahead_max"] > v["mem_cap"]      # T_ahead grows past C_mem
    assert v["memory_warning"]              # flagged as warning, not failure
    assert not v["memory_ok"]
    # Tiering matters as much as the curve: compute-bound is a success with a
    # memory warning, not an infeasible result. A caller that keys off the
    # feasible flag has to keep this case, or every compute-bound cell drops
    # out of a sweep.
    rep = check(res)
    assert "tier=warn-memory" in rep.lines[0]


def test_status_tiers_distinguish_outcomes():
    """sanity.check reports the three caller-visible outcomes distinctly."""
    ok_res = solve(_feasible_inputs())                     # ok regime
    cb = _feasible_inputs(); cb.T_fwd = 0.01               # compute-bound
    nm = _feasible_inputs()
    nm.slots_mem_bytes = (nm.n_seq * nm.S * nm.d_model)         # < one M_slot
    assert ok_res.feasible and "tier=ok" in check(ok_res).lines[0]
    cb_res = solve(cb)
    assert cb_res.feasible and "tier=warn-memory" in check(cb_res).lines[0]
    nm_res = solve(nm)
    assert not nm_res.feasible
    assert check(nm_res).lines[0].startswith("infeasible")


def test_load_bound_feasible_with_status():
    """A weight load so large that the loader cannot keep up even at α≡1.
    The policy emits a curve — accepting the stalls — and flags the regime.
    It does not collapse to a flat α≡1: the layers still differ in how much
    each one can recover, and a flat curve throws that away."""
    inp = _feasible_inputs()
    inp.t_w = 1.0
    res = solve(inp)
    assert res.feasible
    assert "load-bound" in res.status.lower(), res.status
    assert res.alpha.min() >= -1e-9 and res.alpha.max() <= 1.0 + 1e-9


# ------------------------------------------------------- slot derivation


def test_slot_count_floored_from_budget():
    """The slot count is ⌊budget / M_slot⌋, clamped at num_layers, where
    M_slot is weight + per-layer decode-peak KV + the activation buffer (see
    `_m_slot`). Nothing scales the budget on the way in."""
    inp = _feasible_inputs()
    n_slots = min(int(inp.slots_mem_bytes // _m_slot(inp)), inp.num_layers)
    assert _num_slots(inp) == n_slots


def test_slot_count_clamps_to_num_layers_on_big_budget():
    """A large budget on a small model admits far more slots than there are
    layers, so the slot count must clamp at num_layers rather than growing
    past the model."""
    inp = _feasible_inputs()
    assert _num_slots(inp) == inp.num_layers


def test_c_mem_shrinks_with_budget():
    """A smaller budget must never yield a larger C_mem on T_ahead — the
    relation has to be monotone, or a tighter machine could be handed a more
    aggressive schedule than a roomier one."""
    caps = []
    for gib in (0.01, 0.1, 1.0, 128.0):
        inp = _feasible_inputs()
        inp.slots_mem_bytes = gib * 2**30
        caps.append(_c_mem(inp))
    assert caps == sorted(caps)   # non-decreasing in the budget


# ---------------------------------------------------------- block form


def _n_act(alpha: float, c: int) -> int:
    return sum(block_is_act(k, alpha) for k in range(c))


def test_block_is_act_ceiling_rule():
    """The runtime form rule must satisfy the counting property exactly: the
    first c blocks (k < c) contain ⌈c·α⌉ activation blocks, for every c. That
    is what makes the realised NVMe traffic match what the solver priced."""
    for alpha in (0.0, 0.1, 0.3333333, 0.5, 0.7777, 1.0):
        for c in (1, 17, 100, 200):
            assert _n_act(alpha, c) == int(np.ceil(c * alpha))


def test_block_is_act_edges():
    assert _n_act(0.0, 16) == 0
    assert _n_act(1.0, 16) == 16


def test_block_is_act_takes_the_policy_curve():
    """The solved curve feeds straight into the form rule, with no rounding
    or adjustment in between — the counting property must hold for the α
    values the policy actually emits, not only for round ones."""
    res = solve(_feasible_inputs())
    assert res.feasible
    for a in res.alpha:
        assert _n_act(float(a), 128) == int(np.ceil(128 * float(a)))


# ------------------------------------ activation buffer + gen_len sizing


def test_activation_buffer_folded_into_the_slot():
    """The α=1 activation buffer (0.5·kv per layer) is part of the slot: the
    slot count is closed-form over M_slot = weight + 1.5·kv, and the fold
    never buys slots relative to M_W + M_KV alone — slots are paid for with
    a guaranteed activation buffer."""
    inp = _big_inputs()
    n0 = _num_slots(inp)
    assert 3 <= n0 < inp.num_layers                      # genuinely budget-bound
    kv = 4.0 * inp.n_seq * _whole_blocks(inp) * inp.d_model
    n_no_reload = min(int(inp.slots_mem_bytes // (_m_slot(inp) - 0.5 * kv)),
                      inp.num_layers)
    assert n0 <= n_no_reload


def test_c_mem_is_eq_19():
    """C_mem = N_slots·(M_W + M_KV) / BW_NVMe (Eq. 19): what is loaded ahead
    sits in the slots, and a slot's loaded bytes are at most its weights and
    its K and V — each activation block read into the activation buffer stands
    in for a KV block of twice its size, so M_A adds nothing beyond
    M_W + M_KV."""
    inp = _big_inputs()
    kv = 4.0 * inp.n_seq * _whole_blocks(inp) * inp.d_model
    want = _num_slots(inp) * (inp.weight_bytes + kv) / inp.BW_NVMe
    assert _c_mem(inp) == pytest.approx(want)
    assert 0.0 < _c_mem(inp) <= inp.slots_mem_bytes / inp.BW_NVMe


def test_m_kv_is_reserved_in_whole_blocks():
    """M_KV (Eq. 7) reserves ⌈(S+gen_len)/block_size⌉ whole blocks per
    sequence: a context one token past a block boundary is charged a whole
    extra block, in the slot count and in C_mem alike."""
    inp = _big_inputs()
    inp.S = 1025                                       # 64 blocks + 1 token
    kv = 4.0 * inp.n_seq * 1040 * inp.d_model          # 65 whole blocks
    assert _num_slots(inp) == min(
        int(inp.slots_mem_bytes // (inp.weight_bytes + 1.5 * kv)),
        inp.num_layers)
    assert _c_mem(inp) == pytest.approx(
        _num_slots(inp) * (inp.weight_bytes + kv) / inp.BW_NVMe)


def test_gen_len_sizes_slots_at_decode_peak():
    """gen_len enters M_slot as S+gen_len, so the slots are sized for the
    decode-peak context rather than the prompt length — a slot set that fits
    at the start of generation but not at the end is not one that fits.

    This affects slot sizing only; α pricing takes its context from the
    primitives. A doubled context therefore shrinks a budget-bound slot
    set."""
    from dataclasses import replace
    inp = _big_inputs()
    n0 = _num_slots(inp)
    assert n0 < inp.num_layers
    inp2 = replace(inp, gen_len=inp.S)       # S+gen = 2S
    n2 = _num_slots(inp2)
    assert n2 < n0
    assert n2 == min(int(inp2.slots_mem_bytes // _m_slot(inp2)),
                     inp2.num_layers)


