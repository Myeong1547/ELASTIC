"""Convert a `cli_measure_units` record into `SolverInputs` — the one place
a cell is priced.

A units record (the `units` block of the JSON) carries the unit costs and the
per-layer weight bytes, measured at prompt S_op and gen_len:

    read_GBps               NVMe read rate at the runtime's read shape
    write_GBps              NVMe write rate (post-cache-exhaustion floor)
    T_kv_gen_per_token_s    activation → K/V kv_gen per token, runtime
                            kv_gen launches
    T_attn_per_token_s      one LSE partial attention call, per token
    T_dense_per_token_s     LN, QKV, out proj, FFN, per token
    weight_bytes            per-layer weight bytes

The solver takes per-sequence, per-layer primitives, all at the run's
maximum length S + gen (α cannot change during a run, so the curve must hold
at the last decode step):

    t_w         = weight_bytes / read
    t_kv        = (S + gen) · kv_bytes_per_token / read      (∝ context)
    t_save_kv   = block_size · kv_bytes_per_token / write    (one block)
    T_fwd       = T_dense_per_token + (S + gen) · T_attn_per_token
    T_kv_gen    = (S + gen) · T_kv_gen_per_token_s

with kv_bytes_per_token = 4·d_model (fp16 K and V). A sequence brings one
token to the dense ops and S + gen keys to the attention. An activation block
halves both the read and the write byte counts, which the solver encodes as
its C2 term, so no activation primitive (t_act, t_save_act) exists.
The record's measured read rate also serves as BW_NVMe, the memory-to-ahead-
time conversion behind the upper bound C_mem.

`read_GBps` / `write_GBps` arguments replace the record's rates for
pricing only (hand-entered drive constants); BW_NVMe keeps the measured
read rate.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..cost.markers import (KVGEN_STORE_SHAPE, LAYER_STEP_SHAPE,
                            run_partial_attn_tokens, run_kv_gen_tokens)
from .solver import SolverInputs


def check_units_record(units: dict, *, S: int, gen_len: int,
                       kv_gen_tokens: int | None = None) -> None:
    """Raise ValueError unless the record prices a run at (S, gen_len) in this
    environment: the cell's context, the layer-step shape, the kv_gen store
    shape and launch size, and the partial attention call volume it was
    measured at (a record missing a field raises KeyError).
    The single acceptance rule: the solve and the campaign's units-cache test
    both call it."""
    if int(S) != int(units["S_op"]) or int(gen_len) != int(units["gen_len"]):
        raise ValueError(
            f"(S, gen)=({S}, {gen_len}) ≠ measured (S_op, gen_len)="
            f"({units['S_op']}, {units['gen_len']}); the layer step must "
            "be re-measured at the cell's context.")
    prov = units["provenance"]
    _lshape = prov["layer_step_shape"]
    if _lshape != LAYER_STEP_SHAPE:
        raise ValueError(
            f"units record's layer step was measured as {_lshape!r}, not "
            f"{LAYER_STEP_SHAPE!r} (one LSE call through arrival_decode and "
            f"the dense ops over its tokens, per token). Re-measure: `python -m "
            f"elastic.cost.cli_measure_units --remeasure`.")
    _shape = prov["kvgen_store_shape"]
    if _shape != KVGEN_STORE_SHAPE:
        raise ValueError(
            f"units record was measured with kvgen_store_shape={_shape!r}, "
            f"not {KVGEN_STORE_SHAPE!r}. Re-measure: `python -m "
            f"elastic.cost.cli_measure_units --remeasure`.")
    _wave = int(prov["kvgen_wave_tokens"])
    _run_kv_gen_tokens = int(kv_gen_tokens) if kv_gen_tokens is not None \
        else run_kv_gen_tokens()
    if _wave != _run_kv_gen_tokens:
        raise ValueError(
            f"units record's T_kv_gen was measured with {_wave}-token "
            f"kv_gen launches, the run uses {_run_kv_gen_tokens} "
            f"(ELASTIC_LOAD_CHUNK_TOKENS). Re-measure at the run's launch size: "
            f"`python -m elastic.cost.cli_measure_units --remeasure`.")
    drain = int(prov["drain_tokens"])
    if drain != run_partial_attn_tokens():
        raise ValueError(
            f"units record's attention call unit was measured at "
            f"{drain}-token drains, the run uses {run_partial_attn_tokens()} "
            f"(ELASTIC_ARRIVAL_DRAIN_TOKENS). Re-measure at the run's drain.")
    if float(units["weight_bytes"]) <= 0.0:
        raise ValueError("units.weight_bytes must be positive")


def units_to_solver_inputs(units: dict, *,
                           n_seq: int, S: int, d_model: int, num_layers: int,
                           gen_len: int,
                           slots_mem_bytes: float,
                           beta: float,
                           block_size: int,
                           read_GBps: float | None = None,
                           write_GBps: float | None = None,
                           kv_gen_tokens: int | None = None,
                           ) -> SolverInputs:
    """Build `SolverInputs` from a units record for `n_seq` sequences.

    `S` and `gen_len` must equal the record's `S_op` and `gen_len`: the
    layer step is measured at that cell's last decode step, and a different
    context needs a new measurement. The record's kv_gen launch size must be
    the run's (`kv_gen_tokens`, default this environment's
    `ELASTIC_LOAD_CHUNK_TOKENS`); a different launch size needs a new
    T_kv_gen."""
    check_units_record(units, S=S, gen_len=gen_len, kv_gen_tokens=kv_gen_tokens)
    w_bytes = float(units["weight_bytes"])
    read_meas = float(units["read_GBps"]) * 1e9
    read = (read_GBps * 1e9) if read_GBps is not None else read_meas
    write = (write_GBps * 1e9) if write_GBps is not None \
        else float(units["write_GBps"]) * 1e9
    kv_tok_b = 4.0 * float(d_model)
    end = float(S + gen_len)
    return SolverInputs(
        t_w=w_bytes / read,
        t_kv=end * kv_tok_b / read,
        T_fwd=(float(units["T_dense_per_token_s"])
                 + end * float(units["T_attn_per_token_s"])),
        T_kv_gen=float(units["T_kv_gen_per_token_s"]) * end,
        t_save_kv=float(block_size) * kv_tok_b / write,
        weight_bytes=w_bytes,
        n_seq=int(n_seq), S=int(S), d_model=int(d_model),
        num_layers=int(num_layers), gen_len=int(gen_len),
        block_size=int(block_size),
        slots_mem_bytes=float(slots_mem_bytes),
        beta=float(beta),
        BW_NVMe=read_meas,
    )


def load_units(path: Path) -> dict:
    """Read a `cli_measure_units` output JSON (the whole document)."""
    return json.loads(Path(path).read_text())
