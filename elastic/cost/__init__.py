"""Cost-primitive measurement (unit times → units_<model>_S<S>.json).

One record per (model, S, gen), no batch, measured on the target machine
under the conditions the runtime creates (same read shape and queue depth,
same buffer residency, same kernels). Primitives:

    read_GBps              NVMe read rate at the runtime's read shape
    write_GBps             NVMe write rate (the drive's post-write-cache-
                           exhaustion floor, entered as a drive constant)
    T_kv_gen_per_token_s   activation → K/V kv_gen per token
    T_attn_per_token_s     one LSE partial attention call, per token
    T_dense_per_token_s    LN, QKV, out proj, FFN, per token

plus the per-layer weight bytes. `elastic.policy.from_units` turns the
record into `SolverInputs` for any number of sequences N_seq.

CLI entry: ``python -m elastic.cost.cli_measure_units --help``.
"""
