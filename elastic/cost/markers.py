"""Measurement-shape markers: what a units record was measured ON.

A units record prices a shape, not just a machine: a record measured on one
shape returns a plausible number for a run of another. So the shape is
stamped into the record's provenance and the solve refuses a record that
does not carry the current one; re-measuring is the only remedy, there is no
override flag.

Kept in a module of its own so that both ends — the micro that stamps it
(`elastic.cost.measure_units`) and the check that enforces it
(`elastic.policy.from_units`) — read the same string without the policy side
importing torch.
"""

# Where a kv_gen launch's K/V is stored. The K and V buffers are filled in
# load (= file offset) order, so a launch's blocks are one consecutive run
# and the kernel is given the base block.
KVGEN_STORE_SHAPE = "consecutive-base-v1"

# The kv_gen launch size, in tokens (`ELASTIC_LOAD_CHUNK_TOKENS`). It moves
# the per-token kv_gen cost the micro measures, so a record priced at one
# launch size does not price a run at another. The runtime's default is kept
# here, next to the check, so the package default (`elastic.vllm_runtime`)
# and the check read one number.
LOAD_CHUNK_TOKENS_DEFAULT = 2048


def run_kv_gen_tokens() -> int:
    """The kv_gen launch size a run in this environment uses."""
    import os
    return int(os.environ.get("ELASTIC_LOAD_CHUNK_TOKENS",
                              LOAD_CHUNK_TOKENS_DEFAULT))


# What the forward units (T_attn_per_token_s / T_dense_per_token_s) were
# measured on: one LSE partial attention call of ELASTIC_ARRIVAL_DRAIN_TOKENS
# tokens through `arrival_decode`, and the dense ops over the same tokens,
# per token, with no batch.
LAYER_STEP_SHAPE = "lse-call-per-token-v1"

# The partial attention call volume, in tokens
# (`ELASTIC_ARRIVAL_DRAIN_TOKENS`): the unit T_attn_per_token_s is measured
# on one call of this volume, so a record priced at one call volume does not
# price a run at another. The runtime's default lives here, next to the
# check, like the kv_gen launch size's.
ARRIVAL_DRAIN_TOKENS_DEFAULT = 32768


def run_partial_attn_tokens() -> int:
    """The partial attention call volume a run in this environment attends
    in."""
    import os
    return int(os.environ.get("ELASTIC_ARRIVAL_DRAIN_TOKENS",
                              ARRIVAL_DRAIN_TOKENS_DEFAULT))
