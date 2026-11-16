"""α-driven block form (activation or KV) — the single canonical rule.

The solver in `elastic.policy.solver` emits one α^i per decoder layer. Given
that α^i and a block's global number k — its ordinal position in the layer's
whole block stream, flattened across all N_seq sequences (the layout's
numbering, see `block_kv_storage.BlockKVLayout`) — the block is an
activation block (recompute path) or a KV block (direct-load path) by

    block k of layer i is ACT  iff  ⌈(k+1)·α^i⌉ > ⌈k·α^i⌉

so among blocks 0..k−1 of a layer's stream, ⌈k·α^i⌉ are activation blocks.
Two consequences the rest of the runtime depends on:

  * The rounding is one-sided — the ceiling rounds towards ACT, the
    cheaper-to-load form — so realised NVMe traffic never exceeds what the
    policy priced. The direction is frozen once a blob has been written,
    because forms fix on-disk offsets.
  * The form is a pure function of (k, α^i), independent of which other
    blocks exist, so blocks appended during decode extend the same stream and
    no block ever changes form.

α is fixed for the lifetime of a run and lies in [0, 1] (the curve is checked
once where it enters, `streaming_loader._clamp_alpha`): it selects each block's
form, the form selects the block's size and hence every later block's file
offset.

The number k is global rather than per-sequence for two reasons. The
one-sided rounding's worst case — the first block of a stream is ACT whenever
α > 0, however small — then costs one block per layer instead of one per
sequence. And a synchronised decode step's N_seq blocks split between the two
forms in the α proportion instead of sharing one form, which would force that
step's load/compute mix to be all-ACT or all-KV. Block region totals are
unaffected: `n_act_blocks`/`n_kv_blocks` over the flattened count are what the
aggregate byte accounting uses.

k is the only block coordinate the runtime carries. Which global numbers a
sequence's blocks occupy is the layout's numbering
(`BlockKVLayout.ks_of_tokens` and its inverses `seq_of_ks` / `j_of_ks` in
`block_kv_storage`): a prefill hands out a sequence's immutable blocks as a
run of consecutive k and its mutable prompt block as one k, and a decode step
hands out one k per sequence."""

from __future__ import annotations

import math


def block_is_act(k: int, alpha: float) -> bool:
    return math.ceil((k + 1) * alpha) > math.ceil(k * alpha)


def n_act_blocks(n: int, alpha: float) -> int:
    return math.ceil(n * alpha)


def n_kv_blocks(n: int, alpha: float) -> int:
    """KV blocks among the first `n`: n − ⌈n·α⌉."""
    return n - n_act_blocks(n, alpha)


def act_rank(k: int, alpha: float) -> int:
    """rank_ACT(k): dense index of activation block `k` among its layer's
    activation blocks = the number of activation blocks before it = ⌈k·α⌉.
    Meaningful only for an activation block k."""
    return n_act_blocks(k, alpha)


def kv_rank(k: int, alpha: float) -> int:
    """rank_KV(k): dense index of KV block `k` among its layer's KV blocks =
    the number of KV blocks before it = k − ⌈k·α⌉. Meaningful only for a KV
    block k.

    Together with `act_rank` this is a bijection: every block k maps to
    exactly one (form, rank), and each form's ranks cover a gapless 0..count
    range. Both are pure functions of (k, α), so blocks appended during decode
    continue the same numbering with no change of form."""
    return k - n_act_blocks(k, alpha)

