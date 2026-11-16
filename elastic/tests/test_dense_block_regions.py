"""Pure-offset unit tests for the [ACT | K | V] block-region layout in
global-number order.

A layer's file space is three block regions, each holding whole blocks in
global-number order: the immutable blocks seq-major, then the mutable blocks
one block index at a time across all sequences. A block's form and its rank
within its block region both come from its global number k — the layer's
whole N_seq-sequence block stream flattened into one ordinal sequence — which is
the only coordinate a block has. That is what lets a single decode step's N_seq
blocks split between activation and KV blocks instead of sharing one form,
and what prices the form rule's one-sided rounding excess once per layer
instead of once per sequence.

These tests pin the geometric invariants the save and reload paths rely on,
none of which any single caller can check for itself:

  * block regions are sized by their own form (over the flattened
    N_seq·max_blocks_per_seq block stream) and tile the layer's file space
    exactly, with no slack;
  * every block's byte range — one for an activation block, K and V for a
    KV block — is disjoint from every other and inside the file;
  * within a step's last N_seq blocks (N_seq adjacent global numbers), the
    activation-block subset — and, separately, the KV-block subset — are
    each rank-contiguous, so a step is at most 3 writes (one per block
    region), though not necessarily exactly 1;
  * a sequence's immutable blocks of one form are consecutive — the
    contiguity a prefill's seq-major source wants;
  * every block region is perfectly tiled by its blocks, whole blocks back
    to back, for any prompt length — an unwritten hole inside a block region
    fragments the file and costs the blob read speed for the rest of its
    life, and neither the bit-exactness gate nor `srcmis` can see it.

They are exhaustive over a deliberately tiny configuration, so a change that
breaks the affine offset form fails here rather than by silently reading the
wrong bytes at scale.
"""

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.labels import (act_rank, block_is_act, kv_rank,
                                    n_act_blocks, n_kv_blocks)


def _mk(alpha, rows=None, **kw):
    """A layout over a per-layer α curve; a scalar `alpha` is the same value
    on every layer."""
    args = dict(
        n_seq=4,
        num_layers=3,
        max_blocks_per_seq=10,
        block_size=16,
        num_heads=4,
        head_dim=8,
        dtype=torch.float16,
    )
    args.update(kw)
    if not isinstance(alpha, list):
        alpha = [float(alpha)] * args["num_layers"]
    L = BlockKVLayout(alpha=alpha, **args)
    if rows is not None:
        L.set_prompt_len(rows)
    return L


def _seq_ks(L, s, n):
    """The global numbers k of sequence `s`'s first `n` blocks, in position
    order: the layout's `ks_of_tokens` at each block's first token."""
    return L.ks_of_tokens(s, np.arange(n, dtype=np.int64) * L.block_size)


def _all_ks(L):
    """Every global number k of the layer's stream, reached through the
    enumeration (so the declared immutable/mutable split is exercised) — and
    checked to be exactly the flattened N_seq·max_blocks_per_seq stream, no more
    and no fewer."""
    ks = np.concatenate([_seq_ks(L, s, L.max_blocks_per_seq)
                         for s in range(L.n_seq)])
    assert sorted(ks.tolist()) == list(
        range(L.n_seq * L.max_blocks_per_seq))
    return ks


def _wave(L, j):
    """The N_seq global numbers a decode step at block index j
    (j >= n_immutable) produces — that step's last N_seq blocks."""
    return [int(L.ks_of_tokens(s, j * L.block_size))
            for s in range(L.n_seq)]


# opt-13b's geometry is the one where a per-token row is not a page
# multiple (hidden 5120 → 10240 B), so a stride that rounds a sequence up to
# whole blocks would leave a non-page-sized gap; opt-6.7b (4096 → 8192 B) is
# a page multiple and could not expose a slack defect at all.
_GEOM_13B = dict(num_heads=40, head_dim=128, max_blocks_per_seq=80, n_seq=6)
_GEOM_67B = dict(num_heads=32, head_dim=128, max_blocks_per_seq=80, n_seq=6)

_ALPHAS = [0.0, 0.5, 0.73, 1.0, [0.0, 0.5, 1.0]]


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_total_bytes_is_exactly_the_flattened_form_sum(alpha):
    """The layout is a pure rearrangement of the flattened N_seq·N-block stream
    (N = max_blocks_per_seq here): it stores neither more nor fewer bytes
    than the global per-form counts imply. Block-region sizing runs
    `n_*_blocks` once over N_seq·N directly rather than once per sequence, so
    the total is exact, with no per-sequence slack."""
    L = _mk(alpha)
    kb = L.block_bytes
    n_seq = L.n_seq
    N = L.max_blocks_per_seq
    bare = sum((2 * n_kv_blocks(N * n_seq, L.alpha_of_layer(ly))
                + n_act_blocks(N * n_seq, L.alpha_of_layer(ly))) * kb
               for ly in range(L.num_layers))
    assert bare == L.total_bytes


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_first_block_rounding_excess_is_ON(alpha):
    """The one-sided (ACT-favouring) rounding's worst case is paid at most
    once per layer (k=0 of the flattened stream), not once per sequence.
    That is what the global number buys."""
    L = _mk(alpha)
    n_seq = L.n_seq
    N = L.max_blocks_per_seq
    for ly in range(L.num_layers):
        al = L.alpha_of_layer(ly)
        exact = N * n_seq * al
        # The layout's own block-region sizes: the ACT block region's buffer
        # blocks end where the KV block region's begin, and the V block
        # region is one K block region further on.
        n_act = L.mem_kv_base(ly) - 1
        n_kv = L.v_delta(ly) // L.block_bytes
        excess_blocks = n_act + n_kv - N * n_seq
        assert excess_blocks == 0            # total block count is exact
        # ACT count only ever overshoots the continuous ideal by < 1 block,
        # regardless of N_seq.
        assert 0 <= n_act - exact < 1 + 1e-9


@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize("rows", [0, 97, 160])
def test_every_block_is_disjoint_and_inside_the_file(alpha, rows):
    L = _mk(alpha, rows=rows)
    kb = L.block_bytes
    spans = []
    for ly in range(L.num_layers):
        al = L.alpha_of_layer(ly)
        for k in _all_ks(L).tolist():
            if block_is_act(k, al):
                offs = [L.act_offset(ly, k)]
            else:
                o = L.k_offset(ly, k)
                offs = [o, o + L.v_delta(ly)]
            for o in offs:
                assert 0 <= o and o + kb <= L.total_bytes, (ly, k)
                spans.append((o, o + kb, (ly, k)))
    spans.sort()
    for (a0, a1, who_a), (b0, _b1, who_b) in zip(spans, spans[1:]):
        assert a1 <= b0, f"{who_a} overlaps {who_b}"


def _block_regions(L, ly):
    """(base, nbytes) of the layer's ACT, K and V block regions, from the
    hoisted tuple and the per-form counts."""
    kb = L.block_bytes
    abase, kbase, vdelta = L.region_bases(ly)
    n = L.n_seq * L.max_blocks_per_seq
    al = L.alpha_of_layer(ly)
    nkv = n_kv_blocks(n, al) * kb
    return ((abase, n_act_blocks(n, al) * kb), (kbase, nkv),
            (kbase + vdelta, nkv))


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_block_regions_tile_the_layer_file_space(alpha):
    """[ACT | K | V] back to back, layer after layer, ending at the total."""
    L = _mk(alpha, rows=97)
    end = 0
    for ly in range(L.num_layers):
        (a0, an), (k0, kn), (v0, vn) = _block_regions(L, ly)
        assert (a0, a0 + an, k0 + kn) == (end, k0, v0)
        assert L.v_delta(ly) == kn           # V follows K block for block
        end = v0 + vn
    assert end == L.total_bytes


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.73, 1.0])
@pytest.mark.parametrize("rows", [0, 97])
def test_last_n_seq_blocks_form_subsets_are_rank_contiguous(alpha, rows):
    """A synchronised decode step's last N_seq blocks (N_seq adjacent global
    numbers) may split between activation and KV blocks, so they are not
    necessarily one contiguous N_seq-wide write, but each form's subset among
    them is rank-contiguous within its own block region, because
    act_rank/kv_rank are monotonic in the global number. So a step is at
    most 3 writes (one per block region), never per-sequence scattered."""
    L = _mk(alpha, rows=rows)
    kb = L.block_bytes
    for ly in range(L.num_layers):
        al = L.alpha_of_layer(ly)
        for j in range(L.n_immutable, L.max_blocks_per_seq):
            wave = _wave(L, j)
            assert sorted(wave) == list(range(min(wave), min(wave) + L.n_seq))
            act_offs = sorted(L.act_offset(ly, k) for k in wave
                              if block_is_act(k, al))
            kv_offs = sorted(L.k_offset(ly, k) for k in wave
                             if not block_is_act(k, al))
            for offs in (act_offs, kv_offs):
                if len(offs) >= 2:
                    assert offs == list(
                        range(offs[0], offs[0] + len(offs) * kb, kb))
            # every sequence lands in exactly one of the two subsets
            assert len(act_offs) + len(kv_offs) == L.n_seq


@pytest.mark.parametrize("alpha,kind", [(1.0, 0), (0.0, 1)])
def test_last_n_seq_blocks_are_n_seq_adjacent_when_alpha_is_saturated(alpha, kind):
    """At the saturation points (α=0 or α=1) every sequence's block among a
    step's last N_seq blocks has the one form that exists, so those blocks are
    exactly N_seq adjacent block positions and one write — the special case of
    the rank-contiguity above."""
    L = _mk(alpha, rows=97)
    kb = L.block_bytes
    f = L.act_offset if kind == 0 else L.k_offset
    for ly in range(L.num_layers):
        for j in range(L.n_immutable, L.max_blocks_per_seq):
            offs = [f(ly, k) for k in _wave(L, j)]
            assert offs == list(range(offs[0], offs[0] + L.n_seq * kb, kb))


def test_a_steps_last_n_seq_blocks_can_have_both_forms():
    """The form is a function of the global number, not of a block's
    position inside its own sequence, so a step's last N_seq blocks are not
    forced to share one form: at α=0.3, N_seq=8 some step's last N_seq blocks
    split."""
    L = _mk(0.3, rows=0, n_seq=8, max_blocks_per_seq=6)
    al = L.alpha_of_layer(0)
    found_split = False
    for j in range(L.n_immutable, L.max_blocks_per_seq):
        labels = {block_is_act(k, al) for k in _wave(L, j)}
        if len(labels) > 1:
            found_split = True
    assert found_split, "expected at least one mixed-label wave at α=0.3, N_seq=8"


@pytest.mark.parametrize("alpha", [0.5, 0.73, 1.0])
def test_immutable_blocks_of_one_sequence_are_consecutive(alpha):
    """The reason immutable blocks are numbered seq-major: a prefill
    produces a whole sequence's rows at once, so its blocks of one form are
    one run."""
    L = _mk(alpha, rows=160)
    kb = L.block_bytes
    assert L.n_immutable == 10
    for ly in range(L.num_layers):
        al = L.alpha_of_layer(ly)
        for s in range(L.n_seq):
            ks = _seq_ks(L, s, L.n_immutable).tolist()
            for act, f in ((True, L.act_offset), (False, L.k_offset)):
                offs = [f(ly, k) for k in ks if block_is_act(k, al) is act]
                assert all(b - a == kb for a, b in zip(offs, offs[1:]))


@pytest.mark.parametrize("geom", [_GEOM_13B, _GEOM_67B])
@pytest.mark.parametrize("rows", [1025, 1024, 1000, 999, 17, 16, 0])
def test_block_regions_are_hole_free_for_any_prompt_length(geom, rows):
    """The property the dense block regions exist for: every block region
    is perfectly tiled by its blocks — whole blocks back to back, no
    per-sequence berth, no reserved piece for the mutable prompt block — for
    any prompt length. An unwritten hole inside a block region costs the
    blob read speed for the rest of its life, so slack is not merely wasted
    space.

    The prompt lengths that are not a multiple of the block size are what
    give the test its power: a layout carrying per-sequence slack fails the
    tiling on exactly those."""
    L = _mk(0.7633, rows=rows, **geom)
    kb = L.block_bytes
    ks = _all_ks(L)
    for ly in range(L.num_layers):
        al = L.alpha_of_layer(ly)
        block_regions = _block_regions(L, ly)
        for kind, f in ((0, L.act_offset), (1, L.k_offset),
                        (2, lambda ly, k: L.k_offset(ly, k) + L.v_delta(ly))):
            base, nbytes = block_regions[kind]
            offs = sorted(f(ly, int(k)) for k in ks
                          if block_is_act(int(k), al) is (kind == 0))
            assert offs[0] == base
            assert offs == list(range(base, base + len(offs) * kb, kb))
            assert len(offs) * kb == nbytes


def test_hot_path_tuple_matches_the_accessors():
    """Builders take `region_bases` once per layer and compute the closed
    form inline. If the two ever disagree, saves and reloads address
    different bytes — silently."""
    L = _mk([0.0, 0.6, 1.0], rows=97)
    for ly in range(L.num_layers):
        tup = L.region_bases(ly)
        assert len(tup) == 3
        abase, kbase, vdelta = tup
        alpha = L.alpha_of_layer(ly)
        assert vdelta == L.v_delta(ly)
        for k in _all_ks(L).tolist():
            if block_is_act(k, alpha):
                hoisted = abase + act_rank(k, alpha) * L.block_bytes
                assert hoisted == L.act_offset(ly, k)
            else:
                hoisted = kbase + kv_rank(k, alpha) * L.block_bytes
                assert hoisted == L.k_offset(ly, k)
                assert hoisted + vdelta == L.k_offset(ly, k) + L.v_delta(ly)

