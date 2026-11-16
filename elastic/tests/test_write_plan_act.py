"""Writes are one request per block region per production event: the
layout's offset formula makes an event's blocks of one block region a
contiguous range, so the request is the first block's offset and the
blocks' views in order, wherever their bytes are — ACT first, then K, then
V. Nothing is checked, merged or cut, and memory adjacency is never looked
at; a request ends where the FILE offset stops continuing it, always on a
block boundary."""

import numpy as np
import pytest
import torch

from elastic.runtime.block_kv_storage import BlockKVLayout
from elastic.runtime.labels import act_rank, block_is_act, n_act_blocks
from elastic.vllm_runtime.nvme_write_router import WriteRouter, _Desc
from elastic.vllm_runtime.streaming_loader import (
    _decode_blocks, _event_writes, _prefill_blocks)

N_SEQ, L, NBLK, BS, H, D = 4, 2, 8, 4, 2, 128  # tok_b = 512 B, kb = 2048 B
KB = BS * H * D * 2
TOK_B = H * D * 2
PROMPT_LEN = 4 * BS + 1                    # n=4, rem=1: mutable prompt blocks


class _Ring:
    """The NVMe engine's write port as `_submit` sees it: every request handed
    over is recorded, in order."""

    def __init__(self):
        self.items = []

    def submit_writev(self, off, bufs, user_data=0):
        self.items.append((off, list(bufs)))
        return object()                      # keepalive handle

    def drain(self):
        pass


class _Router:
    """The producer's view of the router: the layout and the cache blocks,
    plus the router's own hand-over (`_submit`) onto a recording port.
    The cache is a byte array so a view's address names its block."""
    _submit = WriteRouter._submit
    _account_built = WriteRouter._account_built

    def __init__(self, layout):
        self.layout = layout
        self.cache = np.zeros(2 * N_SEQ * NBLK * KB + 4096, dtype=np.uint8)
        self.ring = _Ring()
        self._next_ud = 1
        self._inflight = {}
        self._audit_on = False
        self._timing = True
        self.n_act_saves = self.n_kv_saves = self.n_items = 0
        for _f in ("st_e2s", "st_e2ev", "st_ev2s"):
            setattr(self, _f, 0.0)
        for _f in ("st_src_n", "st_src_bad", "st_ph_n", "st_ph_bad"):
            setattr(self, _f, 0)
        self.st_ph_by = {}
        self.st_src_off = {}

    def kv_block_views(self, layer, cb):
        a0 = (-self.cache.ctypes.data) % 4096
        k = self.cache[a0 + cb * KB:a0 + (cb + 1) * KB]
        v = self.cache[a0 + (N_SEQ * NBLK + cb) * KB:a0 + (N_SEQ * NBLK + cb + 1) * KB]
        return k, v

    def cb_of(self, view):
        a0 = (-self.cache.ctypes.data) % 4096
        d = view.ctypes.data - (self.cache.ctypes.data + a0)
        return (d // KB) % (N_SEQ * NBLK), d // KB >= N_SEQ * NBLK


@pytest.fixture(scope="module")
def layout():
    lay = BlockKVLayout(n_seq=N_SEQ, num_layers=L, max_blocks_per_seq=NBLK,
                        block_size=BS, num_heads=H, head_dim=D,
                        dtype=torch.float16, alpha=[1.0, 0.5])
    lay.set_prompt_len(PROMPT_LEN)
    return lay


@pytest.fixture(scope="module")
def act_buffer():
    """The activation buffer: a row pack of N_seq prompts with room past it,
    page-aligned like the real one."""
    n = 2 * N_SEQ * NBLK * KB
    raw = np.zeros(n + 4096, dtype=np.uint8)
    a0 = (-raw.ctypes.data) % 4096
    return raw[a0:a0 + n]


def _rows():
    return [(p * PROMPT_LEN, p, 0, PROMPT_LEN) for p in range(N_SEQ)]


def _pos(act_buffer, v):
    return int(v.ctypes.data - act_buffer.ctypes.data)


@pytest.mark.parametrize("layer", [0, 1])
def test_immutable_blocks_one_request_each_at_their_offsets(layout,
                                                            act_buffer, layer):
    al = layout.alpha_of_layer(layer)
    r = _Router(layout)
    act_e, kv_e = _event_writes(
        r, layer, _prefill_blocks(layout, layer, _rows(), act_buffer, False))
    n = layout.n_immutable
    want = [(s, j) for s in range(N_SEQ) for j in range(n)]  # production order
    ks = [int(layout.ks_of_tokens(s, j * BS)) for s, j in want]
    acts = [(s, j, k) for (s, j), k in zip(want, ks) if block_is_act(k, al)]
    kvs = [(s, j, k) for (s, j), k in zip(want, ks) if not block_is_act(k, al)]
    # ACT: ONE request for the whole span's ACT range, at the first
    # activation block's offset, one view per block from the pack, in rank
    # order
    assert len(act_e) == 1
    off, views = act_e[0]
    assert off == layout.act_offset(layer, acts[0][2])
    assert [_pos(act_buffer, v) for v in views] == \
        [(s * PROMPT_LEN + j * BS) * TOK_B for s, j, _k in acts]
    assert all(v.nbytes == KB for v in views)
    # and the range really is contiguous by the formula
    assert [layout.act_offset(layer, k) for _s, _j, k in acts] == \
        [off + i * KB for i in range(len(acts))]
    # KV: one K request then one V request, one cache-block view per block
    # (none at all on an alpha=1 layer — no block there is a KV block)
    if not kvs:
        assert kv_e == []
        return
    assert len(kv_e) == 2
    (ko, kviews), (vo, vviews) = kv_e
    assert ko == layout.k_offset(layer, kvs[0][2])
    assert vo == ko + layout.v_delta(layer)
    assert [r.cb_of(v) for v in kviews] == \
        [(layout.mem_block(layer, k), False) for _s, _j, k in kvs]
    assert [r.cb_of(v) for v in vviews] == \
        [(layout.mem_block(layer, k), True) for _s, _j, k in kvs]


@pytest.mark.parametrize("layer", [0, 1])
def test_mutable_prompt_blocks_at_the_back_from_their_place(layout, act_buffer,
                                                           layer):
    """rem=1: each row's block n (its mutable prompt block, at the start of
    the mutable range, at the back of the file) — a whole block from the
    pack, right after the row's immutable blocks. Only the mutable prompt
    blocks: the immutable blocks are the span's."""
    al = layout.alpha_of_layer(layer)
    r = _Router(layout)
    act_e, kv_e = _event_writes(
        r, layer, _prefill_blocks(layout, layer, _rows(), act_buffer, True))
    n = layout.n_immutable
    ks = [N_SEQ * layout.n_immutable + s for s in range(N_SEQ)]
    acts = [(s, k) for s, k in zip(range(N_SEQ), ks) if block_is_act(k, al)]
    kvs = [k for k in ks if not block_is_act(k, al)]
    assert len(act_e) == 1
    off, views = act_e[0]
    assert off == layout.act_offset(layer, acts[0][1])
    assert [_pos(act_buffer, v) for v in views] == \
        [(s * PROMPT_LEN + n * BS) * TOK_B for s, _k in acts]
    if kvs:
        assert len(kv_e) == 2 and all(len(v) == len(kvs) for _o, v in kv_e)
        assert all(o >= layout.region_bases(layer)[1] for o, _v in kv_e)
    else:
        assert kv_e == []


def test_decode_block_is_written_from_its_place(layout, act_buffer):
    """A decode block's request carries the block, from its place
    `act_rank·kb` — where the reload landed its earlier rows and the new
    token's row was appended."""
    layer = 1
    al = layout.alpha_of_layer(layer)
    n = layout.n_immutable
    base = n * BS + 2                       # 3rd token of block n
    rows = [(s, s, base, 1) for s in range(N_SEQ)]
    r = _Router(layout)
    act_e, kv_e = _event_writes(
        r, layer, _decode_blocks(layout, layer, rows, act_buffer))
    ks = [N_SEQ * layout.n_immutable + s for s in range(N_SEQ)]
    acts = [(s, k) for s, k in zip(range(N_SEQ), ks) if block_is_act(k, al)]
    assert len(act_e) == 1
    off, views = act_e[0]
    assert off == layout.act_offset(layer, acts[0][1])
    assert len(views) == len(acts)
    assert [(_pos(act_buffer, v), v.nbytes) for v in views] == \
        [(act_rank(k, al) * KB, KB) for _s, k in acts]
    assert sum(v.nbytes for v in views) == len(acts) * KB
    assert len(kv_e) == 2 and all(len(v) == N_SEQ - len(acts) for _o, v in kv_e)


def test_every_decode_act_source_is_page_aligned(layout, act_buffer):
    """Every save is one whole block: a decode activation block's write
    source is the block itself at its place `act_rank·kb`, block-aligned and
    kb long, for every form, position in the block and layer. A real
    layout's `kb` is a page multiple, so block-aligned is page-aligned
    wherever it matters; the fixture's kb is deliberately small, so the rule
    is stated in kb."""
    n = layout.n_immutable
    for layer in range(layout.num_layers):
        for so in range(BS):
            rows = [(s, s, n * BS + so, 1) for s in range(N_SEQ)]
            act_e, _kv = _event_writes(
                _Router(layout), layer,
                _decode_blocks(layout, layer, rows, act_buffer))
            for _o, views in act_e:
                for v in views:
                    assert _pos(act_buffer, v) % KB == 0
                    assert v.nbytes == KB


def test_slot_map_minus_bs_is_the_act_row(layout):
    """The decode capture writes a row with the step's own slot map: vLLM's
    slot is `mem_block(l,k)·bs + pos mod bs` and `mem_block = 1 + act_rank`
    for an activation block, so `slot − bs` is exactly that row's place in
    the activation buffer, `act_rank(k)·bs + pos mod bs`. No index of its
    own."""
    for layer in range(layout.num_layers):
        al = layout.alpha_of_layer(layer)
        for s_ in range(N_SEQ):
            for j in range(layout.max_blocks_per_seq):
                k = int(layout.ks_of_tokens(s_, j * BS))
                for so in range(BS):
                    slot = layout.mem_block(layer, k) * BS + so
                    if block_is_act(k, al):
                        assert slot - BS == act_rank(k, al) * BS + so
                    else:
                        # A KV block: no activation bytes. Its row lands
                        # past the layer's activation blocks, in activation
                        # buffer space nothing reads.
                        assert slot - BS >= n_act_blocks(N_SEQ * NBLK, al) * BS
                    # always inside the activation buffer the pool sizes
                    assert 0 <= slot - BS < (layout.mem_blocks_needed - 1) * BS


def test_a_newly_opened_decode_block_is_written_whole(layout, act_buffer):
    layer = 0                                # alpha=1: every block ACT
    n = layout.n_immutable
    al = layout.alpha_of_layer(layer)
    rows = [(0, 0, (n + 1) * BS, 1)]         # first token of block n+1
    act_e, _kv = _event_writes(
        _Router(layout), layer,
        _decode_blocks(layout, layer, rows, act_buffer))
    (_o, views), = act_e
    k = int(layout.ks_of_tokens(0, (n + 1) * BS))
    assert [v.nbytes for v in views] == [KB]
    assert _pos(act_buffer, views[0]) == act_rank(k, al) * KB


def test_a_file_offset_gap_starts_a_new_request(layout, act_buffer):
    """The boundary is the file's. Enumerated in row order, a row's
    immutable blocks (k in the immutable range) and its unfilled last block
    (the mutable prompt block, whose k the layout puts in the mutable range,
    away from them) are not one range, so each row yields two requests.
    Nothing but the offsets decides that: the views, their order and their
    bytes are untouched."""
    layer = 0                                   # alpha=1: every block ACT
    n = layout.n_immutable
    blocks = []
    for s in range(N_SEQ):
        ks = [int(layout.ks_of_tokens(s, j * BS)) for j in range(n)]
        ks.append(N_SEQ * n + s)                # the row's unfilled last one
        for i, k in enumerate(ks):
            r0 = (s * (n + 1) + i) * KB
            blocks.append((k, (lambda r0=r0: [act_buffer[r0:r0 + KB]]),
                           layout.mem_block(layer, k)))
    act_e, _kv = _event_writes(_Router(layout), layer, blocks, kv=False)
    assert len(act_e) == 2 * N_SEQ
    for s in range(N_SEQ):
        off_p, views_p = act_e[2 * s]
        off_t, views_t = act_e[2 * s + 1]
        assert off_p == layout.act_offset(
            layer, int(layout.ks_of_tokens(s, 0)))
        assert [_pos(act_buffer, v) for v in views_p] == \
            [(s * (n + 1) + i) * KB for i in range(n)]
        assert off_t == layout.act_offset(layer, N_SEQ * n + s)
        assert [_pos(act_buffer, v) for v in views_t] == [(s * (n + 1) + n) * KB]


def test_the_router_hands_requests_over_as_they_are(layout, act_buffer):
    """One NVMe engine item per request, in order, with the producer's
    views."""
    r = _Router(layout)
    act_e, _kv = _event_writes(
        r, 0, _prefill_blocks(layout, 0, _rows(), act_buffer, False), kv=False)
    d = _Desc(kind="act_batch", layer=0, gate_seq=1, entries=act_e)
    assert r._submit(d) is False
    assert [(o, [v.ctypes.data for v in b]) for o, b in r.ring.items] == \
        [(o, [v.ctypes.data for v in b]) for o, b in act_e]
    assert r.n_items == d.n_items == len(act_e) and r.n_act_saves == 1
