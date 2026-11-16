"""ReadRouter (`elastic-head`): the loader, whose visit pointer leads the
slot rotation.

The loader waits for read completions — and, when every slot is live, for
the writeback to release a slot. Each time one arrives it takes in the
completions that have landed and fills the queue back up:

  * taking in: a layer's weight in → `mark_ready` (the forward pass may
    start the layer); a K/V pair in → hand the blocks to the forward pass
    (arrival push); the activation blocks of a kv_gen launch in (and the
    layer's weight) → launch kv_gen on them on the kv_gen stream and hand
    them over with the event recorded after it; every read of a layer handed
    over → close the layer's feed.
  * filling: keep two layers loading. When the next thing to queue is a new
    layer, check its slot is released and acquire it (`acquire`), build the
    layer's plan and hand its segments to the NVMe engine, in the layer's
    load order (weight | whole ACT_i block region | K_i,V_i alternating), as
    far as the NVMe engine FIFO holds them; the rest go out as completions
    come back.

That is all it does. It never waits for kv_gen (the forward pass's stream
waits on the launch's event), never releases a slot (the writeback does),
and never touches the device: every io_uring call lives in the C
`nvme_engine` pthread (the NVMe engine), which consumes the FIFO in order
and keeps the read side topped up to ELASTIC_READ_QD. The loader sleeps on
two fds — the NVMe engine's read-completion fd and the pool's `release_fd` —
with no timeout.
"""

from __future__ import annotations

import errno
import os
import select
import sys
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from elastic.runtime.labels import n_act_blocks as _n_act_blocks
from elastic.runtime.weight_table import DTYPE as _WEIGHT_DTYPE, WeightTable

from .layer_pool import LayerSlotPool


def act_segments_and_launches(n_blocks: int, region_base: int, k_bytes: int,
                   segment_blocks: int, launch_blocks: int):
    """Cut an ACT_i block region's ranks [0, n_blocks) into read segments
    and, apart from them, into kv_gen launches.

    Returns `(segments, launches)`:
      segments  [(file_off, n, rank0)] — cut every `segment_blocks` blocks:
                the read segment (ELASTIC_READ_SPAN_MB), the same size as
                for the weights and the K_i/V_i block regions;
      launches  [(launch0, n, segment_ids)] — cut every `launch_blocks`
                blocks (ELASTIC_LOAD_CHUNK_TOKENS), with the indices of the
                segments it overlaps: the kv_gen launch goes out once those
                have landed.

    The two sizes are independent. kv_gen's cost per token is set by the
    launch — a launch whose token count is not a multiple of 1024 costs more
    per token (the kernel's tile group, GROUP_M·BLOCK_M, covers 1024 rows) —
    and the read's by its segment; a launch may span two segments and a
    segment may hold several launches.

    A launch is one dense slice [launch0, launch0+n) of the activation buffer
    (a block sits at `rank·k_bytes`, which the decode ACT save and the
    capture both address) AND one stretch of K and V buffer blocks
    (`mem_act_base + rank`), which is what lets kv_gen take a single base
    block.

    Bytes and destinations come from the layout's closed form, so a layer's
    lists cost O(segments + launches), never O(blocks)."""
    segment_blocks = max(1, int(segment_blocks))
    launch_blocks = max(1, int(launch_blocks))
    n_blocks = int(n_blocks)
    segments = [(region_base + a * k_bytes, min(segment_blocks, n_blocks - a), a)
             for a in range(0, n_blocks, segment_blocks)]
    launches = []
    for a in range(0, n_blocks, launch_blocks):
        n = min(launch_blocks, n_blocks - a)
        launches.append((a, n, list(range(a // segment_blocks,
                                       (a + n - 1) // segment_blocks + 1))))
    return segments, launches


class ReadRouter:
    """The read path's scheduler: the loader, whose visit pointer leads the
    rotation. `Read` because the write path has its own (`WriteRouter`, the
    writeback); `Router` because the form deciding a block's path (KV block
    = one DMA hop / activation block = DMA + kv_gen hop, same destination) is
    the routing."""

    _UD_SHIFT = 32               # user_data = (plan id << 32) | segment index
    _UD_MASK = (1 << 32) - 1

    def __init__(self,
                  pool: LayerSlotPool,
                  kv_nvme,
                  weight_path,
                  weight_table: WeightTable,
                  block_load_state: dict[str, Any]):
        self._pool = pool
        self._kvn = kv_nvme
        self._weight_table = weight_table
        self._specs = weight_table.layers
        self._block_load_state = block_load_state
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._arrival = block_load_state["arrival"]
        # The loader's two fds. Both are eventfds: a count written while the
        # loader is busy is still there when it gets to its wait.
        self._cq_fd = int(kv_nvme.completion_fd())
        self._fds = [self._cq_fd, pool.release_fd]
        # The weight blob rides the same read ring as the KV/ACT reads
        # (io_uring carries the fd per SQE), attached once here, before the
        # read side holds any work.
        self._w_fidx = kv_nvme.attach(os.fspath(weight_path))
        self._segment_mb = max(
            1, int(os.environ["ELASTIC_UNIFIED_WEIGHT_MB"]))
        # Each kv_gen launch's arrival item carries an event recorded on the
        # kv_gen stream right after the launch; the forward pass's stream
        # waits on it.
        from elastic.loader.nvme_loader import EventRing
        self._ev_ring = EventRing()
        # Plans in the queue, oldest first, and the loader's own position:
        # the next visit and the layer it holds. Visits are consecutive and
        # the layer cycles with them.
        self._plans: deque = deque()
        self._visit = 0
        self._layer = 0
        # The step being loaded: None for a batch's prefill step (weights
        # only; the prefill step samples the first token), else its decode
        # step t, 1 ≤ t < gen. The loader starts before the first batch, so
        # its first step is that batch's prefill.
        self._step: Optional[int] = None
        # The batch that step belongs to: its longest prompt and its token
        # count, taken from the gate's declaration as the loader leaves the
        # batch's prefill step.
        self._prompt_len = 0
        self._gen = 0
        self._plan_by_pid: dict = {}
        self._pid = 0
        # Segments handed to the NVMe engine whose completion this thread
        # has not taken yet. The NVMe engine holds at most `capacity` of
        # them, so the fill stops there.
        self._segments_out = 0
        self._fifo_cap = int(kv_nvme.capacity)
        self._rio_t0 = 0.0
        # ---- timing (ELASTIC_LOADER_TIMING=1) ----------------------------
        self._timing = os.environ.get("ELASTIC_LOADER_TIMING") == "1"
        self._t = {"n": 0,
                   # the loader's two waits: sleeping in select, and which
                   # fd ended it (cq: a read landed; rel: the writeback
                   # released)
                   "wait_n": 0, "wait_t": 0.0,
                   "wait_cq": 0, "wait_rel": 0,
                   # acquire -> plan built, and -> its segments handed
                   # over; `e2s_max` the worst one
                   "e2b": 0.0, "e2s": 0.0, "e2s_n": 0, "e2s_max": 0.0,
                   # plan build: the extent arithmetic and the segment
                   # build
                   "wlist": 0.0, "wl_it": 0, "dbuild": 0.0,
                   # time with reads in flight, and the kv_gen dispatch
                   "rio": 0.0, "rcompute": 0.0,
                   # bytes handed over, by kind (ACT, KV, weight)
                   "b_act": 0, "b_kv": 0, "b_w": 0,
                   # device occupancy, time-weighted between samples
                   "occ_t": 0.0, "occ_n": 0.0, "occ_z": 0.0,
                   "occ_cq": 0, "occ_s": 0,
                   # segment shape per kind: count, bytes
                   "nk_w": 0, "nk_a": 0, "nk_k": 0,
                   "bk_w": 0, "bk_a": 0, "bk_k": 0}
        self._occ_d = -1
        self._occ_last = 0.0
        self._t_wall0: Optional[float] = None

    # ---- lifecycle ---------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="elastic-head",
                                          daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Teardown: the loader stops at its next wake. What wakes it is the
        writeback's exit (it writes the release fd once, on its way out) —
        the loader has nothing left to load once the forward pass is done."""
        self._stop.set()

    def join(self) -> None:
        self._thread.join()

    def _run(self) -> None:
        try:
            self._serve()
        except BaseException as e:
            import traceback as _tbm
            sys.stderr.write(
                f"[ELASTIC] HEAD ERROR: {type(e).__name__}: {e}\n"
                + _tbm.format_exc())
            sys.stderr.flush()

    # ---- the loop ----------------------------------------------------

    def _serve(self) -> None:
        if self._timing:
            self._t_wall0 = time.perf_counter()
        plans = self._plans
        self._fill(plans)
        while True:
            if self._stop.is_set():
                return
            self._wait()
            if self._stop.is_set():
                return
            self._take_in(plans)
            self._fill(plans)

    def _wait(self) -> None:
        """Sleep until a read lands or the writeback releases a slot. Both fds
        are level-triggered eventfds, so nothing can be slept through, and
        there is no timeout: nothing else can give the loader work."""
        _t = time.perf_counter() if self._timing else 0.0
        _r, _, _ = select.select(self._fds, (), ())
        for _fd in _r:
            # Drain: a count left set would make the next select return at
            # once. Anything posted after this read re-arms it.
            os.read(_fd, 8)
        if self._timing:
            self._t["wait_n"] += 1
            self._t["wait_t"] += time.perf_counter() - _t
            if self._cq_fd in _r:
                self._t["wait_cq"] += 1
            if self._pool.release_fd in _r:
                self._t["wait_rel"] += 1

    def _take_in(self, plans) -> None:
        """Every completion the NVMe engine has posted: account it, then
        launch what that made ready and close the layers it finished — in
        layer order, and within a layer in kv_gen launch order, so the kv_gen
        stream and the arrival feed see the layers in sequence."""
        kvn = self._kvn
        while True:
            r = kvn.try_wait_one()
            if r is None:
                break
            self._reap_segment(r[0], r[1])
        if self._timing:
            self._occ_sample()
        self._dispatch_ready(plans)
        while plans and self._plan_complete(plans[0]):
            self._finish_plan(plans.popleft())

    def _fill(self, plans) -> None:
        """Keep two layers loading: the one the drive is reading and the next,
        queued behind it so the drive goes from one layer's reads to the
        next's without a gap. The device's own in-flight bound is the NVMe
        engine's (ELASTIC_READ_QD × ELASTIC_READ_SPAN_MB); this bounds only
        how many layers' segments sit in its FIFO.

        Segments go out in plan order while fewer than the FIFO's capacity
        are out; a plan the capacity cut short goes on from where it stopped
        at the next fill, which every completion taken is followed by.

        A new layer starts with `acquire`: if its slot is not released yet the
        loader stops here, and the writeback's release wakes it."""
        for p in plans:
            if not self._enqueue(p):
                return
        while sum(1 for p in plans if p["done"] < len(p["segments"])) < 2:
            if self._stop.is_set():
                return
            _t0 = time.perf_counter() if self._timing else 0.0
            slot = self._pool.acquire(self._visit, self._layer)
            if slot is None:
                return
            if self._layer == 0 and self._visit > 0:
                self._next_step()
            p = self._plan_layer(self._layer, slot)
            self._visit += 1
            self._layer = (self._layer + 1
                           if self._layer + 1 < len(self._specs) else 0)
            p["utag"] = self._pid << self._UD_SHIFT
            self._plan_by_pid[self._pid] = p
            self._pid += 1
            plans.append(p)
            if self._timing:
                _tb = time.perf_counter()
                self._t["e2b"] += _tb - _t0
            _all = self._enqueue(p)
            if self._timing:
                _d = time.perf_counter() - _t0
                self._t["e2s"] += _d
                self._t["e2s_n"] += 1
                if _d > self._t["e2s_max"]:
                    self._t["e2s_max"] = _d
            if not _all:
                return

    def _enqueue(self, p) -> bool:
        """Hand plan `p`'s next segments to the NVMe engine, in order, while
        the FIFO has room. True once every one of them is out."""
        segments = p["segments"]
        while p["si"] < len(segments):
            if self._segments_out >= self._fifo_cap:
                return False
            self._sub_segment(p, p["si"])
            p["si"] += 1
        return True

    # ---- one layer's plan --------------------------------------------

    def _plan_layer(self, layer_idx: int, slot) -> dict:
        """Build the plan of the layer whose slot was just acquired: its
        weight segments, its reload segments and the callables that turn
        their completions into the forward pass's input.

        What it reloads is the closed form of the step (`_build_plan`)."""
        self._arrival.begin(layer_idx)
        spec = self._specs[layer_idx]
        plan = self._build_plan(layer_idx, slot)
        # Weight first: the layer loads `[weight | ACT_i | K_i,V_i]`, in the
        # queue in that order, so planning ahead never puts a later layer's
        # weight in front of this layer's reloads.
        chunk = self._segment_mb << 20
        off, nbytes = 0, int(spec.bytes_padded)
        wsegments = []
        while off < nbytes:
            n = min(chunk, nbytes - off)
            wsegments.append((int(spec.file_start) + off, 2,
                           (slot.raw_buf[off:off + n], n), None))
            off += n
        plan["w_left"] = len(wsegments)
        plan["segments"] = wsegments + plan["segments"]
        return plan

    def _next_step(self) -> None:
        """The step being entered follows the previous one: a batch is its
        prefill step, then its decode steps t = 1 .. gen−1 (the prefill step
        produces token 1, decode step t token t+1), then the next batch's
        prefill; a batch of gen 1 has no decode step. The batch's shape is
        the gate's declaration at its prefill step
        (`BlockKVLayout.declare_batch`), made before that step's forward — so
        before any of the prefill's slots is released, and so before the
        loader can leave the prefill step. The loader takes the declarations
        in order (`take_batch`), one per batch: after a gen-1 batch the next
        batch's may already have been made."""
        if self._step is None:
            self._prompt_len, self._gen = (
                self._block_load_state["storage_layout"].take_batch())
            self._step = 1 if self._gen > 1 else None
        elif self._step + 1 < self._gen:
            self._step += 1
        else:
            self._step = None

    def _build_plan(self, layer_idx: int, slot) -> dict:
        """This layer's reload segments: the blocks its previous visit
        left on disk, mapped from their file offsets to their buffer blocks.

        Every sequence steps together, so "on disk" is the prefix [0, N_seq·c)
        of global numbers k, and the rule that gives a block its form is a
        monotone partition of it: with K = N_seq·c, its first ⌈K·α⌉ blocks are
        activation blocks (ranks [0, ⌈K·α⌉) of ACT_i) and the rest KV blocks
        (ranks [0, K − ⌈K·α⌉) of K_i/V_i). The plan is one block count per
        block region, each read from rank 0 of its block region."""
        state = self._block_load_state
        act_loader = state["act_loader"]
        storage_layout = state["storage_layout"]
        act_pool = state["act_pool"]
        weight_table = self._weight_table
        block_size = storage_layout.block_size
        _tw = time.perf_counter() if self._timing else 0.0
        n_kv = n_act = 0
        _nseq = storage_layout.n_seq
        # Decode step t (1 ≤ t < gen) reloads the blocks of every token up to
        # and including the new token at position S + t − 1, ⌈(S + t)/bs⌉ per
        # sequence — the current block too; the forward pass places the new
        # token into it after it arrives. A prefill step reads only the
        # weights.
        _c = (0 if self._step is None else
              -(-(self._prompt_len + self._step) // block_size))
        # "The first c blocks of every sequence" is the first `N_seq·c` global
        # numbers k — a prefix, not a set.
        _K = _nseq * _c
        if _K:
            n_act = _n_act_blocks(_K, storage_layout.alpha_of_layer(layer_idx))
            n_kv = _K - n_act
        if self._timing:
            self._t["wlist"] += time.perf_counter() - _tw
            self._t["wl_it"] += _K
        _ready = [False]
        pool = self._pool

        def _ready_fire():
            # The layer's last weight byte has landed.
            pool.mark_ready(slot)
            _ready[0] = True

        plan = {"layer": layer_idx,
                "sts": [], "segments": [], "dispatch": None, "sub": None,
                "kv_submit": None, "kv_complete": None, "kv_reads": [],
                "done": 0, "si": 0, "di": 0, "utag": 0, "w_left": 0,
                "ready": _ready, "ready_fire": _ready_fire}
        _tb = time.perf_counter() if self._timing else 0.0
        segments: list = []
        if n_act:
            _ap = self._pipeline_layer_reads(
                layer_idx, slot, n_act, block_size,
                storage_layout, act_loader, weight_table, act_pool)
            plan["sts"] = _ap["sts"]
            plan["dispatch"] = _ap["dispatch"]
            plan["sub"] = _ap["sub"]
            segments.extend(_ap["segments"])
        if n_kv:
            _kv_reads, _kv_sub, _kv_cmp = self._direct_read_kv(
                layer_idx, n_kv, storage_layout)
            plan["kv_submit"] = _kv_sub
            plan["kv_complete"] = _kv_cmp
            plan["kv_reads"] = _kv_reads
            for _i, _rd in enumerate(_kv_reads):
                segments.append((_rd[0], 1, None, _i))
        plan["segments"] = segments
        if self._timing:
            self._t["dbuild"] += time.perf_counter() - _tb
        return plan

    # ---- completions -------------------------------------------------

    def _sub_segment(self, p, gi) -> None:
        """Hand segment `gi` of plan `p` to the NVMe engine — one FIFO item
        (a memcpy and a kick; the NVMe engine builds the SQE at its own
        pace)."""
        _off, _kind, _pay, _own = p["segments"][gi]
        _ud = p["utag"] | gi
        _nb = self._segment_bytes(p, _kind, _pay, _own)
        if _kind == 2:
            self._kvn.submit_read(_off, _nb, _pay[0], user_data=_ud,
                                  file_idx=self._w_fidx)
            if self._timing:
                self._shape("w", _nb)
        elif _kind == 0:
            p["sub"](_pay, _ud)
            if self._timing:
                self._shape("a", _nb)
        else:
            p["kv_submit"](_own, _ud)
            if self._timing:
                self._shape("k", _nb)
        if self._segments_out == 0 and self._timing:
            # The drive goes from idle to busy: start the rio clock.
            self._rio_t0 = time.perf_counter()
        self._segments_out += 1

    @staticmethod
    def _segment_bytes(p, kind, pay, own) -> int:
        """The bytes a segment reads — what its completion must report."""
        return p["kv_reads"][own][3] if kind == 1 else pay[1]

    def _shape(self, k: str, nbytes: int) -> None:
        self._t["nk_" + k] += 1
        self._t["bk_" + k] += int(nbytes)

    def _io_done_one(self) -> None:
        self._segments_out -= 1
        if self._segments_out == 0 and self._timing:
            self._t["rio"] += time.perf_counter() - self._rio_t0

    def _reap_segment(self, ud, res) -> None:
        """Account one completion and credit its owner. The io_uring result
        is the only proof the bytes landed: a negative or short one means the
        destination still holds whatever it held before, and crediting it
        would run kv_gen on stale activation-buffer bytes or declare blocks
        arrived whose last bytes were never loaded — so it raises."""
        p = self._plan_by_pid[ud >> self._UD_SHIFT]
        self._io_done_one()
        _off, _kind, _pay, _own = p["segments"][ud & self._UD_MASK]
        _exp = self._segment_bytes(p, _kind, _pay, _own)
        if res < 0 or res < _exp:
            _what = ("ACT", "KV", "weight")[_kind]
            _ctx = (f"{_what} L={p['layer']} ud=0x{ud:x} off={_off} "
                    f"want={int(_exp)}B")
            if res < 0:
                raise OSError(-int(res), os.strerror(-int(res)),
                              f"NVMe read failed: {_ctx}")
            raise OSError(errno.EIO, os.strerror(errno.EIO),
                          f"NVMe SHORT read: {_ctx} got={int(res)}B")
        p["done"] += 1
        if _kind == 2:
            if self._timing:
                self._t["b_w"] += _exp
            p["w_left"] -= 1
            if p["w_left"] == 0:
                # The layer's last weight byte just landed: publish it.
                p["ready_fire"]()
        elif _kind == 0:
            for _st in _own:
                _st["remaining"] -= 1
        else:
            p["kv_complete"](_own)

    def _dispatch_ready(self, plans) -> None:
        """Issue every kv_gen launch whose reads have all landed — in layer
        order, and within a layer in launch order, once the layer's weight is
        in (kv_gen slices its K/V projections out of the weight buffer). A
        later layer's launch never jumps ahead of an earlier layer's."""
        for p in plans:
            sts = p["sts"]
            while (p["di"] < len(sts) and sts[p["di"]]["remaining"] == 0
                    and p["ready"][0]):
                p["dispatch"](sts[p["di"]])
                p["di"] += 1
            if p["di"] < len(sts):
                break

    @staticmethod
    def _plan_complete(p) -> bool:
        return p["done"] >= len(p["segments"]) and p["di"] >= len(p["sts"])

    def _finish_plan(self, p) -> None:
        """Every read of the layer has landed and every kv_gen launch is
        issued: close the layer's feed, so the forward pass's consumption of
        the layer's arrival feed terminates."""
        self._arrival.close(p["layer"])
        del self._plan_by_pid[p["utag"] >> self._UD_SHIFT]
        if self._timing:
            self._t["n"] += 1
            if self._t["n"] % len(self._specs) == 0:
                self._emit_timing()

    def _arrival_push(self, layer_idx, kind: int, rank0: int, n: int,
                      event=None) -> None:
        """Hand the forward pass `n` consecutive ranks of one block region of
        the layer (kind 0 = ACT_i, 1 = K_i/V_i) whose bytes just became
        available for attention. The item names the rank run and nothing
        else: which sequences it touches is the consumer's `seq_of_ks`."""
        from .layer_arrival import ArrivalItem
        self._arrival.push(layer_idx, ArrivalItem(
            kind=int(kind), rank0=int(rank0), n=int(n), event=event))

    # ---- segment builders --------------------------------------------

    def _pipeline_layer_reads(self, layer_idx, slot, n_act, block_size,
                                storage_layout, act_loader, weight_table,
                                act_pool) -> dict:
        """This layer's ACT reads, cut into kv_gen launches.

        Offset-driven: the layer's ACT_i block region is a dense run of
        blocks starting at `act_base` (act_rank is a bijection onto
        0..n_act-1), so the builder cuts the block region's ranks [0, n_act)
        into read segments of the segment size (ELASTIC_READ_SPAN_MB) and,
        apart from that, into kv_gen launches of `launch_blocks` blocks (=
        ELASTIC_LOAD_CHUNK_TOKENS tokens).
        No sequence enters it anywhere: a launch is one dense ACT-rank run,
        and that run is what its arrival item names. The segments read into
        the layer's activation buffer at `rank·K_bytes`; once the ones a
        launch overlaps and the weight have landed, the loader calls kv_gen
        twice for the launch (once for K, once for V) on the kv_gen stream
        (activations in the activation buffer → K,V, read straight through
        UVA and scatter-stored into kv_caches) and pushes its arrival item
        with the event recorded after them.

        Returns `{"sts", "segments", "dispatch", "sub"}`."""
        import torch
        L = storage_layout
        kv_nvme = self._kvn
        elem_bytes = torch.empty(0, dtype=L.dtype).element_size()
        K_bytes = block_size * L.num_heads * L.head_dim * elem_bytes
        hidden_dim = L.num_heads * L.head_dim
        # An ACT reload reads only one K-sized part per block (= the saved
        # hidden). A kv_gen launch is a launch grouping, not an allocation:
        # this many blocks per launch. The layer's whole ACT lives in its
        # slot's activation buffer regardless of how it is grouped.
        launch_blocks = act_pool.launch_blocks
        nkvh, hd = act_loader.num_kv_heads, act_loader.head_dim
        # The ACT path's projection weights are constant for the layer —
        # slice once.
        wk, wv, bk, bv = self._slice_kv_proj(layer_idx, slot,
                                              weight_table, nkvh, hd)
        # Segment size: ELASTIC_READ_SPAN_MB (16 MiB) segments of the block
        # region, the same as for the weights and the K_i/V_i block regions;
        # kv_gen launches are cut apart from it (`act_segments_and_launches`).
        _segment_b = max(1, int(os.environ["ELASTIC_READ_SPAN_MB"])) << 20
        _abase = L.region_bases(layer_idx)[0]
        _dcs, _wvs = act_segments_and_launches(n_act, _abase, K_bytes,
                                    _segment_b // K_bytes, launch_blocks)
        cache = self._block_load_state["kv_caches"][layer_idx]
        # This layer's activation buffer — in its slot, the same rotation
        # (and the same slot number) `cache` (the kv_caches alias, the K and
        # V buffers) and the weight buffer follow.
        act_buffer = act_pool.act_buffer(layer_idx)
        astream = act_loader._stream()
        from elastic.kernels.kv_gen import kv_gen
        # W_k/W_v are read straight from the weight buffer (UVA views),
        # exactly as the biases are. Ordering belongs in the GPU queue
        # (astream is FIFO, so this layer's kv_gen follows whatever is already
        # queued there); the loader never waits on GPU work.
        #
        # Slot lifetime needs no fence here: the forward pass's stream waits
        # on each launch's event before attending it, so the layer end —
        # which the writeback must hear before it releases the slot — is
        # after every kv_gen of the layer on the device.
        ev_ring = self._ev_ring

        def _c_dispatch(st):
            """The launch's reads and the layer's weight are in: run kv_gen
            on it and hand it over."""
            rank0 = st["rank0"]
            ntot = st["nblk"]
            tok = ntot * block_size
            _tc = time.perf_counter() if self._timing else 0.0
            # No gather: the matmul reads the activations in the activation
            # buffer directly through its unified-memory view.
            flat = (act_buffer.t_buf[rank0 * K_bytes:(rank0 + ntot) * K_bytes]
                    .view(L.dtype).view(tok, hidden_dim))
            # K and V buffer block of the launch's first block. Block j of the
            # dense A matrix lands in `_bp + j` — the layout's own base, the
            # ACT twin of `mem_kv_base`.
            _bp = L.mem_act_base() + rank0
            with torch.cuda.stream(astream):
                kv_gen(flat, wk, bk, cache[0], _bp, block_size)
                kv_gen(flat, wv, bv, cache[1], _bp, block_size)
            # The launch's item goes out now — ACT ranks [rank0, rank0+ntot),
            # one run by construction — with an event recorded on the kv_gen
            # stream right after the launch. The forward pass's stream waits
            # on it before attending the launch's blocks; this thread waits
            # for nothing. (astream is in order, so the V call finishing means
            # K did too.)
            _ev = ev_ring.record(astream)
            self._arrival_push(layer_idx, 0, rank0, ntot, _ev)
            if self._timing:
                self._t["rcompute"] += time.perf_counter() - _tc

        def _act_segment_submit(_pay, _ud):
            """Submit one ACT read segment `(off, nbytes, dst)`."""
            _o, _nb, _dst = _pay
            kv_nvme.submit_read(_o, _nb, _dst, user_data=_ud)

        # Order within the layer's load: weight (prepended by
        # `_plan_layer`) -> the whole ACT_i block region -> K and V
        # alternating per segment. Never re-sorted by file offset: that would
        # emit the whole K_i block region before the whole V_i, and a KV
        # block arrives only once both its K and V have been loaded.
        #
        # A segment reads its stretch of consecutive ACT ranks, one
        # contiguous file range, into the same stretch of the activation
        # buffer (`rank·K_bytes`); its owners are the kv_gen launches it
        # overlaps, each of which goes out once all of its segments have
        # landed.
        sts = [{"rank0": int(_w0), "nblk": int(_wn), "remaining": len(_ids)}
               for (_w0, _wn, _ids) in _wvs]
        owners: list = [[] for _ in _dcs]
        for _st, (_w0, _wn, _ids) in zip(sts, _wvs):
            for _d in _ids:
                owners[_d].append(_st)
        segments: list = []
        for (_off0, _nblk, _r0), _own in zip(_dcs, owners):
            _nbytes = _nblk * K_bytes
            _o = _r0 * K_bytes
            segments.append((_off0, 0, (_off0, _nbytes, act_buffer.raw_buf[_o:_o + _nbytes]),
                          _own))
            if self._timing:
                self._t["b_act"] += _nbytes
        return {"sts": sts, "segments": segments,
                "dispatch": _c_dispatch, "sub": _act_segment_submit}

    def _direct_read_kv(self, layer_idx, n_kv, storage_layout):
        """This layer's KV reads: O_DIRECT straight into
        kv_caches[0/1][mem block] — registered UMA memory, so the drive DMAs
        the bytes to their final place, no bounce buffer and no GPU copy.

        Offset-driven: the layer's K_i block region is a dense run of blocks
        from `k_base` and V_i mirrors it one constant `v_delta` further on,
        so a segment is a slice of the block region — `k_base + rank·kb` for
        `n·kb` bytes — landing in K and V buffer blocks `mem_kv_base + rank`,
        one contiguous span.

        A slice arrives (becomes available for attention) when both its K and
        V have been loaded, and goes out as one arrival item (kind 1, rank0,
        n). What orders those bytes against the forward pass is the
        completion itself: the loader takes the CQE first and only then hands
        the item over, so an item the forward pass can see is an item whose
        DMA has landed.

        The block region's ranks [0, n_kv) are read. Returns
        `(reads, _submit_read, _complete_segment)`."""
        import ctypes
        import torch
        L = storage_layout
        kv_nvme = self._kvn
        elem_bytes = torch.empty(0, dtype=L.dtype).element_size()
        K_bytes = L.block_size * L.num_heads * L.head_dim * elem_bytes
        cache = self._block_load_state["kv_caches"][layer_idx]
        kbase = cache[0].data_ptr()
        vbase = cache[1].data_ptr()
        num_blocks = int(cache[0].shape[0])
        total_kbytes = num_blocks * K_bytes
        # Wrap the whole K and V buffers once (zero-copy ctypes→numpy views);
        # per-slice destinations are then cheap numpy slices of this buffer.
        k_whole = np.frombuffer(
            (ctypes.c_uint8 * total_kbytes).from_address(kbase),
            dtype=np.uint8)
        v_whole = np.frombuffer(
            (ctypes.c_uint8 * total_kbytes).from_address(vbase),
            dtype=np.uint8)
        # Segment size: a submit grouping only (bit-identical bytes) —
        # cap a slice at the shared segment size, or a whole α=0 layer would
        # be one multi-GB SQE.
        _CAP_BYTES = max(1, int(os.environ["ELASTIC_READ_SPAN_MB"])) << 20
        # K,V in pairs of one segment each, not whole block region after
        # whole block region: a KV block arrives only once both its K and V
        # have been loaded, and sending the whole K_i first would keep the
        # arrival feed silent across the whole K half and then burst.
        _kbase, _vd = L.region_bases(layer_idx)[1:3]
        _mkv = L.mem_kv_base(layer_idx)
        _PAIR = max(1, _CAP_BYTES // K_bytes)
        # (off0, dst base, dst start, nbytes, (rank0, n)) — the destination is
        # two numbers: ranks r0..r0+n land in K and V buffer blocks
        # `mem_kv_base + r`, one contiguous span, the same for K and for V.
        reads = []
        for _r0 in range(0, int(n_kv), _PAIR):
            _n = min(_PAIR, int(n_kv) - _r0)
            _s = (_mkv + _r0) * K_bytes
            _l = _n * K_bytes
            _mt = (_r0, _n)
            _o = _kbase + _r0 * K_bytes
            reads.append((_o, k_whole, _s, _l, _mt))
            reads.append((_o + _vd, v_whole, _s, _l, _mt))
        n_reads = len(reads)
        # Completion is per segment PAIR: the K and the V segment of a pair
        # carry the same (rank0, n). reads[2i] and reads[2i+1] are pair i.
        rem = bytearray(b"\x02" * (n_reads // 2))

        def _submit_read(_slot, _ud):
            _off, _base, _st, _ln, _ = reads[_slot]
            if self._timing:
                self._t["b_kv"] += _ln
            kv_nvme.submit_read(_off, _ln, _base[_st:_st + _ln],
                                user_data=_ud)

        def _complete_segment(_slot):
            _pair = _slot >> 1
            _left = rem[_pair] - 1
            rem[_pair] = _left
            if _left != 0:
                return
            _r0, _n = reads[_slot][4]
            # No event on this item: the bytes were DMA'd by the drive, not
            # made on a stream. Its completion, taken before this push, is
            # the whole ordering.
            self._arrival_push(layer_idx, 1, _r0, _n, None)

        return (reads, _submit_read, _complete_segment)

    # ---- timing ------------------------------------------------------

    def _occ_sample(self) -> None:
        """Fold the interval since the last sample into the device-occupancy
        integral, then re-read what the drive still owes. A bookkeeping ratio
        sampled when completions are taken, not a reading of the device."""
        kvn = self._kvn
        _now = time.perf_counter()
        if self._occ_d >= 0:
            _dt = _now - self._occ_last
            self._t["occ_t"] += _dt * self._occ_d
            self._t["occ_n"] += _dt
            if self._occ_d == 0:
                self._t["occ_z"] += _dt
        self._occ_last = _now
        self._t["occ_cq"] += kvn.cq_ready()
        _d = kvn.device_inflight()
        self._occ_d = _d if _d > 0 else 0
        self._t["occ_s"] += 1

    def _dev_mem_suffix(self) -> str:
        """torch device-pool bytes. On unified memory these are physical RAM
        outside the process RSS and every standard meminfo counter."""
        try:
            import torch as _t
            return (f" dev(res={_t.cuda.memory_reserved() / 2**30:.1f}G"
                    f" alo={_t.cuda.memory_allocated() / 2**30:.1f}G)")
        except Exception:
            return ""

    def _emit_timing(self) -> None:
        now = time.perf_counter()
        wall = now - self._t_wall0
        d = self._t
        _p = self._pool
        _stg = self._block_load_state["act_pool"]
        _wrt = self._block_load_state["write_router"]
        _wrseg = ""
        if _wrt.st_ndesc:
            _ewi, _ews = _wrt.engine_write_stats()
            _wrseg = (
                f" wr(desc={_wrt.st_ndesc} e2s={_wrt.st_e2s:.1f} "
                f"s2c={_wrt.st_s2c:.1f} {_wrt.st_bytes/1e9:.1f}GB "
                # The NVMe engine's own view: items it consumed and the
                # unit-sized SQEs it composed them into.
                f"items={_ewi} sqes={_ews} "
                # Page misalignment of the write source: a durable
                # property of the file — bytes written from a
                # page-misaligned buffer read back slower for the life of
                # the file.
                f"phmis={_wrt.st_ph_bad}/{_wrt.st_ph_n}"
                + (f"[{','.join(f'{k}:{v}' for k, v in sorted(_wrt.st_ph_by.items()))}]"
                   if _wrt.st_ph_by else "") + " "
                f"srcmis={_wrt.st_src_bad}/{_wrt.st_src_n}"
                + (f" {sorted(_wrt.st_src_off.items())[:6]}"
                   if _wrt.st_src_off else "") + ")")
        # Why the writeback's release stopped, per turn: the visit's layer
        # end had not passed (end) or its writes were still landing (save).
        _tailseg = (f"tail(stop_end={_wrt.n_stop_end} "
                    f"stop_save={_wrt.n_stop_save} "
                    f"e2ev={_wrt.st_e2ev:.1f} ev2s={_wrt.st_ev2s:.1f} "
                    f"wakes_cq={_wrt._wait_n}) ")
        sys.stderr.write(
            f"[LOADER_TIMING] n={d['n']} wall={wall:.1f}s "
            # The loader's own time is everything it did not spend asleep.
            f"busy={wall - d['wait_t']:.1f}s "
            f"enter={_p.n_acquired} wkr(use-wait)={_p.t_use_wait:.1f}/{_p.n_use} "
            f"wait(n={d['wait_n']} t={d['wait_t']:.1f} cq={d['wait_cq']} "
            f"rel={d['wait_rel']}) "
            f"rio={d['rio']:.1f} rcompute={d['rcompute']:.1f} "
            # Occupancy integral (submitted minus already-posted CQEs),
            # time-weighted between samples. depth = what the drive held on
            # average, idle% = the share of the time it held nothing.
            f"occ(depth={d['occ_t']/max(d['occ_n'],1e-9):.2f} "
            f"idle={100*d['occ_z']/max(d['occ_n'],1e-9):.1f}% "
            f"cq={d['occ_cq']/max(d['occ_s'],1):.2f} n={d['occ_s']}) "
            + ("e2s(t=%.2f n=%d avg=%.1fms max=%.0fms e2b=%.2f) " % (
                d["e2s"], d["e2s_n"], 1e3 * d["e2s"] / max(1, d["e2s_n"]),
                1e3 * d["e2s_max"], d["e2b"]))
            + ("wl(it=%d ns/blk=%.0f) " % (
                d["wl_it"], 1e9 * d["wlist"] / max(1, d["wl_it"])))
            # The rotation: `passed` = visits the writeback has released,
            # `live` = visits live right now against n_slots, `moved` =
            # rebinds whose slot is not `layer % n_slots`.
            + ("slots(passed=%d live=%d/%d moved=%d) " % (
                _p.n_released, _p.live_visits(), _p.n_slots, _p.n_moved))
            + _tailseg
            + (f"kind(w={d['nk_w']}/{d['bk_w']/max(d['nk_w'],1)/2**20:.1f}MiB "
               f"a={d['nk_a']}/{d['bk_a']/max(d['nk_a'],1)/2**20:.1f}MiB "
               f"k={d['nk_k']}/{d['bk_k']/max(d['nk_k'],1)/2**20:.1f}MiB) ")
            + (f"bw(act_TB={d['b_act']/1e12:.2f} kv_TB={d['b_kv']/1e12:.2f} "
               f"w_TB={d['b_w']/1e12:.2f} "
               f"uni={(d['b_act']+d['b_kv'])/max(d['rio'],1e-9)/1e9:.2f}) ")
            + f"wlist={d['wlist']:.1f} dbuild={d['dbuild']:.1f} "
            f"staging(regions={_stg.n_slots} "
            f"region_MiB={_stg.act_buffer_bytes >> 20})"
            + _wrseg
            + f"{self._dev_mem_suffix()}\n")
        sys.stderr.flush()

    @staticmethod
    def _slice_kv_proj(layer_idx, slot, weight_table, num_kv_heads, head_dim):
        """Return (k_weight, v_weight, k_bias, v_bias) sliced out of
        the slot's weight buffer via the weight_table's manifest layout.

        The staged blob holds the fused qkv in vLLM's layout: dim 0 is
        [Q | K | V] concat. The bias follows the same split and must be
        sliced out too: kv_gen without it yields wrong K/V for every
        attention read of those blocks.

        The K/V part is sized from the cache's kv head count and Q takes the
        rest: q_size = total − 2·kv_size."""
        spec = weight_table.layers[layer_idx]
        weight_view = None
        bias_view = None
        # OPT fuses qkv as `self_attn.qkv_proj`; BLOOM as
        # `self_attention.query_key_value`. Both are staged in vLLM's
        # [Q|K|V]-on-dim0 layout, so the split below is identical.
        for p in spec.params:
            if p.name.endswith(("qkv_proj.weight", "query_key_value.weight")):
                sub = slot.t_buf[p.inner_offset:p.inner_offset + p.nbytes]
                weight_view = sub.view(_WEIGHT_DTYPE).view(p.shape)
            elif p.name.endswith(("qkv_proj.bias", "query_key_value.bias")):
                sub = slot.t_buf[p.inner_offset:p.inner_offset + p.nbytes]
                bias_view = sub.view(_WEIGHT_DTYPE).view(p.shape)
        total = int(weight_view.shape[0])
        kv_size = int(num_kv_heads) * int(head_dim)
        q_size = total - 2 * kv_size          # total == q_size + 2·kv_size
        k_w = weight_view[q_size:q_size + kv_size, :]
        v_w = weight_view[q_size + kv_size:, :]
        k_b = bias_view[q_size:q_size + kv_size]
        v_b = bias_view[q_size + kv_size:]
        return k_w, v_w, k_b, v_b
