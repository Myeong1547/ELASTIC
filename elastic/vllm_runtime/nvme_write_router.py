"""WriteRouter (`elastic-tail`): the writeback, whose visit pointer trails
the slot rotation.

The writeback does one thing, per visit in order: it waits for the visit's
completion events, sends the writes they license, waits for those to land,
then releases the visit and steps to the next — and the release is what
wakes the loader if every slot was live.

  * A completion event is a compute point on the forward pass's stream: a
    span's ACT at its LayerNorm, its KV at the attention seam, the layer
    end. The producer reserves a ticket, registers the writes it licenses,
    stamps the slot (for a layer end) and only then marks the ticket on its
    stream (`reserve` → `enqueue_writes` → `mark`). The mark is a stream
    callback the driver runs when the GPU gets there: it raises the progress
    counter and writes `_sig_fd`. So the licence to write is the completion
    itself, heard when it happens.
  * A write is sent the moment its ticket has passed — the NVMe engine gets
    the bytes as soon as they are final — in ticket order, which is the
    order the producer made them. Requests go to the NVMe engine as they are
    (`[(file offset, [views]), …]`); the NVMe engine composes the SQEs.
  * A visit is released once its layer end has passed and every write
    licensed up to it has landed: the copy-free save DMA-reads its source
    straight out of the visit's K and V buffers and activation buffer, so a
    write still landing is the drive reading the slot. Release =
    `release_tail` on the pool, then `notify_released` wakes the loader and
    the forward thread.

The writeback sleeps on two fds and nothing else, with no timeout: the
progress signal (a compute point passed) and the NVMe engine's
write-completion fd (a write landed). Nobody else wakes it — a producer
registering writes does not, the forward pass finishing with a slot does not
(its layer end is marked after the stamp). If a visit is never released,
its layer end or its writes never arrived, and that is where to look.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass(eq=False)
class _Desc:
    """One production event's write requests, one request per block-region
    range, in the order the producer made them. `kind` ∈ {kv_batch,
    act_batch} (the audit's names). `entries` = `[(file offset, [one view per
    block]), …]`: the range starts at the offset the layout gives its first
    block, and each view is a block's bytes wherever they are. The NVMe engine
    composes the SQEs; this router hands each request over as one NVMe
    engine item.

    Submission: once its own ticket has been passed (`gate_seq`, the
    compute point its bytes are final at — reserved before the descriptor is
    registered and marked on the stream after). Completion = all its CQEs
    reaped.
    """
    kind: str
    layer: int
    # The ticket of the compute point this batch's bytes are final at, on the
    # producing (compute) stream. The stream is in order, so
    # `router._pub_cnt[0] >= gate_seq` means the GPU has passed that point; the
    # counter is raised by the point's own stream callback, which also wakes
    # the writeback.
    gate_seq: int
    entries: list = field(default_factory=list)
    # Which producer made these requests (for the audit / phase counters).
    path: str = "?"
    # filled at submission:
    n_items: int = 0
    n_done: int = 0
    keepalive: list = field(default_factory=list)
    # build-once / resumable submission (a full NVMe engine FIFO can split a
    # batch across turns — already-submitted items must not be re-issued):
    writes: "Optional[list]" = None
    next_w: int = 0
    completed: bool = False
    # Write-service instrumentation: enqueue→submit and submit→CQE wall
    # times, aggregated on the router. t_ev is when the writeback first saw
    # the ticket passed, splitting e2s into "bytes not produced yet" (enq→ev)
    # and "produced but unsubmitted" (ev→sub), which have different owners
    # (GPU vs this router).
    t_enq: float = 0.0
    t_ev: float = 0.0
    t_sub: float = 0.0
    n_bytes: int = 0
    # Batch identity, for ELASTIC_WRITE_AUDIT only (which requests came from
    # one event).
    did: int = 0


class WriteRouter:
    """Every ELASTIC NVMe write: the writeback, whose visit pointer trails
    the rotation.

    `layout` is the blob's `BlockKVLayout` (every request's offset is its
    closed form). `ring` is the shared NVMe engine's (`nvme_engine`) write
    port: `submit_writev` appends to the NVMe engine's FIFO, completions come
    off its output ring, and short-write remainders never reach this router
    at all. `pool` is the slot pool whose visits this writeback releases.
    """

    def __init__(self, layout, kv_caches, engine, pool):
        self.layout = layout
        self.kv_caches = kv_caches
        # The write side of the one NVMe engine (loader.c `nvme_engine`).
        self.engine = engine
        self.ring = engine.write_port
        self._pool = pool
        # There is no router-wide lock. The state is split by owner:
        #
        #   _inbox      producers append (deque append is atomic); the
        #               writeback takes from the front
        #   _fifo       the writeback's alone: every registered descriptor,
        #               in ticket order, until it has landed and the release
        #               has looked past it
        #   _inflight   the writeback's alone: it is the only submitter and
        #               the only reaper
        self._inbox: "deque[_Desc]" = deque()
        self._fifo: "deque[_Desc]" = deque()
        self._sub = 0            # _fifo[:_sub] went to the NVMe engine
        self._inflight: dict[int, tuple] = {}   # ud -> (desc, off, want)
        self._next_ud = 1
        self._next_did = 0
        self.n_kv_saves = 0        # KV runs built
        self.n_act_saves = 0       # ACT runs built
        self.n_items = 0           # NVMe engine items handed over for them
        # Why the writeback's release stopped, counted per turn: the visit's
        # layer end had not passed yet, or a write it licensed was still
        # landing.
        self.n_stop_end = 0
        self.n_stop_save = 0
        # Aggregated write-service times: sums over completed descriptors of
        # (submit − enqueue) and (CQE-observe − submit), plus bytes. These are
        # sums of overlapping intervals, so they compare against each other,
        # never against wall-clock directly.
        self.st_e2s = 0.0
        self.st_e2ev = 0.0   # enqueue → ticket heard passed
        self.st_ev2s = 0.0   # ticket heard passed → handed to the NVMe engine
        self.st_s2c = 0.0
        self.st_ndesc = 0
        self.st_bytes = 0
        # Page alignment of the write source, counted rather than assumed.
        # O_DIRECT only requires the filesystem's probed minimum (512 here),
        # which is all the NVMe engine's enqueue check enforces, but on this
        # drive a file written from a page-misaligned source reads back
        # slower for its whole lifetime, and the KV/ACT blob is written once
        # and re-read every decode step.
        self.st_src_n = 0
        self.st_src_bad = 0
        # Phase mismatch: source address mod 4096 against file offset mod
        # 4096, counted next to the source's absolute alignment above.
        self.st_ph_n = 0
        self.st_ph_bad = 0
        # {write path: phase mismatches}.
        self.st_ph_by: dict = {}
        # (descriptor kind, source address mod 4096) -> count, so a misaligned
        # source can be traced to the buffer it came from instead of guessed.
        self.st_src_off: dict = {}
        # Compute-stream progress. The counter is raised by the stream
        # callback a producer marks its stream with (`mark`), the same call
        # that makes `_sig_fd` readable — so the writeback learns a point has
        # passed when it passes, and reads the fact with a plain load.
        self._pub_cnt = np.zeros(1, dtype=np.int64)
        self._pub_ticket = 0
        self._sig_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        from elastic.loader.nvme_loader import StreamSignal
        self._signal = StreamSignal(self._sig_fd, self._pub_cnt)
        # The stream the producer marks on. Tickets rise in the producer's
        # program order and the counter is a max, so every mark rides that
        # one stream (a switch to a new stream is handed off by an event —
        # streaming_loader `_use_worker_priority_stream`); the stop ticket
        # rides it too, behind the last layer end.
        self._mark_stream = None
        self._stop_ticket: Optional[int] = None
        # How many turns a landed write, rather than a passed compute point,
        # woke.
        self._wait_n = 0
        # ELASTIC_WRITE_AUDIT=1 records every NVMe engine item at submit
        # ("S") and, at exit, the NVMe engine's own item and SQE counts
        # ("E"), written to ELASTIC_WRITE_AUDIT_OUT.
        self._audit_on = os.environ.get("ELASTIC_WRITE_AUDIT", "0") == "1"
        # The service-time and source-phase counters (`_account_built`) are
        # read only by the ELASTIC_LOADER_TIMING beacons, so they are kept
        # only then.
        self._timing = os.environ.get("ELASTIC_LOADER_TIMING") == "1"
        self._audit_log: list = []
        if self._audit_on:
            import atexit
            import json as _json

            def _dump():
                out = os.environ.get("ELASTIC_WRITE_AUDIT_OUT",
                                     "/tmp/write_audit.json")
                try:
                    # What the device actually saw: the NVMe engine's
                    # composed SQE count next to the items this router
                    # handed it.
                    _items, _sqes = self.engine_write_stats()
                    with open(out, "w") as fh:
                        _json.dump(self._audit_log + [["E", _items, _sqes]],
                                   fh)
                except Exception:
                    pass
            atexit.register(_dump)
        # Started last in __init__, so the writeback's first turn can never
        # see a half-built router.
        self._feeder = threading.Thread(
            target=self._feed_main, daemon=True, name="elastic-tail")
        self._feeder.start()

    # ---------------- producers (the forward thread) ------------------

    def engine_write_stats(self) -> "tuple[int, int]":
        """(items, sqes) of the shared NVMe engine's write side — see
        `NvmeEngine.write_stats`."""
        return self.engine.write_stats()

    def kv_block_views(self, layer: int, cb: int):
        """(K view, V view) of cache block `cb` of `layer` — the slot's K and
        V buffers themselves are the DMA source; the slot's release condition
        holds them until the CQEs land."""
        from elastic.runtime.block_kv_storage import _uma_byte_view
        cache = self.kv_caches[layer]
        return (_uma_byte_view(cache[0, cb]), _uma_byte_view(cache[1, cb]))

    def enqueue_writes(self, layer: int, kind: str, entries: list,
                       gate_seq: int, path: str = "?") -> None:
        """One production event's requests, `[(file offset, [views]), …]`,
        in the producer's order, licensed by `gate_seq` — a ticket from
        `reserve` that the producer marks on its stream right after this call
        (`mark`). Registering before marking is what makes the mark find the
        descriptor already here. Nothing is copied or recycled: a view is the
        block's bytes where they are, and the visit is not released until
        these land. `entries` is never empty: a producer with nothing to
        write registers nothing."""
        d = _Desc(kind=kind, layer=layer, gate_seq=int(gate_seq),
                  entries=list(entries), path=path)
        d.t_enq = time.perf_counter()
        self._next_did += 1
        d.did = self._next_did          # audit identity only (see _Desc.did)
        # No lock and no wake: the writeback hears about the descriptor when
        # its ticket's mark fires, which the producer launches after this.
        self._inbox.append(d)

    def enqueue_marked(self, layer: int, kind: str, entries: list,
                       path: str = "?") -> None:
        """`reserve` → `enqueue_writes` → `mark`, for a producer whose
        requests are the only thing its compute point licenses (a span's ACT
        at its LN, its KV at the attn seam, the x-wheel's)."""
        t = self.reserve()
        self.enqueue_writes(layer, kind, entries, gate_seq=t, path=path)
        self.mark(t)

    def reserve(self) -> int:
        """The next ticket. A producer reserves it, registers what the ticket
        licenses (`enqueue_writes`, a slot's `done_seq`), and only then marks
        it — so when the mark fires, everything it licenses is already here.
        Producers are the forward thread alone, so tickets rise in stream
        order."""
        self._pub_ticket += 1
        return self._pub_ticket

    def mark(self, ticket: int, stream=None) -> None:
        """Mark `ticket` on `stream` (default: the caller's current stream):
        after everything already queued there, the driver raises the progress
        counter to it and the writeback hears it (`StreamSignal`)."""
        if stream is None:
            stream = torch.cuda.current_stream()
        self._mark_stream = stream
        self._signal.mark(int(ticket), stream=stream)

    # ---------------- the writeback -----------------------------------

    def _feed_main(self) -> None:
        """The writeback: sleep until a compute point passes or a write
        lands, take in what that changed — send what it licensed, account
        what landed, release every visit that is now finished — and sleep
        again.
        Ends once the stop ticket has passed and every write has landed."""
        import select as _select
        try:
            cq_fd = int(self.ring.completion_fd())
            fds = [self._sig_fd, cq_fd]
            while True:
                _r, _, _ = _select.select(fds, (), ())
                # Drain what woke us: an eventfd counter left set would make
                # the next select return at once. Anything posted after this
                # read re-arms it, so nothing is missed.
                for _fd in _r:
                    if _fd == cq_fd:
                        self._wait_n += 1
                    os.read(_fd, 8)
                if self._turn():
                    break
        finally:
            # The writeback is done: at teardown this is what ends the
            # loader's last wait (it reads its stop flag).
            self._pool.notify_released()

    def _turn(self) -> bool:
        """One wake's worth of work. Returns True when the writeback is
        done."""
        # The counter first, then the registrations: a ticket read here as
        # passed had everything it licenses registered before it was marked.
        passed = int(self._pub_cnt[0])
        while self._inbox:
            self._fifo.append(self._inbox.popleft())
        self._reap()
        self._send(passed)
        self._release(passed)
        return (self._stop_ticket is not None
                and passed >= self._stop_ticket
                and not self._fifo and not self._inbox
                and not self._inflight)

    def _reap(self) -> None:
        """Account every write completion the NVMe engine has posted."""
        while True:
            got = self.ring.try_wait_one()
            if got is None:
                return
            ud, res = got
            desc, off, want = self._inflight.pop(ud)
            if res < 0:
                # No bytes landed: the reload would serve the file's previous
                # content back as saved state.
                raise RuntimeError(
                    f"write failed kind={desc.kind} L={desc.layer} off={off} "
                    f"errno={-res} ({os.strerror(-res)})")
            if res != want:
                # The NVMe engine owns short-write remainders and otherwise
                # delivers full length or an error, so a short one here is an
                # EOF the bounds checks make unreachable.
                raise RuntimeError(
                    f"short write kind={desc.kind} L={desc.layer} off={off} "
                    f"{res}/{want}")
            desc.n_done += 1
            if (desc.n_done >= desc.n_items
                    and desc.next_w >= len(desc.writes)):
                self._complete(desc)

    def _send(self, passed: int) -> None:
        """Hand the NVMe engine every registered write whose ticket has
        passed, in ticket order. Tickets rise in stream order, so the first
        one that has not passed ends the walk. A full NVMe engine FIFO
        (-EAGAIN) leaves the descriptor resumable; a landing write frees room
        and wakes the writeback."""
        fifo = self._fifo
        while self._sub < len(fifo):
            d = fifo[self._sub]
            if d.gate_seq > passed:
                return
            if not d.t_ev:
                d.t_ev = time.perf_counter()
            full = self._submit(d)
            if full:
                return
            self._sub += 1

    def _release(self, passed: int) -> None:
        """Release, in visit order, every visit whose layer end has passed
        and whose writes have landed.

        The layer end is the slot's own stamp: `forward_done_visit` says the
        stamp is this visit's (the forward pass writes `done_seq` first, then
        `forward_done_visit`), and `done_seq` is the ticket its layer end was
        marked with.
        The writes are the FIFO's front: descriptors sit in ticket order and
        every write a visit licensed carries a ticket no later than its layer
        end, so once the oldest write still landing carries a later ticket,
        nothing of this visit's is left in flight."""
        fifo = self._fifo
        while fifo and fifo[0].completed:
            fifo.popleft()
            self._sub -= 1
        pool = self._pool
        n = 0
        while True:
            ts = pool.tail_slot()
            if ts is None:
                break
            t, s = ts
            if s.forward_done_visit < t or passed < s.done_seq:
                self.n_stop_end += 1
                break
            if fifo and fifo[0].gate_seq <= s.done_seq:
                self.n_stop_save += 1
                break
            pool.release_tail()
            n += 1
        if n:
            pool.notify_released()

    def _complete(self, desc: _Desc) -> None:
        desc.completed = True
        if desc.t_sub:
            self.st_s2c += time.perf_counter() - desc.t_sub
            self.st_bytes += desc.n_bytes
            self.st_ndesc += 1
        desc.keepalive.clear()

    # ---------------- stop ---------------------------------------------

    def stop(self) -> None:
        """Teardown, on the forward thread: mark a stop ticket behind the
        last layer end, on the stream the layer ends are marked on. The
        writeback hears it like any compute point, lands every write still
        owed and exits; this joins it — no timeout, because what it waits
        for is the drive finishing writes it has already been handed."""
        if os.environ.get("ELASTIC_LOADER_TIMING") == "1":
            import sys as _sys
            print(f"[WRITE_ROUTER] n_kv={self.n_kv_saves} "
                  f"n_act={self.n_act_saves} items={self.n_items} "
                  f"wr(desc={self.st_ndesc} e2s={self.st_e2s:.1f}s "
                  f"s2c={self.st_s2c:.1f}s {self.st_bytes/1e9:.1f}GB) "
                  f"src_page_misaligned={self.st_src_bad}/{self.st_src_n}",
                  file=_sys.stderr, flush=True)
        th = self._feeder
        if th is not None and th.is_alive():
            t = self.reserve()
            self._stop_ticket = t
            self.mark(t, stream=self._mark_stream)
            th.join()
        self._feeder = None
        # `_sig_fd` stays open: the C signal holds its number, and a mark's
        # callback raises the counter before it writes the fd — the writeback
        # can exit on the stop ticket between the two — so a closed number
        # could already belong to some other file when that write comes.

    # ---------------- submission arithmetic ---------------------------

    def _account_built(self, d: _Desc) -> None:
        """Once-per-descriptor accounting at build time: bytes, the source
        page-alignment / phase counters, and the enqueue→submit split.

        The phase counters count BLOCKS, not iovec segments. An iovec segment
        is a run of blocks (source and file both stride by `kb`, so a run is in
        phase iff its first block is), so `srcmis`/`phmis` stay the same
        ratio however the blocks are grouped into segments."""
        items = []
        for off, bufs in d.writes:
            o = int(off)
            for b in bufs:
                v = (b if isinstance(b, np.ndarray)
                     else np.frombuffer(b, dtype=np.uint8))
                items.append((o, int(v.nbytes), int(v.ctypes.data)))
                o += int(v.nbytes)
        _kb = self.layout.block_bytes
        for _it in items:
            _nblk = max(1, _it[1] // _kb)
            self.st_src_n += _nblk
            self.st_ph_n += _nblk
            if (_it[2] & 4095) != (_it[0] & 4095):
                self.st_ph_bad += _nblk
                _p = d.path if d.kind == "act_batch" else "kv"
                self.st_ph_by[_p] = self.st_ph_by.get(_p, 0) + _nblk
            if _it[2] & 4095:
                self.st_src_bad += _nblk
                self.st_src_off[(d.kind, _it[2] & 4095)] = \
                    self.st_src_off.get((d.kind, _it[2] & 4095), 0) + 1
        d.t_sub = time.perf_counter()
        d.n_bytes = sum(it[1] for it in items)
        self.st_e2s += d.t_sub - d.t_enq
        self.st_e2ev += (d.t_ev or d.t_sub) - d.t_enq
        self.st_ev2s += d.t_sub - (d.t_ev or d.t_sub)

    def _submit(self, d: _Desc) -> bool:
        """Expand a licensed descriptor into NVMe engine items (its requests,
        taken once and cached on the descriptor) and hand every one from
        d.next_w to the NVMe engine's write FIFO. Returns True when the FIFO
        refused (-EAGAIN) part-way — the descriptor resumes from d.next_w
        once a write lands. A request that cannot be handed over is a block
        the blob will not hold, so it raises."""
        import errno as _errno
        if d.writes is None:
            d.writes = [(int(o), list(bufs)) for o, bufs in d.entries]
            if d.kind == "kv_batch":
                self.n_kv_saves += 1
            else:
                self.n_act_saves += 1
            self.n_items += len(d.writes)
            if self._timing:
                self._account_built(d)
        n_now = 0
        _audit = self._audit_on
        while d.next_w < len(d.writes):
            off, bufs = d.writes[d.next_w]
            ud = self._next_ud
            self._next_ud += 1
            try:
                iov = self.ring.submit_writev(off, bufs, user_data=ud)
            except OSError as e:
                if e.errno == _errno.EAGAIN:
                    # NVMe engine FIFO full — admit what is queued so the
                    # FIFO can empty, then resume this descriptor when a
                    # write lands.
                    self.ring.drain()
                    return True
                raise RuntimeError(
                    f"write submit failed kind={d.kind} L={d.layer} "
                    f"off={off}: {e}") from e
            d.n_items += 1
            d.next_w += 1
            n_now += 1
            d.keepalive.append((iov, bufs))
            want = sum(
                (b.nbytes if isinstance(b, np.ndarray)
                 else np.frombuffer(b, dtype=np.uint8).nbytes)
                for b in bufs)
            self._inflight[ud] = (d, off, int(want))
            if _audit:
                # ["S", kind, layer, offset, bytes, iovec segments,
                #  user data, event id, path]
                self._audit_log.append(
                    ["S", d.kind, d.layer, int(off), int(want),
                     len(bufs), ud, d.did, d.path])
        # Batch boundary: the descriptor's walk is complete, so admit it
        # (`drain`). Everything above accumulated in the NVMe engine FIFO
        # and goes to the device now, together — never at the walk's own
        # pace.
        if n_now:
            self.ring.drain()
        return False

