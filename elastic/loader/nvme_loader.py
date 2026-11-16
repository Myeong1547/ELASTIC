"""ctypes binding for libnvme_loader.so.

Mirrors src/loader.h. The field order of every ctypes.Structure below MUST
stay in sync with it.

Every function is registered with explicit argtypes/restype. That is not
tidiness: ctypes defaults an unannotated integer argument to C `int`, so a
64-bit file offset or byte count would be silently truncated to 32 bits and
the call would read or write the wrong range without any error.
"""

from __future__ import annotations

import ctypes
import errno
import os
from collections import deque
from pathlib import Path

import numpy as np


class _CIovec(ctypes.Structure):
    """Mirrors `struct iovec` (<sys/uio.h>): {void* iov_base; size_t iov_len}."""
    _fields_ = [
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    ]


def _find_lib() -> Path:
    """Locate libnvme_loader.so at
    <project_root>/native_loader/build/libnvme_loader.so."""
    here = Path(__file__).resolve().parent       # elastic/loader
    project_root = here.parent.parent             # parent of elastic/
    lib = project_root / "native_loader" / "build" / "libnvme_loader.so"
    if lib.exists():
        return lib
    raise FileNotFoundError(
        f"{lib} not found. Build it with `make -C native_loader`.")


class _Lib:
    _instance = None

    @classmethod
    def get(cls) -> ctypes.CDLL:
        if cls._instance is None:
            lib = ctypes.CDLL(str(_find_lib()))

            lib.nvme_loader_alloc_buffer.restype = ctypes.c_void_p
            lib.nvme_loader_alloc_buffer.argtypes = [ctypes.c_size_t, ctypes.c_size_t]

            lib.nvme_loader_free_buffer.restype = None
            lib.nvme_loader_free_buffer.argtypes = [ctypes.c_void_p]

            # ---- nvme_engine -----------------------------------------
            lib.nvme_engine_start.restype = ctypes.c_void_p
            lib.nvme_engine_start.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint, ctypes.c_uint,          # read_ring, write_ring
                ctypes.c_uint,                         # read_qd
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.nvme_engine_attach_file.restype = ctypes.c_int
            lib.nvme_engine_attach_file.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.nvme_engine_stop.restype = None
            lib.nvme_engine_stop.argtypes = [ctypes.c_void_p]
            lib.nvme_engine_enqueue.restype = ctypes.c_int
            lib.nvme_engine_enqueue.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(_CEngineItem),
            ]
            lib.nvme_engine_write_drain.restype = None
            lib.nvme_engine_write_drain.argtypes = [ctypes.c_void_p]
            lib.nvme_engine_write_stats.restype = None
            lib.nvme_engine_write_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            lib.nvme_engine_reap.restype = ctypes.c_int
            lib.nvme_engine_reap.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_uint,
            ]
            lib.nvme_engine_out_fd.restype = ctypes.c_int
            lib.nvme_engine_out_fd.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nvme_engine_out_ready.restype = ctypes.c_uint
            lib.nvme_engine_out_ready.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_int]
            lib.nvme_engine_inflight.restype = ctypes.c_uint
            lib.nvme_engine_inflight.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_int]
            lib.nvme_engine_alignment.restype = ctypes.c_size_t
            lib.nvme_engine_alignment.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_int]
            lib.nvme_engine_file_size.restype = ctypes.c_uint64
            lib.nvme_engine_file_size.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_uint]
            lib.nvme_engine_queue_cap.restype = ctypes.c_uint
            lib.nvme_engine_queue_cap.argtypes = []
            lib.nvme_stream_signal_bind.restype = None
            lib.nvme_stream_signal_bind.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
            cls._instance = lib
        return cls._instance


_PY = None


def _py_lib() -> ctypes.PyDLL:
    """libnvme_loader.so through a PyDLL handle: calls made through it keep
    the GIL. Used only for the stream/event entry points — each is a short
    driver call in C that never calls back into Python — so the thread making
    one does not hand the interpreter to another thread and wait to get it
    back, thousands of times a step."""
    global _PY
    if _PY is None:
        py = ctypes.PyDLL(str(_find_lib()))
        py.nvme_stream_mark.restype = ctypes.c_int
        py.nvme_stream_mark.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        py.nvme_stream_event_new.restype = ctypes.c_void_p
        py.nvme_stream_event_new.argtypes = []
        py.nvme_event_record.restype = ctypes.c_int
        py.nvme_event_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        py.nvme_stream_wait.restype = ctypes.c_int
        py.nvme_stream_wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _PY = py
    return _PY


class StreamSignal:
    """Hear a stream reach a point instead of asking whether it has.

    `mark(ticket, stream)` adds a stream callback (loader.c
    `nvme_stream_mark`) on `stream`. The driver runs it once everything queued
    before it on that stream has completed: it raises `counter[0]` to the
    ticket and writes 1 to `fd`. So the moment a compute point is passed is
    the moment `fd` becomes readable, and `counter[0] >= t` is the fact the
    reader checks after waking. The writeback (WriteRouter) is the one
    listener: it has to hand writes to the drive, which the GPU cannot do, so
    it must hear the layer end on the host."""

    def __init__(self, fd: int, counter: np.ndarray):
        if counter.dtype != np.int64 or not counter.flags["C_CONTIGUOUS"]:
            raise TypeError("StreamSignal counter must be contiguous int64")
        self._counter = counter          # the C side keeps its address
        _Lib.get().nvme_stream_signal_bind(
            int(fd), counter.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        self._py = _py_lib()

    def mark(self, ticket: int, stream) -> None:
        """Mark `ticket` on `stream`."""
        t = int(ticket)
        rc = self._py.nvme_stream_mark(stream.cuda_stream, t)
        if rc != 0:
            raise RuntimeError(
                f"StreamSignal.mark failed (rc {rc}) for ticket {t}: the "
                f"tail would never hear this point pass")


class EventRing:
    """Timing-free CUDA events, reused round a ring, for ordering one stream
    after work on another on the device: `record(stream)` marks the end of
    what is queued there so far, `wait(stream, ev)` makes another stream wait
    for it. Neither makes the calling thread wait, and both keep the GIL.

    Reuse is safe at any ring size: a wait binds to the event's last record
    before the wait is issued, and a later record of the same event is for
    later work on the same in-order stream — the worst a wrap can do is make a
    wait cover more work than it needed, never less. The size only limits how
    often a wait over-covers."""

    def __init__(self, n: int = 4096):
        self._py = _py_lib()
        self._ev = []
        for _ in range(int(n)):
            ev = self._py.nvme_stream_event_new()
            if not ev:
                raise RuntimeError("EventRing: cuEventCreate failed")
            self._ev.append(int(ev))
        self._i = 0

    def record(self, stream) -> int:
        ev = self._ev[self._i % len(self._ev)]
        self._i += 1
        rc = self._py.nvme_event_record(ev, stream.cuda_stream)
        if rc != 0:
            raise RuntimeError(f"EventRing.record failed (rc {rc})")
        return ev

    @staticmethod
    def wait(stream, ev: int) -> None:
        rc = _py_lib().nvme_stream_wait(stream.cuda_stream, ev)
        if rc != 0:
            raise RuntimeError(f"EventRing.wait failed (rc {rc})")


class _CEngineItem(ctypes.Structure):
    """Mirrors `nvme_engine_item_t` (loader.h)."""
    _fields_ = [
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("buf", ctypes.c_void_p),
        ("iovcnt", ctypes.c_uint32),
        ("file_idx", ctypes.c_uint32),
        ("ud", ctypes.c_uint64),
    ]


def alloc_aligned(size: int, alignment: int = 4096) -> np.ndarray:
    """Return a uint8 numpy view over a page-aligned host buffer (the
    O_DIRECT grain and then some — `nvme_loader_alloc_buffer` raises the
    alignment to the page size).

    We return an ndarray subclass so we can stash the C allocation guard on
    the array; plain ndarray rejects arbitrary attributes.
    """
    lib = _Lib.get()
    p = lib.nvme_loader_alloc_buffer(
        ctypes.c_size_t(size), ctypes.c_size_t(alignment))
    if not p:
        raise MemoryError(f"alloc_buffer({size}) failed")
    addr = int(p)
    carr = (ctypes.c_uint8 * size).from_address(addr)
    base = np.frombuffer(carr, dtype=np.uint8)
    arr = base.view(_GuardedArray)
    arr._carr = carr
    arr._guard = _BufferGuard(lib, addr)
    return arr


def _src_addr_len(s) -> tuple[int, int]:
    """(address, nbytes) of any contiguous byte source: numpy uint8 views,
    ctypes arrays (`_uma_byte_view` returns these), memoryview/bytes-like.
    Zero-copy — np.frombuffer wraps the buffer protocol."""
    if isinstance(s, np.ndarray):
        if s.dtype != np.uint8 or not s.flags.c_contiguous:
            raise TypeError("ndarray src must be contiguous uint8")
        return int(s.ctypes.data), int(s.nbytes)
    v = np.frombuffer(s, dtype=np.uint8)
    return int(v.ctypes.data), int(v.nbytes)


class _BufferGuard:
    """RAII free helper attached to numpy arrays from alloc_aligned."""
    __slots__ = ("_lib", "_addr")

    def __init__(self, lib, addr: int):
        self._lib = lib
        self._addr = addr

    def __del__(self):
        if self._addr:
            self._lib.nvme_loader_free_buffer(ctypes.c_void_p(self._addr))
            self._addr = 0


class _GuardedArray(np.ndarray):
    """ndarray subclass that carries our C-side allocation guard.
    The guard frees the backing posix_memalign() buffer when the
    last reference is GC'd."""
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._carr = getattr(obj, "_carr", None)
        self._guard = getattr(obj, "_guard", None)


_ENGINE_READ = 0
_ENGINE_WRITE = 1


class NvmeEngine:
    """The NVMe engine — the device's only conversation partner.

    One C pthread (`nvme_engine` in loader.c) owns every io_uring call on the
    blob — one ring, two input FIFOs (read: KV/activation plus the attached
    weight blob; write), popped with write priority (writes only while an
    admitted batch exists or one is on the device, else reads), the read side
    kept topped up one-out-one-in to `read_qd` (ELASTIC_READ_QD) and the
    write side composed into SQEs of one ELASTIC_WRITE_INFLIGHT_MB unit, a
    new one filled only while less than one unit is in flight. Both
    directions are O_DIRECT. Python never touches a ring: the loader
    (ReadRouter) plans reads and interprets their completions, the writeback
    (WriteRouter) hands over writes and releases a slot once its writes have
    landed, and both talk to the device only through the two ports below.

    A port enqueues items into its side's engine FIFO (a memcpy and an
    eventfd kick) and takes completions off that side's output ring, heard
    on its fd. Short transfers are resubmitted inside the engine, so a
    reaped completion is full-length or an error — never a live remainder.

    Enqueue never blocks the caller: the submit-next-while-dispatching-
    current rotation lives in C, where a GIL slice cannot pause it."""

    def __init__(self, path: str | os.PathLike,
                 read_ring: int = 256, write_ring: int = 256,
                 read_qd: int = 0):
        lib = _Lib.get()
        if read_qd <= 0:
            read_qd = max(1, int(
                os.environ.get("ELASTIC_READ_QD", "8") or 8))
        # The write side takes its FIFO's requests in order; its fill
        # bound is in-flight bytes (ELASTIC_WRITE_INFLIGHT_MB, default
        # 32 MiB). The bound lives in the C thread and gates only the fill,
        # never the enqueue, and at least one item always fits, so it cannot
        # deadlock. Deep outstanding writes degrade how the blob is laid
        # down and how fast it reads back later.
        err = ctypes.c_int(0)
        h = lib.nvme_engine_start(
            os.fsencode(str(path)),
            ctypes.c_uint(int(read_ring)), ctypes.c_uint(int(write_ring)),
            ctypes.c_uint(int(read_qd)),
            ctypes.byref(err))
        if not h:
            code = -int(err.value) if err.value else errno.EIO
            raise OSError(code, os.strerror(code), str(path))
        self._h = ctypes.c_void_p(h)
        self._lib = lib
        self.read_port = EnginePort(self, _ENGINE_READ)
        self.write_port = (EnginePort(self, _ENGINE_WRITE)
                           if write_ring else None)

    def attach(self, path: str | os.PathLike) -> int:
        """Attach another backing file to the READ ring (the weight blob).
        Setup-time only, with nothing in flight."""
        err = ctypes.c_int(0)
        idx = self._lib.nvme_engine_attach_file(
            self._h, os.fsencode(str(path)), ctypes.byref(err))
        if idx < 0:
            code = -int(err.value) if err.value else errno.EIO
            raise OSError(code, os.strerror(code), str(path))
        return int(idx)

    def write_drain(self) -> None:
        """Open the write valve: admit everything enqueued so far. From then
        until the batch has landed the pop priority refills only writes;
        reads already on the device finish over the start of the batch, and
        no read is refilled until the batch has landed. Write admission is
        batch-grained — enqueued items sit in the engine FIFO until the
        producer calls this at its batch boundary."""
        if self._h:
            self._lib.nvme_engine_write_drain(self._h)

    def write_stats(self) -> "tuple[int, int]":
        """(items, sqes) on the write side: FIFO items fully submitted and
        the SQEs the engine composed them into. The engine turns the
        offset-ordered FIFO into SQEs of at most one unit
        (ELASTIC_WRITE_INFLIGHT_MB) of file-contiguous bytes — abutting
        items joined, an over-long item split — so sqes < items counts
        joins and sqes > items counts splits."""
        if not self._h:
            return self._final_wstats
        i = ctypes.c_uint64(0)
        q = ctypes.c_uint64(0)
        self._lib.nvme_engine_write_stats(self._h, ctypes.byref(i),
                                          ctypes.byref(q))
        return int(i.value), int(q.value)

    def close(self) -> None:
        """Stop the engine. What was on the device lands first; reads still
        queued are discarded (the STOP line counts them)."""
        if self._h:
            # The write totals outlive the engine: the write audit's exit
            # record reads them after the run's teardown has stopped it.
            self._final_wstats = self.write_stats()
            self._lib.nvme_engine_stop(self._h)
            self._h = ctypes.c_void_p(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EnginePort:
    """One side (read or write) of an NvmeEngine: `submit*` enqueues into
    that side's engine FIFO, completions come off its output ring and are
    heard on its fd. See NvmeEngine for what happens underneath."""

    def __init__(self, engine: NvmeEngine, side: int):
        self._e = engine
        self._lib = engine._lib
        self._side = side
        self.alignment = int(self._lib.nvme_engine_alignment(
            engine._h, ctypes.c_int(side)))
        self.size = int(self._lib.nvme_engine_file_size(
            engine._h, ctypes.c_uint(0)))
        # Items the side holds outstanding (enqueued, completion not yet
        # reaped); an enqueue past it is refused with EAGAIN.
        self.capacity = int(self._lib.nvme_engine_queue_cap())
        # reap scratch (single consumer per port by design)
        self._r_ud = (ctypes.c_uint64 * 64)()
        self._r_res = (ctypes.c_int64 * 64)()
        self._r_pend: deque = deque()

    # -- enqueue side ---------------------------------------------------

    def _enq(self, offset: int, size: int, buf_addr: int, iovcnt: int,
             file_idx: int, ud: int) -> None:
        it = _CEngineItem(int(offset), int(size),
                          ctypes.c_void_p(int(buf_addr)),
                          int(iovcnt), int(file_idx), int(ud))
        rc = self._lib.nvme_engine_enqueue(
            self._e._h, ctypes.c_int(self._side), ctypes.byref(it))
        if rc < 0:
            raise OSError(-rc, os.strerror(-rc))

    def submit_read(self, offset: int, size: int, dst: np.ndarray,
                    user_data: int = 0, file_idx: int = 0) -> None:
        if dst.dtype != np.uint8:
            raise TypeError("dst must be uint8 numpy array")
        self._enq(offset, size, dst.ctypes.data, 0, file_idx, user_data)

    def drain(self) -> None:
        """Valve trigger — admit everything queued so far to the device.
        Write-port operation: the producer calls it once its batch is whole."""
        self._e.write_drain()

    def submit_writev(self, offset: int, srcs, user_data: int = 0):
        """Gather write: `srcs` land at one contiguous file range from
        `offset`. The returned iovec array AND every source segment must
        outlive the completion — the engine references them
        asynchronously."""
        n = len(srcs)
        if n == 0:
            raise ValueError("submit_writev needs >=1 src")
        arr = (_CIovec * n)()
        total = 0
        for i, s in enumerate(srcs):
            addr, nb = _src_addr_len(s)
            arr[i].iov_base = addr
            arr[i].iov_len = nb
            total += nb
        self._enq(offset, total, ctypes.addressof(arr), n, 0, user_data)
        return arr

    # -- completion side ------------------------------------------------

    def try_wait_one(self):
        """(ud, res) from the output ring, or None. res is full-length or
        negative (or a short byte count only on an EOF the bounds checks
        make unreachable) — the engine already resubmitted every
        recoverable remainder."""
        if self._r_pend:
            return self._r_pend.popleft()
        n = self._lib.nvme_engine_reap(
            self._e._h, ctypes.c_int(self._side),
            self._r_ud, self._r_res, ctypes.c_uint(64))
        if n <= 0:
            return None
        for i in range(1, n):
            self._r_pend.append((int(self._r_ud[i]), int(self._r_res[i])))
        return int(self._r_ud[0]), int(self._r_res[0])

    def wait_one(self) -> tuple[int, int]:
        """Blocking pop: sleep on the output fd until a completion is there.
        For callers with nothing else to wait for (the units measurement);
        the loader and the writeback wait on their own fd sets."""
        import select as _select
        fd = self.completion_fd()
        while True:
            got = self.try_wait_one()
            if got is not None:
                return got
            _r, _, _ = _select.select([fd], (), ())
            for _f in _r:
                os.read(_f, 8)

    def cq_ready(self) -> int:
        """Completions the ENGINE has published that this port has not
        popped yet (plus any batch-reaped leftovers held locally)."""
        return len(self._r_pend) + int(self._lib.nvme_engine_out_ready(
            self._e._h, ctypes.c_int(self._side)))

    def completion_fd(self) -> int:
        return int(self._lib.nvme_engine_out_fd(
            self._e._h, ctypes.c_int(self._side)))

    def device_inflight(self) -> int:
        """What the device still holds on this side (engine-submitted minus
        engine-reaped; the engine reaps promptly)."""
        return int(self._lib.nvme_engine_inflight(
            self._e._h, ctypes.c_int(self._side)))

    def alloc_aligned(self, size: int) -> np.ndarray:
        return alloc_aligned(size, self.alignment)

    def attach(self, path: str | os.PathLike) -> int:
        return self._e.attach(path)
