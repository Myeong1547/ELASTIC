"""The NVMe engine (nvme_engine) round-trips through the two ports.

The NVMe engine is the process's only io_uring caller: python enqueues
items on per-side FIFOs and takes (ud, res) off per-side output rings. These
tests cover the surface the loader (ReadRouter) and the writeback
(WriteRouter) actually use — single reads, gather writes (one view or
several), the completion fd, FIFO capacity beyond the refill target,
synchronous validation errors, and buffer-protocol write sources.

Every read-back goes through the NVMe engine's own read side too, so a write
completion is never trusted about where the bytes landed.
"""

import ctypes
import errno
import os
import select

import numpy as np
import pytest

from elastic.loader.nvme_loader import NvmeEngine


@pytest.fixture
def blob(tmp_path):
    p = tmp_path / "etest.bin"
    with open(p, "wb") as f:
        f.truncate(4 << 20)
    return str(p)


@pytest.fixture
def eng(blob):
    e = NvmeEngine(blob, read_ring=8, write_ring=8, read_qd=4)
    yield e
    e.close()


def test_write_then_read_roundtrip(eng):
    w, r = eng.write_port, eng.read_port
    keep = []
    buf = w.alloc_aligned(8192)
    buf[:] = np.random.randint(0, 256, 8192, dtype=np.uint8)
    src = buf.copy()
    keep.append(w.submit_writev(4096, [buf], user_data=7))
    w.drain()
    assert w.wait_one() == (7, 8192)
    dst = r.alloc_aligned(8192)
    r.submit_read(4096, 8192, dst,
                  user_data=11)
    assert r.wait_one() == (11, 8192)
    assert bytes(dst) == bytes(src)


def test_gather_writev_and_ctypes_source(eng):
    w, r = eng.write_port, eng.read_port
    s1 = w.alloc_aligned(4096)
    s1[:] = 1
    raw2 = w.alloc_aligned(4096)
    raw2[:] = 2
    # buffer-protocol source (the shape a zero-copy tensor byte view takes)
    c2 = (ctypes.c_ubyte * 4096).from_address(raw2.ctypes.data)
    iov = w.submit_writev(65536, [s1, c2], user_data=3)
    w.drain()
    assert w.wait_one() == (3, 8192)
    del iov
    dst = r.alloc_aligned(8192)
    r.submit_read(65536, 8192, dst,
                  user_data=4)
    assert r.wait_one() == (4, 8192)
    assert dst[:4096].tolist() == [1] * 4096
    assert dst[4096:].tolist() == [2] * 4096


def test_backlog_past_refill_target_all_complete(eng):
    """32 reads against target 4: the FIFO holds the backlog and the NVMe
    engine's one-out-one-in drains it completely, ud-exact."""
    r = eng.read_port
    dsts = [r.alloc_aligned(4096) for _ in range(32)]
    for i, d in enumerate(dsts):
        r.submit_read(4096 * (i % 8), 4096, d,
                      user_data=100 + i)
    got = set()
    for _ in range(32):
        ud, res = r.wait_one()
        assert res == 4096
        got.add(ud)
    assert got == set(range(100, 132))
    assert r.device_inflight() == 0
    assert r.cq_ready() == 0


def test_completion_fd_wakes(eng):
    r = eng.read_port
    fd = r.completion_fd()
    assert fd >= 0
    d = r.alloc_aligned(4096)
    r.submit_read(0, 4096, d,
                  user_data=42)
    rr, _, _ = select.select([fd], [], [], 2.0)
    assert rr, "completion fd never signalled"
    os.read(fd, 8)
    assert r.try_wait_one() == (42, 4096)


def test_validation_is_synchronous(eng):
    r, w = eng.read_port, eng.write_port
    d = r.alloc_aligned(4096)
    with pytest.raises(OSError):
        r.submit_read(1, 4096, d,
                      user_data=1)                     # unaligned offset
    with pytest.raises(OSError):
        r.submit_read(0, 8 << 20, d,
                      user_data=1)                     # past EOF
    d2 = r.alloc_aligned(2 * r.alignment)
    with pytest.raises(OSError) as ex:
        r.submit_read(0, r.alignment + 1, d2,
                      user_data=1)                     # size not aligned
    assert ex.value.errno == errno.EINVAL
    s = w.alloc_aligned(4096)
    with pytest.raises(OSError):
        w.submit_writev(4 << 20, [s], user_data=1)   # past backing size


def test_ports_share_the_file(eng):
    """Both ports report the same backing file."""
    assert eng.read_port.size == eng.write_port.size == 4 << 20
    assert eng.read_port.alignment >= 512


def test_write_valve_holds_until_drain(eng):
    """Write admission is batch-grained: enqueued items do not reach the
    device until the producer opens the valve, then they all go together."""
    import time
    w, r = eng.write_port, eng.read_port
    keep = []
    bufs = []
    for i in range(4):
        b = w.alloc_aligned(4096)
        b[:] = np.full(4096, 16 + i, dtype=np.uint8)
        bufs.append(b)
        keep.append(w.submit_writev(i * 4096, [b], user_data=100 + i))
    time.sleep(0.3)
    assert w.try_wait_one() is None      # valve closed: nothing completed
    w.drain()
    got = sorted(w.wait_one() for _ in range(4))
    assert got == [(100 + i, 4096) for i in range(4)]
    dst = r.alloc_aligned(16384)
    r.submit_read(0, 16384, dst,
                  user_data=9)
    assert r.wait_one() == (9, 16384)
    for i in range(4):
        assert bytes(dst[i * 4096:(i + 1) * 4096]) == bytes(bufs[i])


def test_write_valve_snapshot_excludes_later_items(eng):
    """drain() admits what is queued at the call; an item enqueued after it
    waits for its own drain (a batch boundary is a queue position, not a
    sticky open state)."""
    import time
    w = eng.write_port
    keep = []
    a = w.alloc_aligned(4096)
    a[:] = 1
    keep.append(w.submit_writev(0, [a], user_data=1))
    w.drain()
    assert w.wait_one() == (1, 4096)
    b = w.alloc_aligned(4096)
    b[:] = 2
    keep.append(w.submit_writev(4096, [b], user_data=2))
    time.sleep(0.2)
    assert w.try_wait_one() is None
    w.drain()
    assert w.wait_one() == (2, 4096)


def test_read_side_has_no_valve(eng):
    """Reads must flow with no drain anywhere in the path."""
    w, r = eng.write_port, eng.read_port
    keep = []
    b = w.alloc_aligned(4096)
    b[:] = 7
    keep.append(w.submit_writev(0, [b], user_data=1))
    w.drain()
    assert w.wait_one() == (1, 4096)
    d = r.alloc_aligned(4096)
    r.submit_read(0, 4096, d, user_data=2)
    assert r.wait_one() == (2, 4096)


# ---- write-priority schedule: one ring, pop priority ----

def test_write_priority_holds_queued_reads_until_batch_lands(blob):
    """Reads enqueued after a drain pop only once the entire admitted write
    batch has landed, so every one of them observes the batch's bytes even
    when the write covering the read range is the batch's last item. Without
    the pop priority they run concurrently with the filler writes, complete
    before the last item lands, and read the old zeros."""
    e = NvmeEngine(blob, read_ring=8, write_ring=8, read_qd=2)
    try:
        w, r = e.write_port, e.read_port
        keep = []
        n_reads, rsz = 64, 65536
        # batch: 7 x 512 KiB filler writes elsewhere, then 0xAB over the
        # read range as the batch's last item
        fill = w.alloc_aligned(512 * 1024)
        fill[:] = 5
        for j in range(7):
            keep.append(w.submit_writev(262144 + j * 512 * 1024, [fill],
                           user_data=2000 + j))
        mark = w.alloc_aligned(rsz)
        mark[:] = 0xAB
        keep.append(w.submit_writev(0, [mark], user_data=2100))
        w.drain()
        # every read below is enqueued while the batch is admitted, so the
        # schedule owes it a post-batch pop: 64/64 must see the mark
        dsts = [r.alloc_aligned(rsz) for _ in range(n_reads)]
        for i in range(n_reads):
            r.submit_read(0, rsz,
                          dsts[i], user_data=1000 + i)
        for _ in range(8):
            ud, res = w.wait_one()
            assert res > 0
        for i in range(n_reads):
            ud, res = r.wait_one()
            assert res == rsz
        got_mark = sum(1 for d in dsts
                       if d[0] == 0xAB and d[-1] == 0xAB)
        assert got_mark == n_reads, got_mark
    finally:
        e.close()


def test_drain_without_writes_is_inert(blob):
    e = NvmeEngine(blob, read_ring=4, write_ring=4, read_qd=2)
    try:
        w, r = e.write_port, e.read_port
        w.drain()                        # empty valve: no phase
        d = r.alloc_aligned(4096)
        r.submit_read(0, 4096, d,
                      user_data=1)
        assert r.wait_one() == (1, 4096)
    finally:
        e.close()


def test_alternating_batches_no_deadlock(blob):
    e = NvmeEngine(blob, read_ring=8, write_ring=8, read_qd=4)
    try:
        w, r = e.write_port, e.read_port
        keep = []
        wb = w.alloc_aligned(4096)
        wb[:] = 3
        rb = [r.alloc_aligned(4096) for _ in range(8)]
        total_r = total_w = 0
        for round_ in range(20):
            for i in range(8):
                r.submit_read(4096 * i, 4096, rb[i],
                              user_data=round_ * 100 + i)
            for j in range(2):
                keep.append(w.submit_writev(4096 * (8 + j), [wb],
                               user_data=round_ * 100 + 50 + j))
            w.drain()
            for _ in range(8):
                _, res = r.wait_one()
                assert res == 4096
                total_r += 1
            for _ in range(2):
                _, res = w.wait_one()
                assert res == 4096
                total_w += 1
        assert (total_r, total_w) == (160, 40)
    finally:
        e.close()


def test_priority_drops_after_batch_and_new_writes_wait(blob):
    """After an admitted batch fully lands the pop priority drops: writes
    enqueued later (no drain yet) accumulate un-admitted and reads flow past
    them; their own drain raises the priority again."""
    e = NvmeEngine(blob, read_ring=8, write_ring=8, read_qd=2)
    try:
        w, r = e.write_port, e.read_port
        keep = []
        a = w.alloc_aligned(4096)
        a[:] = 1
        keep.append(w.submit_writev(0, [a], user_data=1))
        w.drain()
        assert w.wait_one() == (1, 4096)          # batch landed, phase over
        b = w.alloc_aligned(4096)
        b[:] = 2
        # no drain: not admitted
        keep.append(w.submit_writev(4096, [b], user_data=2))
        for i in range(8):                        # reads flow past it
            d = r.alloc_aligned(4096)
            r.submit_read(0, 4096, d,
                          user_data=100 + i)
            ud, res = r.wait_one()
            assert res == 4096
        assert w.try_wait_one() is None           # still waiting in the FIFO
        w.drain()                                 # its own trigger
        assert w.wait_one() == (2, 4096)
    finally:
        e.close()


# ---- write-side compose: one unit per SQE, whatever the request grouping --

@pytest.fixture
def blob16(tmp_path):
    p = tmp_path / "unit.bin"
    with open(p, "wb") as f:
        f.truncate(16 << 20)
    return str(p)


@pytest.fixture
def eng_unit1(blob16, monkeypatch):
    """An NVMe engine whose write unit is 1 MiB (ELASTIC_WRITE_INFLIGHT_MB=1),
    so the compose rules can be exercised with small buffers."""
    monkeypatch.setenv("ELASTIC_WRITE_INFLIGHT_MB", "1")
    e = NvmeEngine(blob16, read_ring=8, write_ring=8, read_qd=4)
    yield e
    e.close()


def _readback(eng, off, n):
    r = eng.read_port
    dst = r.alloc_aligned(n)
    r.submit_read(off, n, dst,
                  user_data=999)
    assert r.wait_one() == (999, n)
    return bytes(dst)


def test_abutting_items_are_joined_into_one_unit_sqe(eng_unit1):
    """8 requests of 64 KiB at consecutive offsets, however they were
    grouped upstream, go to the device as ONE writev of 512 KiB; every
    request still completes on its own with its own ud and size."""
    e = eng_unit1
    w = e.write_port
    keep = []
    bufs = []
    for i in range(8):
        b = w.alloc_aligned(65536)
        b[:] = np.full(65536, 10 + i, dtype=np.uint8)
        bufs.append(b)
        keep.append(w.submit_writev(i * 65536, [b], user_data=200 + i))
    w.drain()
    got = sorted(w.wait_one() for _ in range(8))
    assert got == [(200 + i, 65536) for i in range(8)]
    assert e.write_stats() == (8, 1)
    back = _readback(e, 0, 8 * 65536)
    for i in range(8):
        assert back[i * 65536:(i + 1) * 65536] == bytes(bufs[i])


def test_an_item_longer_than_the_unit_is_split_and_completes_once(eng_unit1):
    """A 3 MiB request becomes three 1 MiB SQEs and exactly one
    completion, of the full size, after the last piece lands."""
    e = eng_unit1
    w = e.write_port
    keep = []
    n = 3 << 20
    b = w.alloc_aligned(n)
    b[:] = np.arange(n, dtype=np.uint32).view(np.uint8)[:n]
    keep.append(w.submit_writev(1 << 20, [b], user_data=7))
    w.drain()
    assert w.wait_one() == (7, n)
    assert w.try_wait_one() is None
    assert e.write_stats() == (1, 3)
    assert _readback(e, 1 << 20, n) == bytes(b)


def test_a_unit_boundary_falls_inside_an_item(eng_unit1):
    """768 KiB + 768 KiB abutting: SQE 1 = the first item plus 256 KiB of
    the second, SQE 2 = the rest. Both items complete whole, in order of
    their last byte landing."""
    e = eng_unit1
    w = e.write_port
    keep = []
    n = 768 << 10
    a = w.alloc_aligned(n); a[:] = 0xA1
    b = w.alloc_aligned(n); b[:] = 0xB2
    keep.append(w.submit_writev(0, [a], user_data=1))
    keep.append(w.submit_writev(n, [b], user_data=2))
    w.drain()
    got = [w.wait_one() for _ in range(2)]
    assert sorted(got) == [(1, n), (2, n)]
    assert e.write_stats() == (2, 2)
    back = _readback(e, 0, 2 * n)
    assert back[:n] == bytes(a) and back[n:] == bytes(b)


def test_items_that_do_not_abut_are_not_joined(eng_unit1):
    e = eng_unit1
    w = e.write_port
    keep = []
    a = w.alloc_aligned(4096); a[:] = 1
    b = w.alloc_aligned(4096); b[:] = 2
    c = w.alloc_aligned(4096); c[:] = 3
    keep.append(w.submit_writev(0, [a], user_data=1))
    keep.append(w.submit_writev(2 << 20, [b], user_data=2))      # gap
    keep.append(w.submit_writev((2 << 20) + 4096, [c],           # abuts b
                                user_data=3))
    w.drain()
    got = sorted(w.wait_one() for _ in range(3))
    assert got == [(1, 4096), (2, 4096), (3, 4096)]
    assert e.write_stats() == (3, 2)
    assert _readback(e, 0, 4096) == bytes(a)
    assert _readback(e, 2 << 20, 8192) == bytes(b) + bytes(c)


def test_a_gathered_request_is_split_at_the_unit_across_its_pieces(eng_unit1):
    """A writev of four 512 KiB views (2 MiB) with a 1 MiB unit: two SQEs,
    the cut between the second and third view; one completion."""
    e = eng_unit1
    w = e.write_port
    parts = []
    for i in range(4):
        v = w.alloc_aligned(512 << 10)
        v[:] = 0x40 + i
        parts.append(v)
    iov = w.submit_writev(4 << 20, parts, user_data=5)
    w.drain()
    assert w.wait_one() == (5, 2 << 20)
    del iov
    assert e.write_stats() == (1, 2)
    back = _readback(e, 4 << 20, 2 << 20)
    for i in range(4):
        assert back[i * (512 << 10):(i + 1) * (512 << 10)] == bytes(parts[i])


def test_a_gathered_request_is_split_inside_a_piece(eng_unit1):
    """A writev whose view boundaries do not fall on the unit: three 768 KiB
    views written at 0 with a 1 MiB unit. The cuts at 1 and 2 MiB fall
    inside views 1 and 2, so the request goes out as three SQEs and
    completes once."""
    e = eng_unit1
    w = e.write_port
    parts = []
    for i in range(3):
        v = w.alloc_aligned(768 << 10)                  # 3 x 768 KiB
        v[:] = 0x70 + i
        parts.append(v)
    iov = w.submit_writev(0, parts, user_data=6)          # cut at 1 MiB,
    w.drain()                                              # inside view 1
    assert w.wait_one() == (6, 3 * (768 << 10))
    del iov
    assert e.write_stats() == (1, 3)
    back = _readback(e, 0, 3 * (768 << 10))
    for i in range(3):
        assert back[i * (768 << 10):(i + 1) * (768 << 10)] == bytes(parts[i])


def test_a_request_with_more_views_than_iov_max_is_cut_at_a_view_edge(
        blob16, monkeypatch):
    """The NVMe engine owns the kernel's writev limit too: a request of 2048
    views inside one 8 MiB unit goes out as two SQEs of IOV_MAX (1024)
    views each, cut at a view edge, and completes once."""
    monkeypatch.setenv("ELASTIC_WRITE_INFLIGHT_MB", "8")
    e = NvmeEngine(blob16, read_ring=8, write_ring=8, read_qd=4)
    try:
        w = e.write_port
        n, vb = 2048, 4096
        raw = w.alloc_aligned(n * vb)
        raw[:] = np.arange(n * vb // 4, dtype=np.uint32).view(np.uint8)
        views = [raw[i * vb:(i + 1) * vb] for i in range(n)]
        iov = w.submit_writev(0, views, user_data=11)
        w.drain()
        assert w.wait_one() == (11, n * vb)
        del iov
        assert e.write_stats() == (1, 2)
        assert _readback(e, 0, n * vb) == bytes(raw)
    finally:
        e.close()


def test_default_unit_is_the_inflight_bound(eng):
    """With the default (32 MiB) unit the small-blob tests above are all
    one item per SQE or joined — never split."""
    w = eng.write_port
    keep = []
    b = w.alloc_aligned(8192); b[:] = 9
    keep.append(w.submit_writev(0, [b], user_data=1))
    w.drain()
    assert w.wait_one() == (1, 8192)
    assert eng.write_stats() == (1, 1)
