"""Read round-trip correctness through the NVMe engine's read port — write a
known pattern, read it back, byte-compare.

This is the foundation everything else stands on: if the loader can return
the wrong bytes, every correctness result above it is meaningless and every
throughput number is measuring the wrong work. The patterns are built so a
wrong page is identifiable, not merely unequal — each page carries its own
index, so a failure says which page arrived.

Covered:
  - aligned page reads (the ordinary case)
  - reads at arbitrary aligned offsets inside the blob
  - several reads in flight at once, each landing in its own buffer
  - two page sizes, one half the other, packed in one file
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from elastic.loader.nvme_loader import NvmeEngine


PAGE_4K = 4096
PAGE_64K = 64 * 1024


def _make_blob(path: Path, n_pages: int, page_size: int) -> bytes:
    """Build a blob whose page i is its own 8-byte index repeated
    (page_size/8) times, so a read that lands on the wrong page reports
    WHICH page it got rather than just "not equal"."""
    blob = bytearray()
    for i in range(n_pages):
        chunk = struct.pack("<Q", i) * (page_size // 8)
        blob.extend(chunk)
    path.write_bytes(bytes(blob))
    return bytes(blob)


def _verify_page(buf: np.ndarray, page_index: int, page_size: int) -> None:
    expected = struct.pack("<Q", page_index) * (page_size // 8)
    actual = bytes(buf[:page_size])
    assert actual == expected, (
        f"page {page_index}: first 8 bytes are "
        f"{int.from_bytes(actual[:8], 'little')}, expected {page_index}")


@pytest.fixture
def blob_4k(tmp_path):
    p = tmp_path / "blob_4k.bin"
    _make_blob(p, n_pages=64, page_size=PAGE_4K)
    return p


@pytest.fixture
def blob_64k(tmp_path):
    p = tmp_path / "blob_64k.bin"
    _make_blob(p, n_pages=32, page_size=PAGE_64K)
    return p


def _engine(path, qd=1):
    return NvmeEngine(str(path), read_ring=8, write_ring=0, read_qd=qd)


def _read(port, off, n, ud=0):
    buf = port.alloc_aligned(n)
    port.submit_read(off, n, buf, user_data=ud)
    got, nbytes = port.wait_one()
    assert (got, nbytes) == (ud, n)
    return buf


def test_first_page(blob_4k):
    with _engine(blob_4k) as e:
        _verify_page(_read(e.read_port, 0, PAGE_4K), 0, PAGE_4K)


def test_pages_mid_blob(blob_4k):
    with _engine(blob_4k) as e:
        for i in (5, 17, 31, 63):
            _verify_page(_read(e.read_port, i * PAGE_4K, PAGE_4K), i,
                         PAGE_4K)


def test_larger_pages(blob_64k):
    """64 KiB pages, a few at various offsets."""
    with _engine(blob_64k) as e:
        for i in (0, 3, 15, 31):
            _verify_page(_read(e.read_port, i * PAGE_64K, PAGE_64K), i,
                         PAGE_64K)


def test_several_reads_in_flight_land_in_their_own_buffers(blob_4k):
    """Completions must report the right user_data, and data must land in
    the buffer it was submitted with — a mix-up silently mis-attributes
    pages rather than raising.

    The queue depth holds every read in flight at once: with several
    outstanding completions any mixing of user_data to buffer shows up,
    which a serialised submit/wait pair could never expose."""
    target_pages = [2, 9, 23, 47]
    with _engine(blob_4k, qd=len(target_pages)) as e:
        r = e.read_port
        bufs = {i: r.alloc_aligned(PAGE_4K) for i in target_pages}
        for i in target_pages:
            r.submit_read(i * PAGE_4K, PAGE_4K, bufs[i],
                          user_data=i)
        seen = set()
        for _ in target_pages:
            ud, nbytes = r.wait_one()
            assert nbytes == PAGE_4K, f"short read: {nbytes}"
            _verify_page(bufs[ud], ud, PAGE_4K)
            seen.add(ud)
        assert seen == set(target_pages)


def test_act_vs_kv_size_round_trip(tmp_path):
    """Two page sizes, one half the other, must round-trip out of the same
    file with no bytes crossing between them — a size confusion here would
    read half of the following page as if it were the requested one."""
    kv_pg, act_pg = PAGE_64K, PAGE_64K // 2
    blob = tmp_path / "act_kv.bin"
    # 4 large pages, then 8 half-size pages. The two parts get distinct fill
    # patterns so contamination between them is visible rather than
    # plausible.
    data = bytearray()
    for i in range(4):
        data.extend(struct.pack("<Q", 0xAA00 + i) * (kv_pg // 8))
    for i in range(8):
        data.extend(struct.pack("<Q", 0xBB00 + i) * (act_pg // 8))
    blob.write_bytes(bytes(data))

    with _engine(blob) as e:
        r = e.read_port
        # Each page's tag identifies its part and index.
        for i in range(4):
            buf = _read(r, i * kv_pg, kv_pg)
            tag = int.from_bytes(bytes(buf[:8]), "little")
            assert tag == 0xAA00 + i, f"KV {i}: tag {tag:#x} != {0xAA00+i:#x}"
        # The ACT part begins immediately after the four KV pages.
        act_start = 4 * kv_pg
        for i in range(8):
            buf = _read(r, act_start + i * act_pg, act_pg)
            tag = int.from_bytes(bytes(buf[:8]), "little")
            assert tag == 0xBB00 + i, f"ACT {i}: tag {tag:#x} != {0xBB00+i:#x}"
