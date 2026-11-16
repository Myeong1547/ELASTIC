"""A compute point is heard when the GPU passes it, not asked about.

`StreamSignal.mark` adds a stream callback (`cuStreamAddCallback`) on the
given stream. The driver runs it once everything queued before it has
completed; it raises the bound counter to the ticket and writes the bound fd.
So the fd must stay silent while the GPU is still busy with the work in front
of the mark, and become readable with the counter at the ticket once that
work is done.
"""

import os
import select

import numpy as np
import pytest
import torch

from elastic.loader.nvme_loader import EventRing, StreamSignal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA stream")

# The C side keeps the counter's address and the fd until something rebinds
# them, and a callback may run after the test that queued it has returned.
# Both live as long as the process, so whatever fires late writes into memory
# and a descriptor that still exist.
_FD = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
_CNT = np.zeros(1, dtype=np.int64)


def test_mark_fires_after_the_work_before_it():
    """The fd is silent while the work in front of the mark runs, fires once
    it is done with the counter at the ticket; an older ticket fires the fd
    but never lowers the counter."""
    fd, cnt = _FD, _CNT
    cnt[:] = 0
    while select.select([fd], (), (), 0.0)[0]:
        os.read(fd, 8)
    sig = StreamSignal(fd, cnt)
    stream = torch.cuda.current_stream()
    try:
        torch.cuda.synchronize()
        torch.cuda._sleep(200_000_000)          # a few hundred ms of GPU work
        sig.mark(3, stream)
        r, _, _ = select.select([fd], (), (), 0.0)
        assert r == [] and cnt[0] == 0, "heard before the GPU got there"
        r, _, _ = select.select([fd], (), (), 30.0)
        assert r == [fd] and cnt[0] == 3
        assert os.read(fd, 8)
        sig.mark(2, stream)                     # an older ticket never lowers
        torch.cuda.synchronize()
        r, _, _ = select.select([fd], (), (), 5.0)
        assert r == [fd] and cnt[0] == 3
    finally:
        torch.cuda.synchronize()


def test_event_ring_orders_one_stream_after_another():
    """The ACT item of a kv_gen launch carries an event recorded on the
    kv_gen stream; the forward pass's stream waits on it. Nothing on the host
    waits: the ordering is the device's."""
    ring = EventRing(4)
    src, dst = torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(src):
        torch.cuda._sleep(200_000_000)
        x.fill_(7.0)
    ev = ring.record(src)
    EventRing.wait(dst, ev)
    with torch.cuda.stream(dst):
        y = x.clone()
    torch.cuda.synchronize()
    assert y.item() == 7.0
