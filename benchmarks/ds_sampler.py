#!/usr/bin/env python3
"""10 ms /proc/diskstats + /sys/block inflight sampler, flushed incrementally.

Rows are appended every 100 intervals, so a run killed at an unknown time
keeps everything sampled up to that point. One line per 10 ms interval:

    t_epoch  d_reads d_rsect d_writes d_wsect  infl_r infl_w

d_* are completed-I/O deltas over the interval (the service rate, which
occupancy cannot give); infl_* are instantaneous.

Usage: ds_sampler.py DEV OUT [dur_s=3600]
"""
import sys
import time

dev = sys.argv[1]
out = sys.argv[2]
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0


def stats():
    with open("/proc/diskstats") as f:
        for ln in f:
            p = ln.split()
            if p[2] == dev:
                return (int(p[3]), int(p[5]), int(p[7]), int(p[9]))
    raise RuntimeError(dev)


def infl():
    with open(f"/sys/block/{dev}/inflight") as f:
        a, b = f.read().split()
        return int(a), int(b)


fh = open(out, "w", buffering=1)
t0 = time.perf_counter()
prev = stats()
nxt = 0.010
n = 0
buf = []
while time.perf_counter() - t0 < dur:
    now = time.perf_counter() - t0
    if now < nxt:
        time.sleep(min(0.002, nxt - now))
        continue
    cur = stats()
    ir, iw = infl()
    buf.append(f"{time.time():.3f} {cur[0]-prev[0]} {cur[1]-prev[1]} "
               f"{cur[2]-prev[2]} {cur[3]-prev[3]} {ir} {iw}")
    prev = cur
    nxt += 0.010
    if now - 0.5 > nxt:          # fell behind (suspend etc.): resync
        nxt = now + 0.010
    n += 1
    if len(buf) >= 100:
        fh.write("\n".join(buf) + "\n")
        buf = []
if buf:
    fh.write("\n".join(buf) + "\n")
fh.close()
print(f"[ds_sampler] {n} intervals -> {out}", file=sys.stderr)
