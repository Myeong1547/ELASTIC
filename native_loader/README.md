# native_loader — the NVMe I/O engine

`libnvme_loader.so` is the only code in an ELASTIC process that talks to the
drive. `nvme_engine` is one C thread that owns one io_uring and serves two
FIFOs on it:

- **reads** (weights, KV and activation block reloads) — one item per SQE in
  enqueue order, topped up one-out-one-in to a fixed queue depth at every
  completion;
- **writes** (KV and activation block saves) — admitted per batch by
  `nvme_engine_write_drain`, composed into SQEs of at most one unit of
  file-contiguous bytes (abutting requests joined, long ones split), a new one
  only while less than one unit is in flight (one to two units on the device).

While an admitted write batch exists or a write is on the device, only writes
are refilled; reads resume once the batch has landed. Every file is opened
with `O_DIRECT`, so offsets, sizes and buffers must be multiples of the file's
direct-I/O alignment (from `statx`); enqueue checks this synchronously.

The Python side is `elastic/loader/nvme_loader.py` (ctypes): `NvmeEngine`,
its read/write `EnginePort`s, `StreamSignal`, `EventRing` and
`alloc_aligned`. `ReadRouter` (the loader) and `WriteRouter` (the writeback)
in `elastic/vllm_runtime/` decide what to read and write; they never call
io_uring themselves.

## Build

Requirements: Linux with io_uring, `gcc`, `make`, `liburing-dev`. CUDA is not
needed to build — the stream helpers resolve `libcuda.so.1` at run time.

```bash
sudo apt-get install -y liburing-dev
make            # -> build/libnvme_loader.so   (or `make loader` from the repo root)
make clean
```

The binding loads `native_loader/build/libnvme_loader.so`.

## API (`src/loader.h`)

| Group | Functions |
|---|---|
| Buffers | `nvme_loader_alloc_buffer`, `nvme_loader_free_buffer` — page-aligned host buffers (register them with CUDA yourself for zero-copy GPU access) |
| Engine lifecycle | `nvme_engine_start(path, read_ring, write_ring, read_qd, &err)` (`write_ring = 0` gives a read-only engine), `nvme_engine_attach_file` (another read file, e.g. the weight blob; setup only), `nvme_engine_stop` |
| I/O | `nvme_engine_enqueue` (`nvme_engine_item_t`: a read is one buffer, a write is an iovec array), `nvme_engine_write_drain`, `nvme_engine_reap`, `nvme_engine_out_fd` (eventfd signalled per completion) |
| Queries | `nvme_engine_queue_cap`, `nvme_engine_alignment`, `nvme_engine_file_size`, `nvme_engine_out_ready`, `nvme_engine_inflight`, `nvme_engine_write_stats` |
| CUDA stream helpers | `nvme_stream_signal_bind`, `nvme_stream_mark` (stream callback that raises a counter and writes an eventfd), `nvme_stream_event_new`, `nvme_event_record`, `nvme_stream_wait` |

A completion is posted only when an item's full length has landed (short
transfers are resubmitted internally), or with a negative errno on an I/O
error. A failure of the ring itself aborts the process.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `ELASTIC_WRITE_INFLIGHT_MB` | 32 | Write SQE unit, and the in-flight write bytes below which a new write SQE is filled (one to two units on the device). Read by the library at engine start. |
| `ELASTIC_READ_QD` | 8 | Read refill depth. Read by the Python binding and passed as `read_qd`. |

The engine prints one `[nvme_engine] START` line (ring and queue sizes, write
unit, path) and, for an engine with a write side, one `[nvme_engine] STOP`
line (write phases, write items and SQEs, I/O errors, discarded reads) to
stderr.
