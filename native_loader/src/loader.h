#ifndef ELASTIC_LOADER_H
#define ELASTIC_LOADER_H

#include <stdint.h>
#include <stddef.h>
#include <sys/uio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate / free a host buffer aligned to at least the page size.
 * On a unified-memory system this buffer becomes directly addressable by the
 * GPU after a one-shot cudaHostRegister(..., cudaHostRegisterDefault), which
 * is what makes a read into it a zero-copy load. The registration is not
 * performed here — callers that need GPU access do it themselves, once per
 * buffer rather than once per read. */
void*  nvme_loader_alloc_buffer(size_t size, size_t alignment);
void   nvme_loader_free_buffer(void* buf);


/* ============================ nvme_engine ================================
 * The only thing in this process that talks to the device. A C thread owns
 * every io_uring call: it consumes each side's input FIFO in the order items
 * were enqueued and publishes completions on that side's output ring. The
 * read side is topped up one-out-one-in to a fixed queue depth
 * (ELASTIC_READ_QD). The write side has no queue-depth target: it composes
 * SQEs of at most one unit (ELASTIC_WRITE_INFLIGHT_MB) of file-contiguous
 * admitted bytes from the FIFO head, joining abutting items and splitting
 * long ones — but it is not unbounded either: the fill stops while
 * outstanding bytes are at or above that same unit (default 32 MiB). That
 * bound gates the next fill and never rejects an item, so a small value
 * cannot deadlock; it exists because how deep a write lands permanently sets
 * that blob's read-back speed on this drive.
 *
 * Device access is time-separated at batch grain: nvme_engine_write_drain
 * admits everything queued so far. From then until the batch has landed the
 * pop priority refills only writes; reads already on the device finish over
 * the batch's head, and no read is refilled until the batch has landed. This
 * is not a throughput lever — it makes the device match the solver's serial
 * pricing of it, lets each direction run at its own clean constant, and
 * makes the write interval deterministic. The STOP line reports how many write
 * phases were entered.
 *
 * Why the engine is C: the device's cadence is microseconds, while an
 * interpreter thread's turn is bounded below by its GIL slice, so a Python
 * submitter would feed the drive in bursts with gaps between them.
 * Refilling from C removes the language boundary from the refill loop.
 *
 * Division of labour:
 *   ReadRouter  (python, the loader, thread `elastic-head`) — waits on read
 *       completions and slot releases, maps file offsets to buffer
 *       positions, enqueues read items. Zero io_uring calls.
 *   WriteRouter (python, the writeback, thread `elastic-tail`) — waits on
 *       stream marks and write completions, hands licensed requests over,
 *       releases slots. Zero io_uring calls.
 *   nvme_engine (this) — the only io_uring user: the write-priority pop
 *       order; write SQEs composed of up to one unit (join/split); reads one
 *       item per SQE in enqueue order, never re-sorted by offset. No
 *       completion semantics, no block forms.
 *
 * Contract per item:
 *   - A read's `buf` is its destination buffer. A write's `buf` is a
 *     caller-owned `struct iovec[iovcnt]` (iovcnt > 0) that must stay valid
 *     until the item's completion is reaped from the output ring (io_uring
 *     reads the array asynchronously).
 *   - Validation (alignment / bounds / the read size cap) happens at
 *     enqueue, synchronously.
 *   - Short transfers are resubmitted internally from the landed prefix
 *     (O_DIRECT stops on a sector boundary, so the tail stays aligned). The
 *     output ring carries an item only when its full length landed, or with
 *     res < 0 on an I/O error, or with the landed byte count when the device
 *     reported EOF-no-progress (bounds make that unreachable; the caller's
 *     short-read raise stays the loud failure).
 *   - Enqueue never blocks the caller — it is a copy and a kick — so the
 *     submit-next-while-dispatching-current rotation stays in C, where the
 *     GIL cannot pause it.
 *
 * A ring-level failure (io_uring submit or CQE peek returning an errno) is
 * printed and aborts the process where it happens. */
typedef struct nvme_engine nvme_engine_t;

typedef struct {
    uint64_t offset;     /* file offset */
    uint64_t size;       /* total bytes; a write's must equal sum(iov_len) */
    void*    buf;        /* dst (read) / struct iovec* (write)              */
    uint32_t iovcnt;     /* 0 for a read, >0 for a write                    */
    uint32_t file_idx;   /* read side only; write side must pass 0          */
    uint64_t ud;         /* returned verbatim on the output ring            */
} nvme_engine_item_t;

#define NVME_ENGINE_READ  0
#define NVME_ENGINE_WRITE 1

/* Open both directions on `path` with O_DIRECT (read: O_RDONLY, write:
 * O_RDWR) and start the engine thread. `read_qd` is the read side's
 * one-out-one-in refill goal (clamped to the ring size) — the read contract
 * is "refill at the CQE". The write side takes requests in file-offset
 * order, in whatever grouping, and composes what reaches the device itself:
 * SQEs of at most one unit (ELASTIC_WRITE_INFLIGHT_MB) of file-contiguous
 * bytes, a new one composed only while less than a unit is in flight; every
 * request completes on its own. write_ring == 0 makes a read-only engine.
 * NULL on failure with *err_out set. */
nvme_engine_t* nvme_engine_start(const char* path,
                                 unsigned read_ring, unsigned write_ring,
                                 unsigned read_qd,
                                 int* err_out);
/* Attach another backing file for reads (the weight blob). Call at setup
 * with nothing in flight: the fixed-file table is re-registered. Returns the
 * file index reads address it by, or -1 with *err_out set to a negative
 * errno. */
int      nvme_engine_attach_file(nvme_engine_t*, const char* path,
                                 int* err_out);
/* Stop the engine: it finishes what is on the device and is freed. Reads
 * still queued are discarded (their count is on the STOP line). */
void     nvme_engine_stop(nvme_engine_t*);
/* Enqueue one item on `side`. 0 on success; -EAGAIN when the FIFO or the
 * outstanding budget is full (caller retries later); -EINVAL/-EBADF etc. on
 * validation failure. */
int      nvme_engine_enqueue(nvme_engine_t*, int side,
                             const nvme_engine_item_t* item);
/* Write valve: admit everything enqueued so far to the device. From then
 * until the batch has landed the pop priority refills only writes; reads
 * already on the device finish over the batch's head, and no read is
 * refilled until the batch has landed. The write FIFO holds items until
 * this is called (batch-grained admission); the read side has no valve. A
 * drain during an active phase extends it. */
void     nvme_engine_write_drain(nvme_engine_t*);
/* Write side: FIFO items fully submitted, and the SQEs the engine composed
 * them into. The engine turns the offset-ordered FIFO into SQEs of at most
 * one unit (ELASTIC_WRITE_INFLIGHT_MB) of file-contiguous bytes — abutting
 * items joined, an over-long item split — so sqes < items means joins,
 * sqes > items means splits. Both also printed at stop. */
void     nvme_engine_write_stats(nvme_engine_t*, uint64_t* items,
                                 uint64_t* sqes);
/* Reap up to `cap` completions from `side`'s output ring into uds/ress.
 * Returns the count (0 = none pending). Non-blocking. */
int      nvme_engine_reap(nvme_engine_t*, int side,
                          uint64_t* uds, int64_t* ress, unsigned cap);
/* A readable fd signalled once per output-ring entry — what the loader and
 * the writeback wait on (select). Counter-accumulating (eventfd); drain with
 * an 8-byte read and re-check the ring before sleeping. */
int      nvme_engine_out_fd(nvme_engine_t*, int side);
/* Completions pushed but not yet reaped by the caller. */
unsigned nvme_engine_out_ready(nvme_engine_t*, int side);
/* What the device still holds on `side` (engine-submitted minus
 * engine-reaped; the engine reaps promptly, so this is device-owed). */
unsigned nvme_engine_inflight(nvme_engine_t*, int side);
size_t   nvme_engine_alignment(nvme_engine_t*, int side);
uint64_t nvme_engine_file_size(nvme_engine_t*, unsigned file_idx);
/* Items one side holds outstanding (enqueued, completion not yet reaped):
 * an enqueue past this many returns -EAGAIN. */
unsigned nvme_engine_queue_cap(void);

/* Stream progress signal. A mark is a stream callback (cuStreamAddCallback;
 * userData = ticket): once everything queued before it on the producing
 * stream has completed, it raises the bound counter to the ticket and writes
 * 1 to the bound eventfd. Bind once, before the first mark. */
void     nvme_stream_signal_bind(int fd, int64_t* counter);
/* Mark `ticket` on a CUDA stream; make and record timing-free events; make a
 * stream wait on one. Driver calls only, resolved from the libcuda already
 * loaded in the process; bind them through a PyDLL so the caller keeps the
 * GIL. Return a CUresult (0 = ok) or -errno; the event constructor returns
 * NULL on failure. */
void*    nvme_stream_event_new(void);
int      nvme_stream_mark(void* stream, int64_t ticket);
int      nvme_event_record(void* ev, void* stream);
int      nvme_stream_wait(void* stream, void* ev);

#ifdef __cplusplus
}
#endif

#endif
