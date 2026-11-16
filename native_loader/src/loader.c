#define _GNU_SOURCE
#include "loader.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include <liburing.h>
#include <limits.h>
#ifndef IOV_MAX
#define IOV_MAX 1024
#endif

/* Number of backing files attachable to one ring. io_uring carries the fd
 * per SQE, so a single ring can hold weight-blob and KV/activation-file
 * reads in flight together; splitting them across rings would force a drain
 * of one before touching the other, idling the device at every boundary. */
#define NVME_LOADER_MAX_FILES 8


/* The engine's ring and its read files. */
typedef struct nvme_loader {
    int                 fds[NVME_LOADER_MAX_FILES];
    size_t              f_align[NVME_LOADER_MAX_FILES];
    uint64_t            f_size[NVME_LOADER_MAX_FILES];
    unsigned            n_files;
    /* 1 => the fd table is registered with the ring, so a read SQE carries
     * the table index plus IOSQE_FIXED_FILE instead of a raw fd. Turned on by
     * attach_file, which is how one read ring addresses the weight blob and
     * the KV/activation file. */
    int                 files_reg;
    struct io_uring     ring;

    /* Pending submissions not yet flushed via io_uring_submit. */
    unsigned            pending_submit;

    /* Completion eventfd (see nvme_loader_completion_fd); -1 until it is
     * registered. Registering one lets a caller WAIT on the ring with
     * poll/select, which is the only way to wait on a completion AND on
     * something of its own in the same call. */
    int                 cq_efd;
} nvme_loader_t;


/* O_DIRECT alignment of an open regular file. */
static size_t probe_alignment(int fd) {
#ifdef STATX_DIOALIGN
    /* Take the kernel's exact O_DIRECT constraint (statx STATX_DIOALIGN) —
     * usually the device's logical block size, finer than the filesystem
     * block size statvfs reports — rather than assume the filesystem block
     * size. */
    {
        struct statx sx;
        if (statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0
            && (sx.stx_mask & STATX_DIOALIGN)
            && sx.stx_dio_mem_align > 0 && sx.stx_dio_offset_align > 0) {
            size_t a = sx.stx_dio_mem_align > sx.stx_dio_offset_align
                       ? sx.stx_dio_mem_align : sx.stx_dio_offset_align;
            if (a < 512) a = 512;
            return a;
        }
    }
#endif
    struct statvfs sv;
    if (fstatvfs(fd, &sv) == 0 && sv.f_bsize > 0) {
        /* Fallback when the kernel cannot report DIO alignment: the
         * filesystem block size is a safe over-estimate — coarser than
         * necessary, but never too fine. */
        size_t a = (size_t)sv.f_bsize;
        if (a < 512) a = 512;
        return a;
    }
    return 4096;
}

/* Open `path` with O_DIRECT | O_RDONLY and initialise one io_uring of depth
 * queue_depth. NULL on failure with *err_out (if non-NULL) set to a negative
 * errno. */
static nvme_loader_t* nvme_loader_open(const char* path,
                                       unsigned    queue_depth,
                                       int*        err_out)
{
    int fd = open(path, O_CLOEXEC | O_RDONLY | O_DIRECT);
    if (fd < 0) {
        if (err_out) *err_out = -errno;
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        if (err_out) *err_out = -errno;
        close(fd);
        return NULL;
    }

    nvme_loader_t* L = calloc(1, sizeof(*L));
    if (!L) { close(fd); if (err_out) *err_out = -ENOMEM; return NULL; }
    L->fds[0]      = fd;
    L->n_files     = 1;
    L->cq_efd      = -1;
    L->f_align[0]  = probe_alignment(fd);
    L->f_size[0]   = (uint64_t)st.st_size;

    /* A plain ring: no kernel submission poller. The engine's C thread owns
     * this queue and refills at the CQE, so the enter syscall and O_DIRECT's
     * page pinning are not on anyone's critical path and there is nothing for
     * a poller thread to take over. */
    int rc = io_uring_queue_init(queue_depth, &L->ring, 0);
    if (rc < 0) {
        if (err_out) *err_out = rc;
        free(L); close(fd);
        return NULL;
    }
    return L;
}

/* Attach a further backing file to the ring. The fd table is re-registered,
 * so from then on every read SQE carries a table index plus
 * IOSQE_FIXED_FILE. Returns the file index reads address it by, or -1 with
 * *err_out set to a negative errno. */
static int nvme_loader_attach_file(nvme_loader_t* L, const char* path,
                                   int* err_out)
{
    if (!L || !path) { if (err_out) *err_out = -EINVAL; return -1; }
    if (L->n_files >= NVME_LOADER_MAX_FILES) {
        if (err_out) *err_out = -ENOSPC;
        return -1;
    }
    int fd = open(path, O_CLOEXEC | O_RDONLY | O_DIRECT);
    if (fd < 0) { if (err_out) *err_out = -errno; return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        if (err_out) *err_out = -errno;
        close(fd);
        return -1;
    }

    unsigned idx = L->n_files;
    L->fds[idx]     = fd;
    L->f_align[idx] = probe_alignment(fd);
    L->f_size[idx]  = (uint64_t)st.st_size;
    L->n_files      = idx + 1;

    /* (Re)register the whole table so read SQEs can address any attached
     * file by index. This is why callers must attach at setup with nothing
     * in flight: an in-flight SQE's file index would be invalidated. */
    if (L->files_reg) io_uring_unregister_files(&L->ring);
    if (io_uring_register_files(&L->ring, L->fds, L->n_files) < 0) {
        /* The unregister above already succeeded, so the ring now has no
         * table: drop back to raw fds, which every read path can address. */
        L->n_files = idx;
        close(fd);
        L->files_reg = 0;
        if (err_out) *err_out = -EBADF;
        return -1;
    }
    L->files_reg = 1;
    return (int)idx;
}

static void nvme_loader_close(nvme_loader_t* L) {
    if (!L) return;
    io_uring_queue_exit(&L->ring);
    if (L->cq_efd >= 0) close(L->cq_efd);
    for (unsigned i = 0; i < L->n_files; i++)
        if (L->fds[i] >= 0) close(L->fds[i]);
    free(L);
}

void* nvme_loader_alloc_buffer(size_t size, size_t alignment) {
    void* p = NULL;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    /* Raise to the page size even when the filesystem's O_DIRECT minimum is
     * smaller: for a 512-byte alignment on a large size, glibc's
     * posix_memalign hands back page+512 (its mmap chunk header rounded up
     * to the requested alignment), leaving the buffer one sector past a
     * page.
     *
     * O_DIRECT is satisfied either way; the page alignment is for the write
     * side. On this drive, bytes written from a page-misaligned source read
     * back materially slower than the same bytes written from a page-aligned
     * one — independent of the read destination's alignment, and durable:
     * rewriting the file in place from a misaligned buffer does not restore
     * the rate, writing it from an aligned one does. The KV/activation file
     * is written once and read every decode step, so the write source's
     * alignment sets the read rate for the whole run. */
    size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    if (pg > 0 && alignment < pg) alignment = pg;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}
void nvme_loader_free_buffer(void* buf) { free(buf); }

/* fd argument for a read SQE on `file_idx`: the registered-table index when
 * the table is registered (paired with IOSQE_FIXED_FILE), else the raw fd. */
static inline int sqe_fd_for(const nvme_loader_t* L, unsigned file_idx) {
    return L->files_reg ? (int)file_idx : L->fds[file_idx];
}

/* ============================ correctness bound ==========================
 * A correctness bound, not a tuning constant.
 *
 * io_uring_prep_read takes its length as a 32-bit `unsigned`, and the kernel
 * caps a single read at MAX_RW_COUNT regardless. One layer's weights in a
 * large model can exceed 4 GiB, so a request built from a larger segment
 * does not fail — the length wraps mod 2^32 and the caller receives a
 * silently truncated buffer with a success return.
 *
 * A read item is one SQE, so the read side rejects anything larger than
 * this at enqueue. Writes are cut by the engine into SQEs of at most one
 * unit, so a write item of any size is fine.
 *
 * 1 GiB sits well under both limits and is a multiple of any O_DIRECT sector
 * size, so it can never itself produce an unaligned sub-read.
 * ======================================================================== */
#define ELASTIC_MAX_SINGLE_READ ((size_t)1 << 30)   /* 1 GiB */

/* Non-blocking reap of one CQE: 0 with (user_data, res), or -EAGAIN at once
 * when none is pending. */
static int nvme_loader_try_wait_one(nvme_loader_t* L,
                                    uint64_t* user_data_out,
                                    ssize_t*  res_out)
{
    struct io_uring_cqe* cqe = NULL;
    int rc = io_uring_peek_cqe(&L->ring, &cqe);
    if (rc < 0) return rc;              /* -EAGAIN: nothing pending */
    if (user_data_out) *user_data_out = cqe->user_data;
    if (res_out) *res_out = cqe->res;
    io_uring_cqe_seen(&L->ring, cqe);
    return 0;
}

/* A readable fd that the kernel signals whenever it posts a CQE.
 *
 * Without it, a caller with nothing to do has to pick one thing to sleep on.
 * Sleeping on a completion means ignoring everything else it is waiting for;
 * sleeping on its own condition means sleeping through completions, and a
 * loader that does that stops taking work back out of the queue for the
 * duration. An fd is the only form both kinds of event share, so this makes
 * the ring speak it and the caller waits on the set.
 *
 * Level-triggered by construction: the counter is only cleared by a read, so
 * a CQE posted between the caller's last peek and its wait leaves the fd
 * readable and the wait returns at once. No wakeup can be missed.
 *
 * Created and registered once, at engine start. Negative errno if the kernel
 * refuses. The engine sleeps on nothing else, so for it that refusal is a
 * failed start (`nvme_engine_start`). */
static int nvme_loader_completion_fd(nvme_loader_t* L) {
    int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (efd < 0) return -errno;
    int rc = io_uring_register_eventfd(&L->ring, efd);
    if (rc < 0) { close(efd); return rc; }
    L->cq_efd = efd;
    return efd;
}

/* Hand the queued SQEs to the kernel. 0, or -errno with the batch left
 * intact for a retry. */
static int nvme_loader_submit_flush(nvme_loader_t* L) {
    if (L->pending_submit == 0) return 0;
    int n = io_uring_submit(&L->ring);
    /* Submit failed outright (-EBUSY, -EINTR, ...). The SQEs are untouched in
     * the ring, so leave pending_submit alone — a later flush must retry them
     * — and hand the errno up rather than dropping it. */
    if (n < 0) return n;
    /* io_uring_submit returns how many SQEs the kernel took, which can be
     * fewer than were queued (CQ backpressure). Keep the remainder in
     * pending_submit so the next flush pushes it: zeroing it would strand
     * those SQEs in the ring, every later flush would be a no-op, and the
     * engine would wait for a CQE the kernel was never asked to produce. */
    unsigned took = (unsigned)n;
    L->pending_submit = took >= L->pending_submit ? 0u
                                                  : L->pending_submit - took;
    return 0;
}

/* ============================ nvme_engine ================================
 * See loader.h for the contract.
 *
 * One ring, one thread, two FIFOs. The engine keeps a logical side per
 * direction — its own input FIFO, output ring, entries and bounds — but one
 * server thread walks both FIFOs onto one io_uring, and the whole read/write
 * schedule is the pop priority:
 *
 *   write  first, exclusively, while an admitted batch exists or any write
 *          is on the device. Admission is valved at batch grain: nothing
 *          leaves the FIFO until the producer signals the batch is whole
 *          (nvme_engine_write_drain). The producers enqueue requests in
 *          file-offset order, whatever their grouping (one block, or a
 *          block region's blocks listed as views); the FIFO is therefore
 *          sequential by offset. What reaches the device is decided HERE
 *          and only here: an SQE is up to one unit (ELASTIC_WRITE_INFLIGHT_MB,
 *          default 32 MiB — the depth measured to read back at full speed)
 *          of file-contiguous admitted bytes in at most IOV_MAX pieces —
 *          items that abut are joined into one writev, an item longer than
 *          the unit is split at unit boundaries, one with more views than
 *          fit is cut at a view edge (`eng_write_compose`) — and every FIFO
 *          item completes on its own when its last byte lands
 *          (`eng_entry_done`). The same number is the fill bound on in-flight
 *          bytes: a new unit is composed only while less than it is in
 *          flight, so the device holds between one and two units (a
 *          sub-unit SQE — a range that stops abutting — leaves room for a
 *          full one); an unbounded walk saturates its tag space and lands
 *          the blob in a shape that reads back slower for good.
 *   read   only on a device that holds no writes, topped up one-out-one-in
 *          to `qd` (ELASTIC_READ_QD) at the CQE — that refill is the reason
 *          this engine exists. Reads the trigger caught in flight finish
 *          over the batch's head (at most one queue of reads); everything
 *          behind them queues in the FIFO, which keeps absorbing producers,
 *          so nothing upstream ever blocks.
 *
 * Why serial and not concurrent: the solver prices the device serially
 * (C1 = t_w + N_seq*(t_kv + t_save_kv)), whereas running both directions
 * at once degrades each into a regime-dependent blend. Separating them is
 * wall-neutral at equal work, so it is not a throughput lever: it makes the
 * device match the pricing model, lets each direction run at its own clean
 * constant, and makes the write interval deterministic. What a write batch
 * costs is still a drive regime (in-cache versus write-cache-exhausted);
 * the schedule only decides what else the device is doing meanwhile.
 *
 * Why one thread suffices: what a shared thread could break is the read
 * side's refill-at-the-CQE contract, by letting a write burst's
 * io_uring_submit (whose body is O_DIRECT page pinning) sit between a posted
 * read CQE and its refill. Under the serial schedule there is no read refill
 * while a write batch lands — that pause is the specification — and during
 * read service the thread submits no writes.
 *
 * Why an SQE always has room: the ring is opened with an SQ entry for every
 * engine entry of both sides, and an engine entry holds at most one SQE at a
 * time (a short-transfer tail reuses its entry only after the CQE).
 *
 * Locking: each logical side has one mutex (`mu`) guarding its input FIFO,
 * output ring and the outstanding budget — producers (python threads) and
 * the python reaper take it for microseconds. The entry tables, `inflight`
 * counters and the ring belong to the engine thread alone. `wmu`
 * serialises the work pass against attach, which must not race a ring
 * call; stop is the atomic `stop` flag, a kick and a join. `stop` and the
 * cumulative `errors` counter are atomic.
 * ======================================================================== */

#include <poll.h>

#define NVME_ENGINE_QCAP (1u << 16)      /* input FIFO entries per side */
#define ENG_UD_WR (1ull << 48)           /* CQE user_data: write-entry bit */

/* One FIFO item's share of a write SQE: `len` bytes of item `seq` (its FIFO
 * sequence number). */
typedef struct {
    uint64_t seq;
    uint64_t len;
} eng_mem_t;

typedef struct {
    nvme_engine_item_t it;               /* what the SQE carries — on the
                                          * write side the composed unit */
    uint64_t done;                       /* bytes landed so far */
    /* Write side: the composed SQE's own iovec (entry-owned — it must
     * outlive the CQE, a short-transfer tail is rebuilt from it into
     * `tail`) and the FIFO items it carries. All three hold IOV_MAX
     * entries, allocated at setup. nmem == 0 on the read side. */
    struct iovec* vec;
    struct iovec* tail;
    eng_mem_t* mem;
    unsigned nmem;
} eng_entry_t;

/* One logical side: FIFO in, completions out. No thread and no ring of its
 * own — the engine's single server owns those. */
typedef struct {
    /* shared (under mu) */
    pthread_mutex_t mu;
    nvme_engine_item_t* q;
    uint64_t qhead, qtail;
    uint64_t admit_upto;                 /* valve: fill pops only below this
                                          * queue position. Write side starts
                                          * closed (0) and opens per batch via
                                          * nvme_engine_write_drain; read side
                                          * is permanently open (UINT64_MAX). */
    uint64_t* o_ud;
    int64_t*  o_res;
    uint64_t ohead, otail;
    uint64_t outstanding;                /* enqueued - reaped-by-caller */
    int out_efd;
    /* engine-thread-private */
    eng_entry_t* entries;
    unsigned nring;
    unsigned* freeids;
    unsigned nfree;
    unsigned inflight;                   /* on-device */
    uint64_t infl_bytes;                 /* bytes of the above (short-transfer
                                          * tails keep their full size until
                                          * the last byte lands) */
    unsigned* resume_ids;                /* entries owed a tail resubmit */
    unsigned nresume;
    unsigned qd;                     /* read: refill goal; write: nring */
    uint64_t max_bytes;                  /* fill bound on in-flight bytes
                                          * (0 = none; write side only) — and
                                          * the write side's SQE unit */
    /* Write side: the head item may be partly submitted (split at a unit
     * boundary); `hcur` is how many of its bytes already left. Per-item
     * landing state, parallel to `q` (same index): bytes landed so far and
     * whether its completion was posted. Valid while the item is
     * outstanding, which the enqueue bound guarantees no wrap over. */
    uint64_t hcur;
    uint64_t* q_landed;
    unsigned char* q_posted;
    uint64_t n_items;                    /* FIFO items fully submitted */
    uint64_t n_sqes;                     /* SQEs they were submitted as */
} eng_side_t;

struct nvme_engine {
    /* One ring. `L` carries it plus the read fd(s) — the registered file
     * table, where weight blobs attach. Writes ride the same ring on their
     * own raw fd (`wfd`), opened O_RDWR; write SQEs never use
     * IOSQE_FIXED_FILE. */
    nvme_loader_t* L;
    int      wfd;                        /* -1 = read-only engine */
    size_t   w_align;
    uint64_t w_size;
    pthread_t th;
    int started;
    pthread_mutex_t wmu;                 /* work pass vs attach/stop */
    int kick_efd;                        /* new work / drain / stop */
    int cq_efd;                          /* ring's CQE eventfd */
    eng_side_t s[2];                     /* [NVME_ENGINE_READ, NVME_ENGINE_WRITE] */
    int stop;                            /* atomic */
    uint32_t errors;                     /* atomic; device errors, cumulative */
    /* Diagnostics for the pop priority: w_phase mirrors "writes own the
     * device right now" (engine-thread-owned), w_phases counts entries into
     * that state and is printed when the engine stops. */
    int w_phase;
    uint64_t w_phases;
};

static inline void eng_err(nvme_engine_t* E) {
    __atomic_fetch_add(&E->errors, 1u, __ATOMIC_RELAXED);
}

static inline int eng_stopping(nvme_engine_t* E) {
    return __atomic_load_n(&E->stop, __ATOMIC_ACQUIRE);
}

static inline void eng_kick(nvme_engine_t* E) {
    uint64_t one = 1;
    ssize_t wr = write(E->kick_efd, &one, sizeof one);
    (void)wr;
}

/* The ring itself refused (submit or CQE peek returned an errno). Nothing on
 * it can be delivered any more, so say where and stop the process here. */
static void eng_ring_failed(const char* what, int rc) {
    fprintf(stderr, "[nvme_engine] FATAL io_uring %s failed: %s\n",
            what, strerror(-rc));
    abort();
}

static void eng_out_push(eng_side_t* S, uint64_t ud, int64_t res) {
    pthread_mutex_lock(&S->mu);
    /* Capacity: outstanding <= QCAP is enforced at enqueue and every output
     * entry retires one outstanding item, so the ring cannot overflow. */
    S->o_ud [S->otail % NVME_ENGINE_QCAP] = ud;
    S->o_res[S->otail % NVME_ENGINE_QCAP] = res;
    S->otail++;
    pthread_mutex_unlock(&S->mu);
    uint64_t one = 1;
    ssize_t wr = write(S->out_efd, &one, sizeof one);
    (void)wr;                            /* counter overflow: impossible here */
}

static void eng_free_entry(eng_side_t* S, unsigned id) {
    S->freeids[S->nfree++] = id;
    S->inflight--;
    S->infl_bytes -= S->entries[id].it.size;   /* paired with the fill's add */
}

/* Post one FIFO item's completion once. An item posted before its last
 * byte was even submitted (error or short landing on an earlier unit of a
 * split item) is finished as far as the caller is concerned — the caller
 * may release its buffers on this completion — so its unsubmitted remainder
 * is dropped from the FIFO here rather than composed later from memory the
 * caller no longer holds. */
static void eng_item_post(eng_side_t* S, uint64_t seq, int64_t res) {
    unsigned i = (unsigned)(seq % NVME_ENGINE_QCAP);
    if (S->q_posted[i]) return;
    S->q_posted[i] = 1;
    pthread_mutex_lock(&S->mu);
    if (seq == S->qhead && S->hcur > 0) { S->qhead++; S->hcur = 0; }
    pthread_mutex_unlock(&S->mu);
    eng_out_push(S, S->q[i].ud, res);
}

/* Deliver an entry's completion. Read side (nmem == 0): one push. Write side:
 * the SQE carried `nmem` item shares; `res` < 0 is that error for each of
 * them, a full landing credits each share to its item and posts the item
 * when its last byte is in (it may have been split across SQEs), a short
 * landing (EOF, unreachable for a valid item) posts what each item has so
 * the caller's short-transfer raise fires. */
static void eng_entry_done(eng_side_t* S, unsigned id, int64_t res) {
    eng_entry_t* ent = &S->entries[id];
    if (ent->nmem == 0) {
        eng_out_push(S, ent->it.ud, res);
        return;
    }
    int full = (res >= 0 && (uint64_t)res == ent->it.size);
    uint64_t landed = res < 0 ? 0 : (uint64_t)res;
    uint64_t pre = 0;
    for (unsigned m = 0; m < ent->nmem; m++) {
        const eng_mem_t* me = &ent->mem[m];
        unsigned i = (unsigned)(me->seq % NVME_ENGINE_QCAP);
        if (res < 0) {
            eng_item_post(S, me->seq, res);
        } else {
            uint64_t have = landed > pre ? landed - pre : 0;
            if (have > me->len) have = me->len;
            S->q_landed[i] += have;
            if (S->q_landed[i] >= S->q[i].size)
                eng_item_post(S, me->seq, (int64_t)S->q[i].size);
            else if (!full)
                eng_item_post(S, me->seq, (int64_t)S->q_landed[i]);
        }
        pre += me->len;
    }
}

/* Iovec pieces of [pos, pos + *len) of write item `it`, at most `cap` of
 * them, written to `out`. Returns the count and sets *len to the bytes those
 * pieces cover — short of the request only when `cap` cut it, and then at a
 * view edge. */
static unsigned eng_item_pieces(const nvme_engine_item_t* it, uint64_t pos,
                                uint64_t* len, unsigned cap,
                                struct iovec* out) {
    const struct iovec* v = (const struct iovec*)it->buf;
    unsigned n = 0;
    uint64_t skip = pos, need = *len, got = 0;
    for (uint32_t j = 0; j < it->iovcnt && need && n < cap; j++) {
        uint64_t l = v[j].iov_len;
        if (skip >= l) { skip -= l; continue; }
        uint64_t avail = l - skip;
        uint64_t take = avail < need ? avail : need;
        out[n].iov_base = (char*)v[j].iov_base + skip;
        out[n].iov_len = (size_t)take;
        n++;
        need -= take;
        got += take;
        skip = 0;
    }
    *len = got;
    return n;
}

/* Compose the next write SQE into entry `ent` from the FIFO head: up to one
 * unit of file-contiguous admitted bytes in at most IOV_MAX pieces, starting
 * `S->hcur` bytes into the head item, across as many items as abut (joined)
 * and cutting an item at the unit boundary or, when its views run past
 * IOV_MAX, at a view edge (split). *end_seq / *end_pos say where the FIFO
 * cursor lands once the SQE is accepted. Caller holds S->mu and has seen an
 * admitted item at the head. Every cut is a multiple of the unit, an item
 * size or a view length, all alignment multiples, so the pieces keep
 * O_DIRECT's constraints. */
static void eng_write_compose(eng_side_t* S, eng_entry_t* ent,
                              uint64_t* end_seq, uint64_t* end_pos) {
    uint64_t unit = S->max_bytes;
    uint64_t lim = S->admit_upto < S->qtail ? S->admit_upto : S->qtail;
    uint64_t seq = S->qhead, pos = S->hcur, total = 0, next_off = 0;
    unsigned niov = 0, nmem = 0;
    while (seq < lim && total < unit && niov < (unsigned)IOV_MAX) {
        const nvme_engine_item_t* it = &S->q[seq % NVME_ENGINE_QCAP];
        if (nmem && it->offset + pos != next_off) break;   /* not abutting */
        uint64_t take = it->size - pos;
        if (take > unit - total) take = unit - total;
        niov += eng_item_pieces(it, pos, &take, (unsigned)IOV_MAX - niov,
                                ent->vec + niov);
        ent->mem[nmem].seq = seq;
        ent->mem[nmem].len = take;
        nmem++;
        total += take;
        next_off = it->offset + pos + take;
        if (pos + take == it->size) { seq++; pos = 0; }
        else pos += take;
    }
    const nvme_engine_item_t* h = &S->q[S->qhead % NVME_ENGINE_QCAP];
    ent->it.offset = h->offset + S->hcur;
    ent->it.size = total;
    ent->it.buf = ent->vec;
    ent->it.iovcnt = niov;
    ent->nmem = nmem;
    *end_seq = seq;
    *end_pos = pos;
}

/* Queue the SQE for side `side`, entry `id`, from its cursor (`done` bytes
 * into the item). Reads: one plain read into the item's buffer, addressed
 * through the fixed-file table once a file is attached. Writes: the entry's
 * composed writev on the raw write fd; a short-transfer tail skips the
 * landed prefix into the entry's `tail`. Validation already ran at
 * enqueue. */
static void eng_prep(nvme_engine_t* E, int side, unsigned id) {
    eng_side_t* S = &E->s[side];
    eng_entry_t* ent = &S->entries[id];
    const nvme_engine_item_t* it = &ent->it;
    uint64_t off = it->offset + ent->done;
    struct io_uring_sqe* sqe = io_uring_get_sqe(&E->L->ring);
    if (side == NVME_ENGINE_READ) {
        io_uring_prep_read(sqe, sqe_fd_for(E->L, it->file_idx),
                           (char*)it->buf + ent->done,
                           (unsigned)(it->size - ent->done), off);
        if (E->L->files_reg) io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
    } else {
        const struct iovec* use = ent->vec;
        int cnt = (int)it->iovcnt;
        if (ent->done) {
            uint64_t skip = ent->done;
            cnt = 0;
            for (uint32_t i = 0; i < it->iovcnt; i++) {
                uint64_t len = ent->vec[i].iov_len;
                if (skip >= len) { skip -= len; continue; }
                ent->tail[cnt].iov_base = (char*)ent->vec[i].iov_base + skip;
                ent->tail[cnt].iov_len  = (size_t)(len - skip);
                skip = 0;
                cnt++;
            }
            use = ent->tail;
        }
        io_uring_prep_writev(sqe, E->wfd, use, cnt, off);
    }
    sqe->user_data = (uint64_t)id | (side == NVME_ENGINE_WRITE ? ENG_UD_WR : 0);
    E->L->pending_submit += 1;
}

static void eng_flush(nvme_engine_t* E) {
    if (E->L->pending_submit) {
        int fr = nvme_loader_submit_flush(E->L);
        if (fr < 0 && fr != -EINTR) eng_ring_failed("submit", fr);
    }
}

/* Serve owed short-transfer tails for one side. They are continuations of
 * already-started items and hold entries, so nothing new may starve them. */
static int eng_fill_tails(nvme_engine_t* E, int side) {
    eng_side_t* S = &E->s[side];
    int prog = 0;
    while (S->nresume) {
        eng_prep(E, side, S->resume_ids[--S->nresume]);
        prog = 1;
    }
    return prog;
}

/* Pop work from one side's FIFO onto the ring while the bounds allow.
 * Read side: one item, one SQE, as it came. Write side: one composed unit
 * (`eng_write_compose`) per SQE, however the items were grouped. */
static int eng_fill_side(nvme_engine_t* E, int side) {
    eng_side_t* S = &E->s[side];
    int prog = 0;
    while (!eng_stopping(E) && S->inflight < S->qd && S->nfree
           && (S->max_bytes == 0 || S->infl_bytes < S->max_bytes)) {
        unsigned id = S->freeids[S->nfree - 1];
        eng_entry_t* ent = &S->entries[id];
        uint64_t popped = 0;
        pthread_mutex_lock(&S->mu);
        int have = (S->qhead != S->qtail) && (S->qhead < S->admit_upto);
        if (have) {
            if (side == NVME_ENGINE_WRITE) {
                uint64_t end_seq = 0, end_pos = 0;
                eng_write_compose(S, ent, &end_seq, &end_pos);
                popped = end_seq - S->qhead;     /* items fully consumed */
                S->qhead = end_seq;
                S->hcur = end_pos;
            } else {
                ent->it = S->q[S->qhead % NVME_ENGINE_QCAP];
                popped = 1;
                S->qhead++;
            }
        }
        pthread_mutex_unlock(&S->mu);
        if (!have) break;
        ent->done = 0;
        eng_prep(E, side, id);
        S->nfree--;
        S->inflight++;
        S->infl_bytes += ent->it.size;
        __atomic_fetch_add(&S->n_items, popped, __ATOMIC_RELAXED);
        __atomic_fetch_add(&S->n_sqes, 1u, __ATOMIC_RELAXED);
        prog = 1;
    }
    return prog;
}

/* The schedule. Tails first (either side — they hold entries), then the pop
 * priority: writes exclusively while an admitted batch exists or any write
 * is on the device; reads only on a device that holds no writes. */
static int eng_fill(nvme_engine_t* E) {
    int prog = 0;
    for (int side = 0; side < 2; side++)
        prog |= eng_fill_tails(E, side);
    eng_side_t* W = &E->s[NVME_ENGINE_WRITE];
    if (E->wfd >= 0) {
        prog |= eng_fill_side(E, NVME_ENGINE_WRITE);
        int w_active = W->inflight || W->nresume;
        if (!w_active) {
            pthread_mutex_lock(&W->mu);
            w_active = (W->qhead != W->qtail) && (W->qhead < W->admit_upto);
            pthread_mutex_unlock(&W->mu);
        }
        if (w_active) {
            if (!E->w_phase) {
                E->w_phase = 1;
                __atomic_fetch_add(&E->w_phases, 1, __ATOMIC_RELAXED);
            }
            eng_flush(E);
            return prog;                 /* reads wait: that is the schedule */
        }
        E->w_phase = 0;
    }
    prog |= eng_fill_side(E, NVME_ENGINE_READ);
    eng_flush(E);
    return prog;
}

static int eng_reap(nvme_engine_t* E) {
    int prog = 0;
    for (;;) {
        uint64_t ud = 0;
        ssize_t res = 0;
        int rc = nvme_loader_try_wait_one(E->L, &ud, &res);
        if (rc == -EAGAIN) break;
        if (rc < 0) eng_ring_failed("CQE peek", rc);
        prog = 1;
        int side = (ud & ENG_UD_WR) ? NVME_ENGINE_WRITE : NVME_ENGINE_READ;
        unsigned id = (unsigned)(ud & ~ENG_UD_WR);
        eng_side_t* S = &E->s[side];
        eng_entry_t* ent = &S->entries[id];
        if (res < 0) {
            eng_err(E);
            eng_entry_done(S, id, (int64_t)res);
            eng_free_entry(S, id);
        } else if (res == 0) {
            /* No progress (EOF). Bounds checks make this unreachable for a
             * valid item; deliver the landed count so the caller's own
             * short-transfer raise fires instead of looping here. */
            eng_entry_done(S, id, (int64_t)ent->done);
            eng_free_entry(S, id);
        } else {
            ent->done += (uint64_t)res;
            if (ent->done < ent->it.size) {
                /* Short transfer: the tail is aligned (O_DIRECT stops on a
                 * sector boundary) — resubmit it from this entry's cursor.
                 * Queued through the resume stack; the completion event
                 * stays deferred until the last byte lands. */
                S->resume_ids[S->nresume++] = id;
            } else {
                eng_entry_done(S, id, (int64_t)ent->it.size);
                eng_free_entry(S, id);
            }
        }
        /* One out, one in, here: the completion is the moment device
         * capacity frees, so the replacement (or the owed tail) goes in
         * before the next completion is even looked at — and the pop
         * priority inside decides which side that replacement comes from. */
        eng_fill(E);
    }
    eng_flush(E);
    return prog;
}

/* The engine's whole life. One pass = reap, fill (the schedule); sleep on
 * the ring's CQE eventfd + the kick. */
static void* eng_main(void* arg) {
    nvme_engine_t* E = (nvme_engine_t*)arg;
    pthread_setname_np(pthread_self(), "nvme_engine");
    struct pollfd pfds[2];
    for (;;) {
        pthread_mutex_lock(&E->wmu);
        int prog = eng_reap(E);
        /* Runs after stop too: the fresh-item pops are gated on !stop
         * inside, but owed short-transfer tails must still resubmit or
         * the join below never sees the device empty. */
        prog |= eng_fill(E);
        int done = eng_stopping(E)
                   && E->s[0].inflight == 0 && E->s[0].nresume == 0
                   && E->s[1].inflight == 0 && E->s[1].nresume == 0;
        pthread_mutex_unlock(&E->wmu);
        if (done)
            break;
        if (prog)
            continue;
        int n = 0;
        pfds[n].fd = E->cq_efd; pfds[n].events = POLLIN; n++;
        pfds[n].fd = E->kick_efd; pfds[n].events = POLLIN; n++;
        /* Every wakeup is an fd — a posted CQE via the ring eventfd, new
         * work / a drain / stop via the kick — so there is no timeout. */
        (void)poll(pfds, (nfds_t)n, -1);
        for (int i = 0; i < n; i++)
            if (pfds[i].revents & POLLIN) {
                uint64_t v;
                ssize_t rd = read(pfds[i].fd, &v, sizeof v);
                (void)rd;
            }
    }
    return NULL;
}

/* `compose` (the write side): each entry gets its IOV_MAX-entry compose
 * scratch here, once, so composing an SQE never allocates. */
static int eng_side_setup(eng_side_t* S, unsigned nring, unsigned qd,
                          uint64_t max_bytes, uint64_t admit_upto,
                          int compose) {
    pthread_mutex_init(&S->mu, NULL);
    S->nring = nring;
    S->qd = (qd == 0 || qd > nring) ? nring : qd;
    S->max_bytes = max_bytes;
    S->admit_upto = admit_upto;
    S->entries    = calloc(nring, sizeof *S->entries);
    S->freeids    = calloc(nring, sizeof *S->freeids);
    S->resume_ids = calloc(nring, sizeof *S->resume_ids);
    S->q     = calloc(NVME_ENGINE_QCAP, sizeof *S->q);
    S->o_ud  = calloc(NVME_ENGINE_QCAP, sizeof *S->o_ud);
    S->o_res = calloc(NVME_ENGINE_QCAP, sizeof *S->o_res);
    S->q_landed = calloc(NVME_ENGINE_QCAP, sizeof *S->q_landed);
    S->q_posted = calloc(NVME_ENGINE_QCAP, sizeof *S->q_posted);
    if (!S->entries || !S->freeids || !S->resume_ids
        || !S->q || !S->o_ud || !S->o_res || !S->q_landed || !S->q_posted)
        return -ENOMEM;
    for (unsigned i = 0; compose && i < nring; i++) {
        eng_entry_t* ent = &S->entries[i];
        ent->vec  = malloc((size_t)IOV_MAX * sizeof *ent->vec);
        ent->tail = malloc((size_t)IOV_MAX * sizeof *ent->tail);
        ent->mem  = malloc((size_t)IOV_MAX * sizeof *ent->mem);
        if (!ent->vec || !ent->tail || !ent->mem)
            return -ENOMEM;
    }
    for (unsigned i = 0; i < nring; i++)
        S->freeids[S->nfree++] = i;
    /* The side's completions are heard on this fd and nowhere else: a side
     * without it could never tell its caller that anything landed. */
    S->out_efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (S->out_efd < 0) return -errno;
    return 0;
}

static void eng_side_teardown(eng_side_t* S) {
    if (S->out_efd > 0) close(S->out_efd);
    if (S->entries)
        for (unsigned i = 0; i < S->nring; i++) {
            free(S->entries[i].vec);
            free(S->entries[i].tail);
            free(S->entries[i].mem);
        }
    free(S->entries); free(S->freeids); free(S->resume_ids);
    free(S->q); free(S->o_ud); free(S->o_res);
    free(S->q_landed); free(S->q_posted);
    pthread_mutex_destroy(&S->mu);
    memset(S, 0, sizeof *S);
    S->out_efd = -1;
}

static void eng_destroy(nvme_engine_t* E) {
    if (E->wfd >= 0) close(E->wfd);
    if (E->L) nvme_loader_close(E->L);          /* closes cq_efd too */
    if (E->kick_efd >= 0) close(E->kick_efd);
    eng_side_teardown(&E->s[0]);
    eng_side_teardown(&E->s[1]);
    pthread_mutex_destroy(&E->wmu);
    free(E);
}

nvme_engine_t* nvme_engine_start(const char* path,
                                 unsigned read_ring, unsigned write_ring,
                                 unsigned read_qd,
                                 int* err_out)
{
    nvme_engine_t* E = calloc(1, sizeof *E);
    if (!E) { if (err_out) *err_out = -ENOMEM; return NULL; }
    E->wfd = -1;
    E->kick_efd = -1;
    E->cq_efd = -1;
    E->s[0].out_efd = E->s[1].out_efd = -1;
    pthread_mutex_init(&E->wmu, NULL);
    int rc = 0;
    /* One ring sized for both directions' concurrent SQEs. The ring's
     * nvme_loader_t (`L`) is opened for reading (its fd + registered table
     * serve the read SQEs); the write direction gets its own raw fd below. */
    int lerr = 0;
    E->L = nvme_loader_open(path, read_ring + write_ring, &lerr);
    if (!E->L) rc = lerr ? lerr : -EIO;
    if (rc == 0) {
        E->kick_efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (E->kick_efd < 0) rc = -errno;
    }
    if (rc == 0) {
        E->cq_efd = nvme_loader_completion_fd(E->L);
        /* The engine sleeps on this fd and the kick, nothing else: without
         * it a completion could not wake the engine. */
        if (E->cq_efd < 0) rc = E->cq_efd;
    }
    if (rc == 0)
        rc = eng_side_setup(&E->s[NVME_ENGINE_READ], read_ring, read_qd,
                            0, UINT64_MAX, 0);
    if (rc == 0 && write_ring) {
        E->wfd = open(path, O_CLOEXEC | O_RDWR | O_DIRECT);
        if (E->wfd < 0) rc = -errno;
        if (rc == 0) {
            struct stat st;
            if (fstat(E->wfd, &st) < 0) rc = -errno;
            else {
                E->w_align = probe_alignment(E->wfd);
                E->w_size = (uint64_t)st.st_size;
            }
        }
        if (rc == 0) {
            /* Write-side unit (32 MiB default): the SQE size the engine
             * composes from the FIFO whatever the requests' grouping, and
             * the fill bound on in-flight bytes (one to two units on the
             * device).
             * A landing-quality number, not a queue-depth tuning knob: on
             * this drive the depth a write lands at permanently sets the
             * read-back rate of the blob it wrote. Landed with 32 MiB in
             * flight it reads back at full sequential rate; with 192 MiB or
             * more, about 25% slower. */
            long mb = 32;
            const char* e = getenv("ELASTIC_WRITE_INFLIGHT_MB");
            if (e && *e) {
                char* end = NULL;
                long v = strtol(e, &end, 10);
                if (end != e && v > 0) mb = v;
            }
            rc = eng_side_setup(&E->s[NVME_ENGINE_WRITE], write_ring, 0,
                                (uint64_t)mb << 20, 0, 1);
        }
    }
    if (rc == 0) {
        if (pthread_create(&E->th, NULL, eng_main, E))
            rc = -EAGAIN;
        else
            E->started = 1;
    }
    if (rc) {
        if (err_out) *err_out = rc;
        __atomic_store_n(&E->stop, 1, __ATOMIC_RELEASE);
        if (E->started) { eng_kick(E); pthread_join(E->th, NULL); }
        eng_destroy(E);
        return NULL;
    }
    fprintf(stderr, "[nvme_engine] START ring(qd=%u) read(ring=%u qd=%u) "
            "write(entries=%u cap=%lluMiB) path=%s\n",
            read_ring + write_ring, E->s[0].nring, E->s[0].qd,
            write_ring ? E->s[1].nring : 0,
            write_ring ? (unsigned long long)(E->s[1].max_bytes >> 20) : 0ULL,
            path);
    return E;
}

void nvme_engine_stop(nvme_engine_t* E) {
    if (!E) return;
    __atomic_store_n(&E->stop, 1, __ATOMIC_RELEASE);
    if (E->started) {
        eng_kick(E);
        pthread_join(E->th, NULL);
    }
    /* Reads still queued never reach the device: the loader (ReadRouter)
     * was stopped and joined before the engine, so nobody waits for them. */
    if (E->wfd >= 0)
        fprintf(stderr, "[nvme_engine] STOP w_phases=%llu w_items=%llu "
                "w_sqes=%llu errors=%u discarded_reads=%llu\n",
                (unsigned long long)__atomic_load_n(&E->w_phases,
                                                    __ATOMIC_RELAXED),
                (unsigned long long)__atomic_load_n(
                    &E->s[NVME_ENGINE_WRITE].n_items, __ATOMIC_RELAXED),
                (unsigned long long)__atomic_load_n(
                    &E->s[NVME_ENGINE_WRITE].n_sqes, __ATOMIC_RELAXED),
                __atomic_load_n(&E->errors, __ATOMIC_RELAXED),
                (unsigned long long)(E->s[NVME_ENGINE_READ].qtail
                                     - E->s[NVME_ENGINE_READ].qhead));
    eng_destroy(E);
}

/* Write side: FIFO items fully submitted, and the SQEs the engine composed
 * them into (joins make sqes < items, splits make sqes > items). */
void nvme_engine_write_stats(nvme_engine_t* E, uint64_t* items,
                             uint64_t* sqes) {
    uint64_t i = 0, q = 0;
    if (E) {
        i = __atomic_load_n(&E->s[NVME_ENGINE_WRITE].n_items, __ATOMIC_RELAXED);
        q = __atomic_load_n(&E->s[NVME_ENGINE_WRITE].n_sqes, __ATOMIC_RELAXED);
    }
    if (items) *items = i;
    if (sqes) *sqes = q;
}

int nvme_engine_attach_file(nvme_engine_t* E, const char* path, int* err_out)
{
    if (!E || !E->L) { if (err_out) *err_out = -EINVAL; return -1; }
    /* Setup-time only: attach re-registers the fixed-file table, so the
     * work pass must not run a ring call meanwhile. */
    pthread_mutex_lock(&E->wmu);
    int idx = nvme_loader_attach_file(E->L, path, err_out);
    pthread_mutex_unlock(&E->wmu);
    return idx;
}

/* Enqueue-time validation: a bad item fails here, synchronously, in the
 * caller's own call. */
static int eng_validate(nvme_engine_t* E, int side,
                        const nvme_engine_item_t* it) {
    size_t a;
    uint64_t fsz;
    if (side == NVME_ENGINE_WRITE) {
        if (it->file_idx) return -EBADF;        /* writes: main file only */
        if (!it->iovcnt) return -EINVAL;         /* a write is an iovec array */
        a = E->w_align;
        fsz = E->w_size;
    } else {
        nvme_loader_t* L = E->L;
        if (it->file_idx >= L->n_files) return -EBADF;
        if (it->iovcnt) return -EINVAL;          /* a read is one buffer */
        if (it->size > ELASTIC_MAX_SINGLE_READ) return -EINVAL;
        a = L->f_align[it->file_idx];
        fsz = L->f_size[it->file_idx];
    }
    if (it->offset % a) return -EINVAL;
    if (it->size == 0) return -EINVAL;
    if (it->offset + it->size > fsz) return -EINVAL;
    if (it->iovcnt == 0) {
        if (!it->buf) return -EINVAL;
        if ((uintptr_t)it->buf % a) return -EINVAL;
        if (it->size % a) return -EINVAL;
        return 0;
    }
    const struct iovec* iov = (const struct iovec*)it->buf;
    uint64_t total = 0;
    for (uint32_t i = 0; i < it->iovcnt; i++) {
        if ((uintptr_t)iov[i].iov_base % a) return -EINVAL;
        if (iov[i].iov_len % a)             return -EINVAL;
        total += (uint64_t)iov[i].iov_len;
    }
    if (total != it->size) return -EINVAL;
    return 0;
}

int nvme_engine_enqueue(nvme_engine_t* E, int side,
                        const nvme_engine_item_t* item)
{
    if (!E || side < 0 || side > 1 || !item) return -EINVAL;
    if (!E->L || (side == NVME_ENGINE_WRITE && E->wfd < 0)) return -EBADF;
    eng_side_t* S = &E->s[side];
    int rc = eng_validate(E, side, item);
    if (rc) return rc;
    pthread_mutex_lock(&S->mu);
    if (S->outstanding >= NVME_ENGINE_QCAP
        || (S->qtail - S->qhead) >= NVME_ENGINE_QCAP) {
        pthread_mutex_unlock(&S->mu);
        return -EAGAIN;
    }
    S->q[S->qtail % NVME_ENGINE_QCAP] = *item;
    S->q_landed[S->qtail % NVME_ENGINE_QCAP] = 0;
    S->q_posted[S->qtail % NVME_ENGINE_QCAP] = 0;
    S->qtail++;
    S->outstanding++;
    pthread_mutex_unlock(&S->mu);
    eng_kick(E);
    return 0;
}

/* Open the write valve up to everything queued so far. Write admission is
 * batch-grained: items accumulate in the FIFO while the producer composes a
 * batch and reach the device only on this call, together — a request that
 * reaches the device alone occupies a die for milliseconds and costs reads
 * far more than its bytes. Items enqueued after this call wait for their own
 * drain.
 * From this call until the admitted batch has landed, the pop priority
 * refills only writes: reads already on the device finish over the batch's
 * head, and no read is refilled until the batch has landed. */
void nvme_engine_write_drain(nvme_engine_t* E) {
    if (!E || E->wfd < 0) return;
    eng_side_t* S = &E->s[NVME_ENGINE_WRITE];
    pthread_mutex_lock(&S->mu);
    S->admit_upto = S->qtail;
    pthread_mutex_unlock(&S->mu);
    eng_kick(E);
}

int nvme_engine_reap(nvme_engine_t* E, int side,
                     uint64_t* uds, int64_t* ress, unsigned cap)
{
    if (!E || side < 0 || side > 1 || !cap) return 0;
    eng_side_t* S = &E->s[side];
    if (!S->q) return 0;
    unsigned n = 0;
    pthread_mutex_lock(&S->mu);
    while (n < cap && S->ohead != S->otail) {
        uds [n] = S->o_ud [S->ohead % NVME_ENGINE_QCAP];
        ress[n] = S->o_res[S->ohead % NVME_ENGINE_QCAP];
        S->ohead++;
        S->outstanding--;
        n++;
    }
    pthread_mutex_unlock(&S->mu);
    return (int)n;
}

int nvme_engine_out_fd(nvme_engine_t* E, int side) {
    if (!E || side < 0 || side > 1 || !E->s[side].q) return -EINVAL;
    return E->s[side].out_efd;
}

unsigned nvme_engine_out_ready(nvme_engine_t* E, int side) {
    eng_side_t* S = &E->s[side];
    if (!S->q) return 0;
    pthread_mutex_lock(&S->mu);
    unsigned n = (unsigned)(S->otail - S->ohead);
    pthread_mutex_unlock(&S->mu);
    return n;
}

unsigned nvme_engine_inflight(nvme_engine_t* E, int side) {
    /* Engine-thread-written u32; a benign torn read is fine for
     * diagnostics. */
    return E->s[side].inflight;
}

size_t nvme_engine_alignment(nvme_engine_t* E, int side) {
    if (!E || !E->L) return 0;
    if (side == NVME_ENGINE_WRITE)
        return E->wfd >= 0 ? E->w_align : 0;
    return E->L->f_align[0];
}

uint64_t nvme_engine_file_size(nvme_engine_t* E, unsigned file_idx) {
    nvme_loader_t* L = E->L;
    if (!L || file_idx >= L->n_files) return 0;
    return L->f_size[file_idx];
}

unsigned nvme_engine_queue_cap(void) {
    return NVME_ENGINE_QCAP;
}

/* ---------------------------------------------------------------- stream
 * progress signal. A mark is a stream callback (cuStreamAddCallback) on the
 * producing stream: the driver runs `nvme_stream_signal_cb` once everything
 * queued before it there has completed. It raises the bound counter to its
 * ticket and writes the bound eventfd, so whoever waits on that fd hears a
 * compute point the moment the GPU passes it, without polling. The counter
 * only ever rises (a CAS max), so a ticket that fires after a later one
 * cannot move it back.
 *
 * userData is the ticket. It runs on the driver's callback thread and must
 * not call CUDA; a store and one write(2) is all it does. */
static int g_sig_fd = -1;
static int64_t* g_sig_cnt;

void nvme_stream_signal_bind(int fd, int64_t* counter) {
    g_sig_cnt = counter;
    __atomic_store_n(&g_sig_fd, fd, __ATOMIC_RELEASE);
}

static void nvme_stream_signal_cb(void* stream, int status, void* user) {
    (void)stream;
    (void)status;
    int fd = __atomic_load_n(&g_sig_fd, __ATOMIC_ACQUIRE);
    int64_t* c = g_sig_cnt;
    int64_t t = (int64_t)(uintptr_t)user;
    int64_t cur = __atomic_load_n(c, __ATOMIC_RELAXED);
    while (cur < t && !__atomic_compare_exchange_n(
               c, &cur, t, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED))
        ;
    uint64_t one = 1;
    ssize_t r = write(fd, &one, sizeof one);
    (void)r;
}

/* Stream marks and event ops from Python without letting go of the
 * interpreter lock: the caller binds these through a PyDLL handle, and each
 * is a driver call with no callback into Python. The driver entry points are
 * found in the libcuda the process already loaded (torch); nothing here links
 * it. */
typedef int (*cu_stream_add_callback_t)(void*,
                                        void (*)(void*, int, void*),
                                        void*, unsigned);
typedef int (*cu_event_record_t)(void*, void*);
typedef int (*cu_stream_wait_event_t)(void*, void*, unsigned);
typedef int (*cu_event_create_t)(void**, unsigned);
static cu_stream_add_callback_t p_addcb;
static cu_event_record_t p_record;
static cu_stream_wait_event_t p_wait;
static cu_event_create_t p_create;

static int cu_resolve(void) {
    if (p_addcb && p_record && p_wait && p_create) return 0;
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) return -ENOENT;
    p_addcb  = (cu_stream_add_callback_t)dlsym(h, "cuStreamAddCallback");
    p_record = (cu_event_record_t)dlsym(h, "cuEventRecord");
    p_wait   = (cu_stream_wait_event_t)dlsym(h, "cuStreamWaitEvent");
    p_create = (cu_event_create_t)dlsym(h, "cuEventCreate");
    return (p_addcb && p_record && p_wait && p_create) ? 0 : -ENOSYS;
}

/* A timing-free CUDA event. NULL on failure. */
void* nvme_stream_event_new(void) {
    if (cu_resolve()) return NULL;
    void* ev = NULL;
    if (p_create(&ev, 2 /* CU_EVENT_DISABLE_TIMING */)) return NULL;
    return ev;
}

/* Mark `ticket` on `stream`: the callback runs once everything queued there
 * before it has completed. Returns a CUresult (0 = ok) or a negative errno
 * if the driver entry points are missing. */
int nvme_stream_mark(void* stream, int64_t ticket) {
    int rc = cu_resolve();
    if (rc) return rc;
    return p_addcb(stream, nvme_stream_signal_cb,
                   (void*)(uintptr_t)ticket, 0);
}

/* Record `ev` on `stream`: the event completes when everything queued there
 * before it has. Returns a CUresult (0 = ok) or -errno. */
int nvme_event_record(void* ev, void* stream) {
    int rc = cu_resolve();
    if (rc) return rc;
    return p_record(ev, stream);
}

/* Make `stream` wait, on the device, for the last record of `ev` made
 * before this call. The calling thread does not wait. */
int nvme_stream_wait(void* stream, void* ev) {
    int rc = cu_resolve();
    if (rc) return rc;
    return p_wait(stream, ev, 0);
}
