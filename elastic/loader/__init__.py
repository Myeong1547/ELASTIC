"""NVMe I/O bindings — the NVMe engine (`NvmeEngine`: one io_uring ring,
one pthread, read/write FIFOs, O_DIRECT) and the aligned host buffers reads
land in.

O_DIRECT is the point, not an optimisation: reads DMA straight into
host-registered buffers the GPU can address, so a loaded page needs no
further copy, and writes never fill memory with dirty page-cache that the
kernel's page-cache writeback would later have to flush against the running
forward pass.

The C library (`libnvme_loader.so`) is built under ``native_loader/build/``
by ``make -C native_loader``, and this binding loads it from there.
"""
