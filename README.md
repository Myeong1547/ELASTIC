# ELASTIC

[한국어](README_KO.md)

*Edge LLM Architecture for Storage-Aware, Throughput-Oriented Inference and Caching*

ELASTIC is an execution architecture for throughput-oriented offline LLM
inference on memory-constrained edge systems. It offloads both model weights
and the KV cache to NVMe storage on coherent unified-memory systems.

The **ELASTIC Engine** coordinates cache layout, loading, consumption and
writeback to sustain data supply as the cache grows during decoding and to
overlap NVMe I/O with GPU computation.

- **Storage layout:** separates blocks updated during decoding from unchanged
  blocks and groups each step's writes into contiguous ranges within each block
  region.
- **Streaming execution:** prefetches into rotating layer slots and computes
  partial attention over blocks as they become ready, then merges the results.
- **KV/activation hybrid caching:** builds on this engine to reduce NVMe
  traffic by storing part of the cache as activations and recomputing K/V on
  the GPU.

This repository implements ELASTIC on vLLM and currently supports OPT models.

## Requirements

- Linux with io_uring and `O_DIRECT` support.
- A coherent UMA system where the CPU and NVIDIA GPU share physical memory.
- Python 3.11+ and CUDA-enabled PyTorch.
- NVMe space for the staged weights and the KV/activation file.

Developed on DGX Spark (NVIDIA GB10, 128 GB unified memory).
Non-UMA systems are not supported.

## Installation

Run these commands from the repository root in the Python environment you
will use for ELASTIC. vLLM requires the pinned commit and local patch below.

```bash
sudo apt-get install -y build-essential liburing-dev

git clone https://github.com/vllm-project/vllm ../vllm
git -C ../vllm checkout bf610c2f56764e1b30bc6065f4ceace3d6e59036
git -C ../vllm apply "$PWD/patches/vllm-elastic.patch"
VLLM_USE_PRECOMPILED=1 python -m pip install -e ../vllm

python -m pip install -r requirements.txt
make build
make check-deps
```

Keep the patched vLLM installation in this environment; installing a stock
`vllm` package over it removes the required integration.

## Quick start

This small OPT-125M example uses a fixed activation ratio of 0.5. Run from the
repository root, place `staged/` on NVMe, and replace the save path below
with a path on your NVMe filesystem.

```bash
python -m elastic.cli.stage_opt \
    --model facebook/opt-125m --output staged/opt-125m

python -m elastic.cli.run_bench \
    --model facebook/opt-125m --blob-dir staged/opt-125m \
    --alpha 0.5 --slots-mem-gib 96 \
    --n-prompts 8 --prompt-len 512 --gen-len 24 --block-size 16 \
    --kv-save-blob /path/on/nvme/kv.blob \
    --csv-out bench.csv --tag opt-125m_demo
```

The run appends throughput results to `bench.csv`. `--slots-mem-gib` sets the
slot memory budget for the layer slots; adjust it to the available system
memory.

To run hybrid caching with cost-based ratios, follow **stage → measure → solve → bench**
in the [usage guide](docs/USAGE.md#measured-hybrid-configuration).

## Tests

```bash
make test
```

## Documentation

- [Usage guide](docs/USAGE.md): first run, hybrid caching, benchmark outputs and campaign.
- [Implementation](docs/DESIGN.md): ELASTIC Engine data flow, storage layout, attention and slot reuse.
- [NVMe engine](native_loader/README.md): native loader build and API.

## License

[MIT](LICENSE). The vLLM files modified by
[patches/vllm-elastic.patch](patches/vllm-elastic.patch) retain vLLM's
Apache-2.0 license.
