# ELASTIC Usage

[한국어](USAGE_KO.md) · [README](../README.md) · [Implementation](DESIGN.md)

This guide covers staging an OPT model, running the NVMe offloading runtime,
and configuring KV/activation hybrid caching from device costs. `run_bench` runs a
fixed batch of generated prompts; it is the benchmark entry point.

## Prepare the environment

Follow the [installation instructions](../README.md#installation) first,
including the pinned vLLM checkout, local patch, and native loader build.
Run the commands below from the repository root in that same Python environment.
The commands use a Linux shell.

Place `staged/` on the NVMe filesystem used for inference. If necessary, mount
or link that directory to the drive. Replace `/path/on/nvme/kv.blob` in the
examples with a writable path on that filesystem. Keep space for the downloaded
checkpoint, the staged weights, and the KV/activation file.

The examples use a 96 GiB slot memory budget from the 128 GB development
system. Adjust it to leave room in system RAM for the OS, activations, and
framework allocations; the slot memory budget does not cap total process memory.

## Quickstart

This OPT-125M example uses a fixed activation ratio to run the complete
streaming path without measuring costs or running the solver. `--alpha 0.5`
selects a uniform activation ratio of 0.5; `0` uses KV blocks only and `1`
uses activation blocks only.

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

Staging creates `model.bin` and `manifest.json`. Its output directory must be
absent or empty; reuse the staged files for subsequent runs. Add `--verify`
when staging to reread every tensor and compare it with the source bytes.

A successful benchmark prints the final result as JSON and appends a row to
`bench.csv`. Use `--repeats` to run more than one generation batch with the
same initialized vLLM engine; its default is `1`.

## Measured hybrid configuration

Use **stage → measure → solve → bench** to select the activation ratios for a
workload. The following OPT-66B example uses 64 sequences (N_seq), prompt length 1024,
and generation length 128. It needs substantially more disk space and runtime
than the first example.

### 1. Stage the weights

```bash
python -m elastic.cli.stage_opt \
    --model facebook/opt-66b --output staged/opt-66b --verify
```

### 2. Measure the device costs

```bash
python -m elastic.cost.cli_measure_units \
    --model opt-66b --s-op 1024 --gen-len 128 \
    --output units/units_opt-66b_S1024_g128.json
```

This step requires the GPU and reads `staged/opt-66b/model.bin`. Unlike the
staging and benchmark commands, the measurement CLI uses the short model name
(`opt-66b`). It finds the blob at `staged/<model>/model.bin`; there is no
`--blob-dir` option for this command.

The record contains the measured read rate and GPU kernel costs, plus the
configured sustained write rate. The latter comes from
`ELASTIC_UNITS_WRITE_GBPS` (default `1.82` GB/s); set it to a value appropriate
for the drive before measuring. It is not measured by this command.

An existing output file is skipped. Pass `--remeasure` to replace it after
changing the device or measurement conditions. A record can be reused across
numbers of sequences, but the prompt and generation lengths must match. Keep runtime
settings consistent between measurement, solving, and benchmarking.

### 3. Generate the hybrid configuration

```bash
python -m elastic.policy.run_solve \
    --from-units units/units_opt-66b_S1024_g128.json \
    --B 64 --S 1024 --gen-len 128 --d-model 9216 --num-layers 64 \
    --slots-mem-bytes 103079215104 --block-size 16 \
    --out-dir policy_out/opt-66b_B64_S1024_g128
```

`--d-model` and `--num-layers` are the model's hidden size and total decoder
layer count. The slot memory budget here is **96 GiB in bytes**. Use the same
budget and block size in the benchmark.

Check `summary.json` for `feasible: true` before continuing. An infeasible
solve exits with code `3`; inspect its `status` and `sanity.txt`, then adjust
the workload or slot memory budget and solve again.

### 4. Run the benchmark

```bash
python -m elastic.cli.run_bench \
    --model facebook/opt-66b --blob-dir staged/opt-66b \
    --alpha-curve policy_out/opt-66b_B64_S1024_g128/alpha_curve.csv \
    --kv-save-blob /path/on/nvme/kv.blob --slots-mem-gib 96 \
    --n-prompts 64 --prompt-len 1024 --gen-len 128 --block-size 16 \
    --csv-out bench.csv --tag opt-66b_B64_S1024_g128
```

Keep `summary.json` next to `alpha_curve.csv`. The runner checks its number of sequences,
prompt length, and generation length against the benchmark arguments and
rejects a mismatch. Solve again when changing these values or the slot memory
budget.

## Inputs, outputs, and memory

| Item | Purpose |
|---|---|
| `staged/<model>/model.bin` | Packed, 4 KiB-aligned model weights read by the NVMe engine. |
| `staged/<model>/manifest.json` | Weight offsets, tensor shapes, model configuration, and dtype. Keep it with the blob. |
| `units/*.json` | Device cost record used by the solver. |
| `policy_out/<run>/alpha_curve.csv` | Activation ratio for each decoder layer. |
| `policy_out/<run>/summary.json` | Feasibility, workload identity, and slot count. Required beside the curve. |
| Other solver outputs | `lead_profile.csv`, `sanity.txt`, and, for feasible solutions, `alpha_plot.png` and `trajectory_plot.png`. |
| `--kv-save-blob` | Runtime KV/activation file. Use a dedicated file for each concurrent run. |
| `--csv-out` | Throughput CSV; one appended row per repeat. |

`--slots-mem-gib` is the slot memory budget for the rotating layer slots,
including their weight, K, V, and activation buffers. It does not bound
transient activations or other process allocations. On UMA hardware these all
consume system RAM. Lower the budget or workload if the system runs out of
memory.

The KV/activation file reserves its full storage layout with `posix_fallocate`;
it is not a sparse placeholder. Its size grows with the model, number of sequences,
and maximum sequence length. A fixed `--alpha 0` still needs this file because
the runtime saves and reloads KV blocks. Reuse it only for sequential runs;
the runner does not delete it after finishing.

The benchmark creates prompts with exactly `--prompt-len` tokens and admits
the whole batch in one prefill step. Generation is greedy and ignores EOS
to keep every sequence active for `--gen-len` tokens. Keep prompt plus
generation length within the model's configured position limit.

The main CSV fields are:

| Field | Meaning |
|---|---|
| `init_s` | vLLM engine initialization time, recorded on the first repeat. |
| `wall_s` | Elapsed time of the generation call. |
| `toks_s` | Generated tokens divided by `wall_s`, including prefill time. |
| `decode_toks_s` | Decode-only throughput for steps after the first generated token, timed with CUDA events. Empty when no decode step occurs. |
| `prefill_wall_s` | `wall_s` minus the measured decode time; includes prompt processing and prefill. |
| `prompt_toks`, `gen_toks` | Actual prompt and generated token counts. |

Rows also identify the model, workload, activation-ratio configuration, slot
memory budget, tag, and repeat. See [the runner](../elastic/cli/run_bench.py)
for the complete schema and `--help` for less common options.

For a model without usable weights, `stage_opt --dummy` accepts a local
directory containing `config.json`, such as `configs/opt-175b`. Pass the same
directory as `run_bench --model`. This uses synthesized weights at full model
size, so the results are throughput measurements; generated text is not meaningful.

## Tests and campaigns

Run the repository test suite in the installed environment:

```bash
make test
```

Some tests require CUDA. The suite covers components of the implementation;
benchmarking the complete NVMe/GPU path still requires the target machine.
For generated-token comparisons on a model that fits in memory, `run_bench`
also offers `--dump-tokens` and an in-memory vLLM `--reference` run. Use the
same model, number of sequences, prompt length, and generation length for both runs.

To generate the multi-workload experiment notebook:

```bash
python benchmarks/build_campaign_nb.py
```

Edit [the builder](../benchmarks/build_campaign_nb.py), then regenerate
`benchmarks/elastic_campaign.ipynb` and run its cells in the installed
environment. Set the model and workload grids, slot memory budget, and NVMe
device in the configuration cell. Review its machine-specific SMART checks and
`fstrim /` command before running the campaign on another system.

The campaign stages models, obtains cost records, solves each workload, and
runs each benchmark in a separate process. Results go to
`benchmarks/results/campaign_<timestamp>/`, including `policy_master.csv`,
`thru.csv`, `throughput.jsonl`, per-run logs, and plots. Cost records are
cached in `benchmarks/results/units_cache/`. Set `ELASTIC_CAMPAIGN_DIR` to an
existing campaign directory to resume; successfully completed benchmark tags
are skipped.
