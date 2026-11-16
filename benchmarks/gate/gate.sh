#!/bin/bash
# Bit-exactness gate: every arm runs opt-125m (or bloom-560m) through the
# elastic streaming loader with a rotating set of layer slots and NVMe
# save/reload, and gate_check.py compares the generated tokens against the
# vanilla vLLM reference dumps (ref_*.json, produced by `run_bench
# --reference`). The S05 arm also runs with ELASTIC_WRITE_AUDIT=1 so the
# checker can report the write request shape (requests, events, largest
# request, NVMe engine items and SQEs); the numbers are printed, not judged
# against a rule.
#
# Usage: benchmarks/gate/gate.sh [results-dir]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
RES="${1:-$ROOT/benchmarks/results/gate_$(date +%Y%m%d_%H%M)}"
mkdir -p "$RES"
cp "$ROOT"/benchmarks/gate/ref_*.json "$RES"/
rm -f "$RES"/ela_*.json "$RES"/ela_*.log "$RES"/ela_*.csv
COMMON="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
M=facebook/opt-125m; D=staged/opt-125m

run() {  # tag alpha plen n gen n_slots [env] [model] [blob] [repeats]
  rm -f "$RES/bcn_$1.txt"
  env $COMMON ELASTIC_SLOTS="$6" \
      ELASTIC_LOADER_TIMING=1 ELASTIC_HOOK_BEACON="$RES/bcn_$1.txt" \
      ${7:-X=X} \
      "$PY" -m elastic.cli.run_bench \
    --model "${8:-$M}" --blob-dir "${9:-$D}" \
    --alpha "$2" --kv-save-blob "$RES/kv_$1.blob" --slots-mem-gib 96 \
    --n-prompts "$4" --prompt-len "$3" --gen-len "$5" --block-size 16 \
    --repeats "${10:-1}" \
    --csv-out "$RES/ela_$1.csv" --tag "ela_$1" \
    --dump-tokens "$RES/ela_$1.json" > "$RES/ela_$1.log" 2>&1
  echo "$1 rc=$? tb=$(grep -c 'Traceback' \
"$RES/ela_$1.log" 2>/dev/null)"
  rm -f "$RES/kv_$1.blob"*; sleep 2
}

# The two zig-zag prefill arms (N_seq·S above ELASTIC_PREFILL_MB_TOKENS). SB05
# runs two generations, so a second batch's zig-zag prefill follows the
# first batch's decode.
run SBIG 0.5 512 128 16 6
run SB05 0.5 512  64 16 6 X=X "$M" "$D" 2
run S10  1.0 512  8 24 6
rm -f "$RES/waudit_S05.json"
run S05  0.5 512  8 24 6 "ELASTIC_WRITE_AUDIT=1 ELASTIC_WRITE_AUDIT_OUT=$RES/waudit_S05.json"
run ST5  0.5 515  8 24 6
run SW2  0.5 512  8 24 2
run SW3  1.0 512  8 24 3
run SD5  0.5 512  8 24 6 ELASTIC_ARRIVAL_DRAIN_TOKENS=512
run SBL5 0.5 515  8 24 6 X=X bigscience/bloom-560m staged/bloom-560m
run SW5  0.5 512  8 24 5
run SW7  1.0 512  8 24 7
run SR2  0.5 512  8 24 5 X=X "$M" "$D" 2
# Every layer resident (n_slots == L = 12): the writeback releases a layer's
# visit only after its writes land, and the next step's visit of the same
# layer reuses the same slot — plus a second generation over it (repeats 2).
run SRL  0.5 512  8 24 12 X=X "$M" "$D" 2
run SRL1 1.0 512  8 24 12 X=X "$M" "$D" 2
run SRL5 0.5 515  8 24 12 X=X "$M" "$D" 2
# gen 1: the prefill step's new token is the whole output and no decode step
# runs; the loader goes from one batch's prefill step to the next's.
run SG1  0.5 512  8  1 6 X=X "$M" "$D" 2

GATE_RES="$RES" "$PY" benchmarks/gate/gate_check.py
echo GATE_DONE $(date +%T)
