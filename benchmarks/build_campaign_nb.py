"""Generate `benchmarks/elastic_campaign.ipynb`.

Run once: `python benchmarks/build_campaign_nb.py`. Edits go in here, not
in the .ipynb directly, so the notebook stays diff-friendly.

The notebook is a one-shot measurement campaign with two tracks:

  1) policy track (OPT family): measure the unit costs per model with
     `cli_measure_units`, then run the hybrid policy (Algorithm 3)
     per (model, B, S) -> per-layer alpha curve, n_slots, regime, and the
     cost-model predicted decode throughput.
  2) throughput track: `run_bench` per (model, B, S, alpha) as an isolated
     subprocess so a fresh CUDA context starts each run and is fully
     freed on exit. Each run is wrapped by a memory monitor
     (/proc/meminfo MemAvailable: on the unified-memory box nvidia-smi
     reports memory.used = N/A and torch's peak misses the host-registered
     KV).

Results stream to disk incrementally (units/*.json, policy_master.csv,
throughput.jsonl) and the throughput loop resumes by tag, so a crash or
interrupt never forces a full re-run.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "elastic_campaign.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# --------------------------------------------------------------- intro
md(r"""
# ELASTIC 측정 캠페인 (정책 + 처리량)

한 번에 돌리는 측정 캠페인. **두 트랙**:

1. **정책 트랙 (OPT 계열)** — `cli_measure_units`로 모델당 unit cost
   (read_GBps · write_GBps · T_kv_gen · T_attn · T_dense)를 측정 → 하이브리드
   정책(Algorithm 3) `solve()`를 (model, B, S)별로 풀어 레이어별 α값 ·
   n_slots · regime · **cost-model 예측 toks/s** 산출. GPU 디코드
   불필요(정책은 CPU).
2. **처리량 트랙 (§0에서 OPT 계열을 자동 stage)** — `run_bench`를
   (model, B, S, α)별 **서브프로세스**로 실행. 매 run이 새 CUDA 컨텍스트로
   시작해 종료 시 완전히 해제 → 커널 OOM 누적 없음. 각 run을 **메모리 모니터**로
   감싸 시스템 RAM peak(/proc/meminfo MemAvailable)을 수집.

**그리드:** 설정 셀의 `MODELS_POLICY` × `S_GRID` × `B_GRID`, gen_len =
`GEN_LEN`. **α 변형:** `SHOWCASE_MODEL`은 **3-way(α=0 · policy · α=1)**, 나머지는
policy 단점 — 전부 **같은 n_slots**에서(α값만 변경) policy가 두 극단을 이김을
보인다. OPT position cap을 넘는 셀과 feasible 곡선이 없는 셀은 건너뛰고, 그 밖의
실패(OOM/timeout)는 기록하고 다음 셀로 넘어간다.

**산출(증분 저장):** `campaign_<TS>/` 아래 `policy_master.csv`, 셀별
`policy/<model>_B<B>_S<S>/`, `throughput.jsonl`, `figs/*.png`. 처리량 루프는
tag로 resume → 중단/크래시 후 이어서 실행.

> 큰 모델(opt-66b)은 디코드 스텝마다 weight 전량(~122 GB)을 NVMe에서
> 스트리밍하므로 B512 셀은 수 시간 걸린다. 실패(OOM/timeout)는 기록하고 다음
> 셀로 넘어간다.
""")

# -------------------------------------------------------------- config
code(r"""
%matplotlib inline
import os, re, sys, json, time, subprocess, threading
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# repo root = dir holding the `elastic` package. The Jupyter kernel's cwd
# varies (notebook dir / workspace root), so search cwd + ancestors. Set REPO
# by hand if it does not resolve.
def _find_repo():
    here = Path.cwd().resolve()
    for c in [here, *here.parents]:
        if (c / "elastic").is_dir():
            return c
    raise RuntimeError(
        "repo root (dir containing elastic/) not found from "
        f"cwd={here}; set REPO manually in this cell.")

REPO = _find_repo()
sys.path.insert(0, str(REPO))

# ---- grid --------------------------------------------------------------
# opt-175b has no usable checkpoint: §0 stages it with --dummy (stage_opt
# synthesizes the checkpoint from configs/opt-175b: paper-value config + the
# public opt-125m tokenizer), a throughput-only artifact; every other
# pipeline stage is the same machinery.
# For the S=1024 grid with the 3-way and the ablation arms, set
# MODELS_POLICY = ["opt-66b", "opt-175b"], S_GRID = [1024],
# SHOWCASE_MODEL = "opt-66b", ABLATION_CELLS = {("opt-66b", 128), ("opt-66b", 256)}.
MODELS_POLICY = ["opt-66b"]
S_GRID  = [512]
B_GRID  = [64, 128, 256, 512]      # B1024 step times are impractical (hours per run)
GEN_LEN = 128
OPT_MAX_POS = 2048                 # OPT learned-position cap; S+GEN_LEN beyond
                                   # it fails at vLLM engine init

# Per-model α variants. SHOWCASE_MODEL gets the full 3-way — α=0 (every
# block a KV block, the ELASTIC Engine baseline), policy, and α=1 (every
# block an activation block) — so the policy is shown beating both extremes;
# every other model runs the policy curve only. All α variants run at the
# same policy n_slots.
#   variant tag -> alpha_spec: None = per-(model,B,S) policy csv; float = uniform.
SHOWCASE_MODEL = None              # None: policy arm only
# Ablation: uniform α = 0.25 / 0.5 / 0.75 on SHOWCASE_MODEL at the listed
# (model, B) cells, at the same policy n_slots as the cell's other variants.
ABLATION_CELLS = set()
ABLATION_VARIANTS = [("a025", 0.25), ("a05", 0.5), ("a075", 0.75)]


def alpha_variants(model, B=None):
    if model == SHOWCASE_MODEL:
        v = [("a0", 0.0), ("policy", None), ("a1", 1.0)]
        if B is not None and (model, B) in ABLATION_CELLS:
            v += ABLATION_VARIANTS
        return v
    return [("policy", None)]
BLOCK_SIZE  = 16
# Slot memory budget (M_budget): the bytes that size the rotating layer
# slots (M_slot = weight + KV + 0.5·KV activation buffer per slot). Not a
# GPU ceiling — the GB10 is coherent unified memory, so the real limit is
# system RAM; 96 GiB leaves headroom for the OS, framework transients and
# activations, which the figure does not bound.
SLOTS_MEM_BYTES = 96 * 2**30      # ELASTIC_SLOTS_MEM_GIB
SLOTS_MEM_GIB = SLOTS_MEM_BYTES / 2**30
BETA = 0.5                        # β: T_target's fraction from C_floor to
                                  # C_burn; 0.5 = halfway
# vLLM gpu_memory_utilization is not a knob here: the KV pool is sized by
# run_bench's num_gpu_blocks_override and the slot pool by the slot memory
# budget.

# Call volumes, pinned identically for the units measurement (§1) and every
# bench run (§3): one varlen partial attention call per DRAIN_TOKENS of
# arrived blocks, one kv_gen launch per CHUNK_TOKENS of loaded activations.
# §1 measures the per-token costs at these call shapes and the solve refuses
# a record measured at another, so the measurement and the run must issue
# the same shape.
DRAIN_TOKENS = 32768              # ELASTIC_ARRIVAL_DRAIN_TOKENS
CHUNK_TOKENS = 2048               # ELASTIC_LOAD_CHUNK_TOKENS

# Shape env shared by §1 and §3. Every entry changes what is measured or how
# the device is driven, so both sides pin the same values instead of
# inheriting whatever the defaults happen to be.
SHAPE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "VLLM_KV_CACHE_LAYOUT": "NHD",
             "ELASTIC_ARRIVAL_DRAIN_TOKENS": str(DRAIN_TOKENS),
             "ELASTIC_ARRIVAL_MAX_ROWS": "128",
             "ELASTIC_LOAD_CHUNK_TOKENS": str(CHUNK_TOKENS),
             "ELASTIC_UNIFIED_WEIGHT_MB": "16",
             "ELASTIC_READ_QD": "8",
             "ELASTIC_WRITE_INFLIGHT_MB": "32"}

REMEASURE = False                 # True forces a fresh units measurement even
                                  # when a valid per-model record is cached
MEASURE_TIMEOUT_S = 3600
STAGE_TIMEOUT_S   = 36000          # HF download -> fused blob; the largest
                                   # models need hours (download resumes across
                                   # runs); already-staged models skip
# No per-cell bench timeout: heavy cells legitimately take many hours
# streaming TB of KV per step — run them to completion rather than reaping a
# progressing cell as a failure.
BENCH_TIMEOUT_S   = None
NVME_DEV = "nvme0n1"              # the drive under test (SMART log, diskstats)


# OPT layer counts (architecture fact; ALL_MODELS carries hidden/heads but
# not layer count).
NUM_LAYERS = {"opt-125m": 12, "opt-1.3b": 24, "opt-6.7b": 32, "opt-13b": 40,
              "opt-30b": 48, "opt-66b": 64, "opt-175b": 96}

# staged models (blob present) -> only these can be live-benched.
# Re-detected after the staging step (§0) since that may add more, so this is
# just the initial view.
STAGED = {p.parent.name: str(p.parent)
          for p in REPO.glob("staged/*/manifest.json")}
MODELS_THRU = [m for m in MODELS_POLICY if m in STAGED]   # auto-detected

TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
# Resume across notebook re-executions: point ELASTIC_CAMPAIGN_DIR at a
# campaign dir (existing or not) to pin OUT there; the throughput loop then
# skips tags already "ok" in that dir's throughput.jsonl.
_cdir = os.environ.get("ELASTIC_CAMPAIGN_DIR")
OUT = Path(_cdir) if _cdir else (
    REPO / "benchmarks" / "results" / f"campaign_{TS}")
for sub in ("policy", "tokens", "figs", "kvblobs", "logs"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
# units cache is stable across campaign runs — measurement is the slow part,
# so a fresh campaign reuses it. Delete a file here to force its re-measure.
UNITS_DIR = REPO / "benchmarks" / "results" / "units_cache"
UNITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = OUT / "throughput.jsonl"

print("REPO :", REPO)
print("OUT  :", OUT)
print("policy models      :", MODELS_POLICY)
print("staged (benchable) :", MODELS_THRU)
print("S", S_GRID, " B", B_GRID, " gen_len", GEN_LEN)
print(f"α variants: {SHOWCASE_MODEL} -> {{a0, policy, a1}}; others -> {{policy}}")
""")

# ---------------------------------------------------------- helpers
md(r"""
## 헬퍼 — 모델 스펙·RAM 모니터·서브프로세스
""")

code(r"""
from elastic.cost.config import ALL_MODELS


def model_dims(m):
    c = ALL_MODELS[m]
    return c.hidden_size, NUM_LAYERS[m]


def _meminfo():
    d = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            d[k] = int(v.strip().split()[0]) * 1024     # kB -> bytes
    except Exception:
        pass
    return d


class MemMonitor:
    "Background sampler: /proc/meminfo MemAvailable — the unified-memory"
    " box's memory signal (nvidia-smi memory.used=N/A, torch's peak misses"
    " the host-registered KV; system MemAvailable is the honest peak)."
    def __init__(self, interval=0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.mem_avail = []
        self.mem_total = _meminfo().get("MemTotal", 0)
        self.base_avail = self.mem_total

    def _loop(self):
        while not self._stop.is_set():
            mi = _meminfo()
            if "MemAvailable" in mi:
                self.mem_avail.append(mi["MemAvailable"])
            self._stop.wait(self.interval)

    def __enter__(self):
        self.base_avail = _meminfo().get("MemAvailable", self.mem_total)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)

    def summary(self):
        peak = ((self.mem_total - min(self.mem_avail)) / 2**30
                if self.mem_avail else None)
        delta = ((self.base_avail - min(self.mem_avail)) / 2**30
                 if self.mem_avail else None)
        return dict(ram_peak_used_gib=peak, ram_delta_gib=delta,
                    min_avail_gib=(min(self.mem_avail) / 2**30
                                    if self.mem_avail else None),
                    n=len(self.mem_avail))


def _as_text(x):
    # On TimeoutExpired, CPython hands back captured stdout/stderr as BYTES
    # even when subprocess.run was called with text=True — so concatenating a
    # str onto it ("\n[TIMEOUT]") raises TypeError and masks the real timeout.
    if x is None:
        return ""
    return x if isinstance(x, str) else x.decode("utf-8", "replace")


def sh(cmd, timeout=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=e, cwd=str(REPO))
        return dict(rc=p.returncode, wall=time.perf_counter() - t0,
                    out=p.stdout, err=p.stderr, timed_out=False)
    except subprocess.TimeoutExpired as ex:
        return dict(rc=-9, wall=time.perf_counter() - t0,
                    out=_as_text(ex.stdout),
                    err=_as_text(ex.stderr) + "\n[TIMEOUT]",
                    timed_out=True)
""")

# ----------------------------------------------------- staging
md(r"""
## 0) staging (HF → fused NVMe blob) + HF weight 삭제

처리량 트랙은 `staged/<model>/`의 fused blob을 io_uring으로 읽으므로, 실측하려면
모델이 먼저 stage돼 있어야 한다. `MODELS_POLICY`의 모델을 자동 stage(이미 있으면
건너뜀)하고, 체크포인트가 없는 모델(`DUMMY_CONFIG`, opt-175b)은 `configs/`의
config로 `--dummy` 합성한다 → 측정·실측 앞에 둬서 한 번 돌리고 자도 된다.

**HF weight 삭제 (`FREE_HF_AFTER_STAGE`).** stage·verify 성공 후 HF 캐시의 weight
샤드(*.safetensors/*.bin)를 삭제(config/tokenizer는 보존). 런타임 모델 로더는
meta-init이고 `download_model()`이 no-op이라 bench 때 HF weight를 **안 읽으므로** 안전 —
모델당 ~2배 점유(HF 원본 + staged blob, opt-66b ≈ 124G+124G)를 절반으로.
""")

code(r"""
# Repo ids: OPT → facebook/opt-*.
HF_REPO = {m: f"facebook/{m}" for m in MODELS_POLICY}
# Models with no usable checkpoint: the "repo" is a local config dir and
# staging synthesizes the blob (stage_opt --dummy, no hub access). run_bench
# takes the same dir as --model (config + tokenizer).
DUMMY_CONFIG = {"opt-175b": str(REPO / "configs" / "opt-175b")}
HF_REPO.update({m: d for m, d in DUMMY_CONFIG.items() if m in HF_REPO})
FREE_HF_AFTER_STAGE = True        # delete HF weight shards once staged+verified
                                  # (safe: the runtime model loader meta-inits
                                  # and its download_model() is a no-op — HF
                                  # weights are never read at bench time, only
                                  # config/tokenizer)


def stage_model(model):
    "Stage a checkpoint -> fused NVMe blob in staged/<model>/."
    out = REPO / "staged" / model
    if (out / "manifest.json").exists():
        return "cached"
    cmd = [sys.executable, "-m", "elastic.cli.stage_opt",
           "--model", HF_REPO[model], "--output", str(out), "--verify"]
    if model in DUMMY_CONFIG:
        cmd.append("--dummy")
    # Plain HTTP download: the Xet transfer path stalls on this box (a few MB
    # in, then zero throughput), which would hang an unattended campaign.
    r = sh(cmd, timeout=STAGE_TIMEOUT_S, env={"HF_HUB_DISABLE_XET": "1"})
    if r["rc"] == 0 and (out / "manifest.json").exists():
        return "ok"
    return f"FAIL rc={r['rc']}: {r['err'][-400:]}"


def free_hf_weights(repo_id):
    "Drop weight shards (keep config/tokenizer/*.json) from the HF cache. The"
    " staged blob is the runtime source; --verify already proved it matches."
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    d = cache / ("models--" + repo_id.replace("/", "--"))
    if not d.exists():
        return "no hf cache"
    freed = 0
    for snap in (d / "snapshots").glob("*"):
        for pat in ("*.safetensors", "*.bin", "*.msgpack", "*.h5", "*.pt"):
            for f in snap.glob(pat):
                real = f.resolve()                 # blobs/<sha> (symlink target)
                try:
                    if real.exists():
                        freed += real.stat().st_size
                        real.unlink()
                    if f.is_symlink() or f.exists():
                        f.unlink()
                except Exception as e:
                    print("    free warn:", f.name, e)
    return f"freed {freed / 2**30:.1f} GiB"


for m in MODELS_POLICY:
    stat = stage_model(m)
    print(f"[stage] {m:11s}: {stat}")
    # Free HF weights only once the model is actually staged (manifest present).
    if FREE_HF_AFTER_STAGE and (REPO / "staged" / m / "manifest.json").exists():
        print(f"        hf-free {m}: {free_hf_weights(HF_REPO[m])}")

# Re-detect staged set AFTER staging so the throughput track picks up new blobs.
STAGED = {p.parent.name: str(p.parent)
          for p in REPO.glob("staged/*/manifest.json")}
MODELS_THRU = [m for m in MODELS_POLICY if m in STAGED]
print("staged (benchable):", MODELS_THRU)
print("not staged:", [m for m in MODELS_POLICY if m not in STAGED])
""")

# ----------------------------------------------------- measure campaign
md(r"""
## 1) units 측정 (모델 단위 — 서브프로세스)

(model, S, gen)마다 `units_cache/units_<m>_S<S>.json` **하나**를 배치 없이
측정하고, 그 모델의 모든 B가 같은 레코드를 소비한다(`REMEASURE=True`면 강제
재측정).

레코드의 프리미티브는 전부 배치가 없다 — `read_GBps`·`write_GBps`는 바이트
단가, `T_kv_gen_per_token_s`는 런타임과 같은 kv_gen 실행 단위(launch)로 잰
토큰당 kv_gen 시간, `T_attn_per_token_s`는 `arrival_decode`의 32768토큰 부분
어텐션 호출 하나를 여러 번 돌린 중앙값의 토큰당 값, `T_dense_per_token_s`는
같은 토큰 수에서 LN·QKV·out proj·FFN의 토큰당 값(가중치 UMA). B는 §2에서
워크로드 산술로만 가격에 들어간다.

- 읽기 단가: staged `model.bin`(그 모델이 실제로 스트리밍하는 파일)의 서로
  떨어진 네 오프셋을 로더와 같은 형상(16 MiB 세그먼트, read_qd 8, nvme_engine)으로
  읽어 중앙값. 오프셋마다 플래시 영역이 달라 median을 쓴다.
- 쓰기 단가: 드라이브의 쓰기캐시 소진 후 플로어 상수
  (`ELASTIC_UNITS_WRITE_GBPS`, 기본 1.82). 캠페인 셀은 캐시 용량을 훨씬 넘겨
  쓰므로 소진 레짐이 실제로 도는 레짐이다.
- 형상 env(`SHAPE_ENV`)는 §3 벤치와 **같은 값**을 핀한다. 측정과 실행이 다른
  콜 형상이면 다른 기계를 잰 것이 된다.

실패 셀은 기록만 하고 진행한다.
""")

code(r"""
from elastic.policy.from_units import check_units_record

def units_ok(path, S):
    # Usable only if the solve would accept it for this cell — the same rule,
    # `from_units.check_units_record`: the cell's (S, gen), the layer-step
    # and store shapes, and the kv_gen launch and partial attention call
    # volumes this campaign runs.
    try:
        u = json.loads(Path(path).read_text())["units"]
        check_units_record(u, S=S, gen_len=GEN_LEN, kv_gen_tokens=CHUNK_TOKENS)
    except Exception:
        return False
    return True


def run_measure(model, S, force=False):
    # The CLI reads staged/<model>/model.bin, so the read rate is measured on
    # the file the bench actually streams.
    out = UNITS_DIR / f"units_{model}_S{S}.json"
    if out.exists() and not force:
        return out, "cached"
    cmd = [sys.executable, "-m", "elastic.cost.cli_measure_units",
           "--model", model, "--s-op", str(S), "--gen-len", str(GEN_LEN),
           "--output", str(out)]
    if force:
        cmd.append("--remeasure")
    r = sh(cmd, timeout=MEASURE_TIMEOUT_S, env=dict(SHAPE_ENV))
    if r["rc"] == 0 and out.exists():
        return out, "ok"
    return None, f"FAIL rc={r['rc']}: {r['err'][-300:]}"


units_index = {}
for m in MODELS_POLICY:
    for S in S_GRID:
        out = UNITS_DIR / f"units_{m}_S{S}.json"
        force = REMEASURE or not (out.exists() and units_ok(out, S))
        p, stat = run_measure(m, S, force=force)
        if p is not None and not units_ok(p, S):
            p, stat = None, "FAIL: schema/S_op mismatch after measure"
        units_index[(m, S)] = p
        print(f"[measure] {m:10s} S={S:5d}: {stat}")
        if p is not None:
            _u = json.loads(Path(p).read_text())["units"]
            print(f"           read {_u['read_GBps']:.2f} GB/s · write "
                  f"{_u['write_GBps']:.2f} GB/s · T_kv_gen "
                  f"{_u['T_kv_gen_per_token_s']*1e6:.3f} µs/token · attn "
                  f"{_u['T_attn_per_token_s']*1e9:.1f} ns/token · dense "
                  f"{_u['T_dense_per_token_s']*1e6:.3f} µs/token")
print("measured:", sum(v is not None for v in units_index.values()),
      "/", len(units_index))
""")

# ------------------------------------------------------- policy solve
md(r"""
## 2) 정책 풀이 (하이브리드 정책, Algorithm 3, in-process)

(model, B, S)마다 units 레코드 → `units_to_solver_inputs` → `solve()`.

**NVMe 단가는 수기 상수 하나씩** — 읽기 `R_GBPS`, 쓰기 `W_GBPS`. 이 값들이
`t_w = weight_bytes / R` · `t_kv` · `t_save_kv` 세 항을 전부 매긴다
(weight도 같은 드라이브의 읽기이므로 같은 R로). units에서 오는 것은 GPU
프리미티브(`T_kv_gen_per_token_s`, `T_attn_per_token_s`, `T_dense_per_token_s`)와
메모리 쪽 사실들(weight_bytes, `BW_NVMe`)이다.

**컨텍스트는 끝점 S+gen**(런의 최댓값)이다. T_fwd는 시퀀스당 밀집 토큰당 값 +
(S+gen) × 어텐션 토큰당 값, T_kv_gen은 토큰당 단가 × (S+gen).
그래서 pred는 실측이 넘어야 하는 예측이지 평균이 아니다.

가격을 만드는 곳은 `elastic.policy.from_units` **한 곳**이고 `run_solve`와
§2b도 같은 함수를 쓴다 — 한쪽에만 반영되는 가격 층이 생길 수 없다.

산출: 셀별 디렉터리 `policy/<m>_B<B>_S<S>/`에 `alpha_curve.csv` +
`summary.json`(run_solve 스키마 — cell{B,S,gen} + slots; run_bench의 셀
교차검증 가드가 곡선 옆 summary.json을 읽는다), 슬롯 수 `n_slots`, 예측
스텝시간·toks/s.

regime: `ok`/`compute-bound`/`load-bound`는 모두 곡선을 낸다(feasible).
`infeasible`은 neg-mem(한 레이어의 M_slot이 슬롯 메모리 예산 초과) 하나뿐.
""")

code(r"""
from elastic.policy.from_units import load_units, units_to_solver_inputs
from elastic.policy.solver import solve, _cost_constants, _num_slots

# NVMe unit prices, hand-entered from this drive's device counters. One read
# constant and one write constant price every cell: the drive has no model
# axis and no immutable/mutable-block axis. from_units substitutes them for
# the record's own rates in t_w, t_kv and t_save_kv; BW_NVMe keeps the
# MEASURED read rate, because that one converts the slots' weight and K/V
# bytes into C_mem (Eq. 19), the upper bound on the ahead time, instead of
# pricing a term of T_NVMe.
R_GBPS = 13.4     # sustained read service while the NVMe engine is busy
W_GBPS = 1.82     # write floor once the drive's write cache is exhausted


def priced_inputs(model, B, S, units_path):
    # The prices are `from_units`'s — the one place a cell is priced, which
    # `run_solve` uses too: the endpoint context S+gen, T_fwd as the
    # per-token dense cost plus S+gen attention keys per sequence, T_kv_gen
    # per token, and these two drive constants. §2b re-prices its
    # variants through here as well.
    d_model, num_layers = model_dims(model)
    u = load_units(units_path)["units"]
    return units_to_solver_inputs(
        u, n_seq=B, S=S, d_model=d_model, num_layers=num_layers, gen_len=GEN_LEN,
        slots_mem_bytes=SLOTS_MEM_BYTES, beta=BETA, block_size=BLOCK_SIZE,
        read_GBps=R_GBPS, write_GBps=W_GBPS, kv_gen_tokens=CHUNK_TOKENS)


def pred_step_s(inp, alpha):
    "Predicted step time = Σ_i max(T_NVMe(α^i), T_comp(α^i)) at these prices."
    C1, C2, C3, C4 = _cost_constants(inp)
    a = np.asarray(alpha, dtype=float)
    return float(np.sum(np.maximum(C1 - C2 * a, C3 + C4 * a)))


def solve_cell(model, B, S, units_path):
    _, num_layers = model_dims(model)
    inp = priced_inputs(model, B, S, units_path)
    res = solve(inp)
    C1, C2, C3, C4 = _cost_constants(inp)
    n_slots = _num_slots(inp)
    curve = None
    t_compute = t_nvme = step = pred = None
    if res.feasible:
        alpha = np.asarray(res.alpha, dtype=float)
        t_compute = float(np.sum(C3 + C4 * alpha))
        t_nvme = float(np.sum(C1 - C2 * alpha))
        step = pred_step_s(inp, alpha)
        pred = B / step if step > 0 else None
        # Per-cell dir in the run_solve layout: alpha_curve.csv with
        # summary.json next to it. run_bench reads that summary to hard-error
        # on a cell mismatch (a curve solved for one B/S/gen cannot silently
        # drive another).
        cdir = OUT / "policy" / f"{model}_B{B}_S{S}"
        cdir.mkdir(parents=True, exist_ok=True)
        curve = cdir / "alpha_curve.csv"
        pd.DataFrame({"layer": range(num_layers), "alpha": alpha}).to_csv(
            curve, index=False)
        (cdir / "summary.json").write_text(json.dumps(dict(
            feasible=bool(res.feasible), status=res.status,
            cell=dict(B=int(B), S=int(S), gen_len=int(GEN_LEN)),
            slots=int(n_slots),
        ), indent=2))
    return dict(model=model, B=B, S=S, num_layers=num_layers,
                bw_nvme=inp.BW_NVMe,
                feasible=bool(res.feasible), status=res.status,
                alpha_sum=(float(np.nansum(res.alpha)) if res.feasible else None),
                n_slots=n_slots, C1=C1, C2=C2, C3=C3, C4=C4,
                t_compute_s=t_compute, t_load_s=t_nvme, pred_step_s=step,
                pred_toks_s=pred,
                alpha_curve=(str(curve) if curve else None))


pol_rows = []
for m in MODELS_POLICY:
    for S in S_GRID:
        up = units_index.get((m, S))
        if up is None:
            print(f"[policy] skip {m} S={S} (no units)")
            continue
        for B in B_GRID:
            try:
                pol_rows.append(solve_cell(m, B, S, up))
            except Exception as e:
                pol_rows.append(dict(model=m, B=B, S=S, feasible=False,
                                     status=f"ERR:{type(e).__name__}:{e}"))
pol_df = pd.DataFrame(pol_rows)
pol_df.to_csv(OUT / "policy_master.csv", index=False)
print("policy cells:", len(pol_df), " feasible:", int(pol_df.feasible.sum()))
pol_df[["model", "B", "S", "feasible", "status", "alpha_sum",
        "n_slots", "pred_toks_s"]]
""")

# ----------------------------------------------------- per-variant pred
md(r"""
## 2b) 변형별 pred (a0 / policy / a1 — §2와 같은 가격·같은 코드 경로)

셀·α변형마다 예측 스텝시간 Σmax(T_NVMe, T_comp)를 계산해 실측 옆에 둔다.
가격은 §2의 `priced_inputs`를 **그대로** 통과하므로 policy 변형에서는 §2의
`pred_toks_s`와 정의상 일치하고, 이 절이 하는 일은 a0(α≡0)·a1(α≡1)의 pred를
같은 코드 경로에서 얻는 것이다.
""")

code(r"""
def variant_pred_toks_s(model, B, S, alpha_arr):
    up = units_index.get((model, S))
    if up is None or alpha_arr is None:
        return None
    step = pred_step_s(priced_inputs(model, B, S, up), alpha_arr)
    return (B / step) if step > 0 else None


def _variant_alpha_arr(model, B, S, variant):
    _, L = model_dims(model)
    spec = dict(alpha_variants(model, B)).get(variant)
    if isinstance(spec, float):              # uniform α (a0, a1, ablation)
        return np.full(L, spec)
    pr = pol_df[(pol_df.model == model) & (pol_df.B == B) & (pol_df.S == S)]
    if len(pr) and pd.notna(pr.alpha_curve.iloc[0]):
        c = pd.read_csv(pr.alpha_curve.iloc[0]).sort_values("layer")
        return c.alpha.to_numpy()
    return None


pred_rows, pred_map = [], {}
for m in MODELS_POLICY:
    for S in S_GRID:
        for B in B_GRID:
            for variant, _ in alpha_variants(m, B):
                a = _variant_alpha_arr(m, B, S, variant)
                v = variant_pred_toks_s(m, B, S, a)
                if v is not None:
                    pred_map[(m, B, S, variant)] = v
                pred_rows.append(dict(model=m, B=B, S=S, variant=variant,
                                      pred_toks_s=v,
                                      R_gbps=R_GBPS, W_gbps=W_GBPS))
pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(OUT / "pred_variants.csv", index=False)
print("preds computed:", int(pred_df.pred_toks_s.notna().sum()),
      "/", len(pred_df))
pred_df[pred_df.pred_toks_s.notna()]
""")

# --------------------------------------------------- throughput campaign
md(r"""
## 3) 처리량 실측 (staged 전 모델, 서브프로세스 + RAM 모니터)

각 (staged model, B, S)를 α 변형별로 실측 — `SHOWCASE_MODEL`은
α=0 · policy · α=1 3-way, 나머지는 policy 단점(`alpha_variants`). 변형은
**전부 같은 정책 n_slots**로 돌려 활성화 비율 α만 바뀌게 한다. feasible 곡선이
없는 셀과 OPT position cap을 넘는 셀은 건너뛰고, 실행한 셀의 실패(OOM/timeout)는
기록한다. tag로 resume(성공한 것만 건너뜀), jsonl 즉시 append.

런 전 위생 가드: 잔존 vLLM 엔진 프로세스 · uptime ≥ 3 h(부팅 직후에는 장치 읽기
서비스가 낮다) · MemAvailable 바닥 · fstrim · NVMe 온도 ≤ 55 °C(직전 셀의
열이 다음 셀 읽기 대역을 매기면 뒤 셀이 체계적으로 벌받는다).
""")

code(r"""
def _smart_units_written():
    "SMART 'Data Units Written' (1 unit = 1000 x 512 B), or None."
    r = subprocess.run(["sudo", "-n", "nvme", "smart-log", f"/dev/{NVME_DEV}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    m = re.search(r"^Data Units Written\s*:\s*([\d,]+)", r.stdout, re.M)
    return int(m.group(1).replace(",", "")) if m else None


def _parse_metrics(stdout):
    for ln in reversed(stdout.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                pass
    return {}


def run_throughput(model, B, S, alpha_spec, n_slots, tag, pred=None):
    # Pre-run hygiene.
    #   1. A stray vLLM engine from a previous run would double-occupy the
    #      device.
    #   2. Reads are slow for the first hours of uptime, so a campaign resumed
    #      right after a reboot would systematically punish its early cells —
    #      stop rather than wait.
    #   3. MemAvailable near the floor is a system-risk signal on this box.
    #   4. fstrim: a campaign creates and deletes TB-scale KV blobs per cell,
    #      and without trim the write path collapses to the exhausted floor.
    #   5. Cool the drive: the previous cell's heat otherwise prices this
    #      cell's reads. Wait up to 8 minutes, then proceed anyway so a failed
    #      thermometer cannot stall an unattended campaign.
    if subprocess.run(["pgrep", "-x", "VLLM::EngineCor"],
                      capture_output=True).returncode == 0:
        raise RuntimeError(f"{tag}: stray VLLM engine alive")
    _up_s = float(open("/proc/uptime").read().split()[0])
    if _up_s < 3 * 3600:
        raise RuntimeError(
            f"{tag}: uptime {_up_s/3600:.1f} h < 3 h — resume later")
    _ma = _meminfo().get("MemAvailable", 0) / 2**30
    if _ma < 100:
        raise RuntimeError(f"{tag}: MemAvailable {_ma:.0f}G < 100G — abort")
    if subprocess.run(["sudo", "-n", "fstrim", "/"],
                      capture_output=True).returncode != 0:
        subprocess.run(["fstrim", "/"], capture_output=True)
    for _ in range(24):
        _sm = subprocess.run(["sudo", "-n", "nvme", "smart-log",
                              f"/dev/{NVME_DEV}"], capture_output=True, text=True)
        _mt = re.search(r"^temperature\s*:\s*(\d+)", _sm.stdout, re.M) \
            if _sm.returncode == 0 else None
        if _mt is None or int(_mt.group(1)) <= 55:
            break
        time.sleep(20)
    blob = STAGED[model]
    dump = OUT / "tokens" / f"{tag}.json"
    cmd = [sys.executable, "-m", "elastic.cli.run_bench",
           "--model", HF_REPO[model], "--blob-dir", blob,
           "--block-size", str(BLOCK_SIZE),
           "--n-prompts", str(B), "--prompt-len", str(S),
           "--gen-len", str(GEN_LEN), "--repeats", "1",
           "--slots-mem-gib", str(SLOTS_MEM_GIB),
           "--csv-out", str(OUT / "thru.csv"),
           "--dump-tokens", str(dump), "--tag", tag]
    # KV/ACT NVMe save+reload blob: the slot reload is the NVMe offload, and
    # it is the traffic α modulates. The blob lives on the device under test,
    # and BlockKVStorage posix_fallocate's the full worst-case layout up front
    # (real extents, not sparse), so a cell whose layout exceeds free disk
    # fails fast at open and is recorded as such. Removed after the run.
    kvblob = OUT / "kvblobs" / f"{tag}.blob"
    for f in (OUT / "kvblobs").glob(f"{tag}.blob*"):
        try: f.unlink()                      # clear any stale prior attempt
        except Exception: pass
    cmd += ["--kv-save-blob", str(kvblob)]
    if isinstance(alpha_spec, str):              # policy curve csv path
        cmd += ["--alpha-curve", alpha_spec]
    else:                                        # uniform scalar
        cmd += ["--alpha", str(alpha_spec)]
    # Weights come from the local blob and config/tokenizer are already
    # cached, so force offline: a flaky network must not fail a local run.
    # SHAPE_ENV pins the call/IO shapes to the ones §1 measured at.
    env = dict(SHAPE_ENV)
    env.update({
        # Step-time beacon: one line per decode step, which is the only
        # in-run scatter and write-regime signal a campaign row carries.
        "ELASTIC_LOADER_TIMING": "1",
        # Upper bound on the runtime slot count. The activation buffers need
        # no separate setting: _compute_n_slots fits one 0.5·kv activation
        # buffer per slot inside the same slot memory budget the policy sized
        # n_slots from.
        "ELASTIC_SLOTS": str(n_slots)})
    # Device-counter sampler alongside every run: the write-cache regime
    # (in-cache -> exhausted) is only visible in completed-I/O deltas per
    # 10 ms interval.
    _ds_log = OUT / f"ds_{tag}.log"
    # SMART data-units-written on both sides of the run (outside the measured
    # interval) answers what the drive wrote versus what we handed it.
    _sw_pre = _smart_units_written()
    _ds = subprocess.Popen(
        [sys.executable, str(REPO / "benchmarks" / "ds_sampler.py"),
         NVME_DEV, str(_ds_log), "1000000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with MemMonitor() as mon:
            r = sh(cmd, timeout=BENCH_TIMEOUT_S, env=env)
    finally:
        _ds.terminate()
        try: _ds.wait(timeout=10)
        except Exception: _ds.kill()
        for f in (OUT / "kvblobs").glob(f"{tag}.blob*"):
            try: f.unlink()                      # free NVMe space after the run
            except Exception: pass
    _sw_post = _smart_units_written()
    # Persist the run's own stdout/stderr: sh() keeps it in memory and the
    # jsonl row keeps only err_tail on failure, so a successful run's
    # LOADER_TIMING beacons would otherwise be discarded. Written after the
    # subprocess exits, so it cannot perturb the run.
    try:
        (OUT / "logs" / f"{tag}.log").write_text(
            r["out"] + "\n===== STDERR =====\n" + r["err"])
    except Exception as _e:
        print("  log-write warn:", _e)
    metrics = _parse_metrics(r["out"])
    if r["timed_out"]:
        status = "timeout"
    elif r["rc"] == 0 and metrics:
        status = "ok"
    else:
        status = f"FAIL rc={r['rc']}"
    row = dict(tag=tag, model=model, B=B, S=S, gen_len=GEN_LEN,
               variant=tag.rsplit("_", 1)[-1], alpha=str(alpha_spec),
               n_slots=int(n_slots),
               status=status, sub_wall_s=r["wall"],
               toks_s=metrics.get("toks_s"), wall_s=metrics.get("wall_s"),
               # decode_toks_s isolates steady-state decode from the heavy B·S
               # prefill forward: run_bench times the decode steps from the
               # prefill/decode boundary the gate marks on the device at the
               # batch's first decode step. The definition matches FlexGen's
               # decode rate (B·(gen−1)/decode-wall) so decode-vs-decode is
               # comparable; the headline cross-system number stays total
               # wall / toks_s.
               decode_toks_s=metrics.get("decode_toks_s"),
               prefill_wall_s=metrics.get("prefill_wall_s"),
               init_s=metrics.get("init_s"), gen_toks=metrics.get("gen_toks"),
               prompt_toks=metrics.get("prompt_toks"),
               # §2b's pred for this variant, carried on the row so the jsonl
               # itself says whether the run landed above the prediction.
               pred_toks_s=pred,
               below_pred=(bool(metrics.get("decode_toks_s") is not None
                                and pred is not None
                                and metrics["decode_toks_s"] < pred)
                           if status == "ok" else None),
               mem=mon.summary(),
               # Drive-side written bytes over the whole run (SMART Data Units
               # Written, 1 unit = 512000 B) vs what we handed the drive.
               smart_written_gb=((_sw_post - _sw_pre) * 512000 / 1e9
                                 if (_sw_pre is not None
                                     and _sw_post is not None) else None),
               err_tail=("" if status == "ok" else r["err"][-400:]))
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


done = set()
if RESULTS_JSONL.exists():
    for ln in open(RESULTS_JSONL):
        try:
            r = json.loads(ln)
            if r.get("status") == "ok":      # only skip successes -> failures retry
                done.add(r["tag"])
        except Exception:
            pass

for m in MODELS_THRU:
    for S in S_GRID:
        for B in B_GRID:
            pr = pol_df[(pol_df.model == m) & (pol_df.B == B) & (pol_df.S == S)]
            if not (len(pr) and bool(pr.feasible.iloc[0])
                    and pd.notna(pr.alpha_curve.iloc[0])):
                print(f"[thru] skip {m} B={B} S={S}: no feasible policy curve")
                continue
            n_slots = int(pr.n_slots.iloc[0])
            # OPT's learned absolute position embeddings cap at OPT_MAX_POS;
            # prompt_len + gen_len beyond it fails at vLLM engine init.
            if S + GEN_LEN > OPT_MAX_POS:
                print(f"[thru] skip {m} B={B} S={S}: OPT {S}+{GEN_LEN} > "
                      f"{OPT_MAX_POS} pos-cap")
                continue
            # All variants of a cell share the policy n_slots, so the
            # comparison isolates α at a fixed slot count.
            for variant, aspec in alpha_variants(m, B):
                tag = f"{m}_B{B}_S{S}_{variant}"
                if tag in done:
                    print("[thru] skip", tag)
                    continue
                alpha_spec = (pr.alpha_curve.iloc[0] if aspec is None else aspec)
                _pred = pred_map.get((m, B, S, variant))
                row = run_throughput(m, B, S, alpha_spec, n_slots, tag,
                                     pred=_pred)
                _dec = row.get("decode_toks_s")
                _flag = "  measured<pred" if row.get("below_pred") else ""
                print(f"[thru] {tag:32s} {row['status']:8s} "
                      f"toks/s={row['toks_s']} decode={_dec} "
                      f"pred={None if _pred is None else round(_pred, 3)}"
                      f"{_flag} "
                      f"ram_peak={row['mem'].get('ram_peak_used_gib')}")
print("throughput runs on disk:", RESULTS_JSONL)
""")

# ------------------------------------------------------------- load
md(r"""
## 4) 결과 로드
""")

code(r"""
def _load_jsonl(p):
    out = []
    if p.exists():
        for _l in open(p):
            _l = _l.strip()
            if not _l:
                continue
            try:                   # tolerate a torn final line from a kill mid-write
                out.append(json.loads(_l))
            except json.JSONDecodeError:
                pass
    return out

thru = _load_jsonl(RESULTS_JSONL)
tdf = pd.json_normalize(thru) if thru else pd.DataFrame()
if not tdf.empty:
    tdf.to_csv(OUT / "throughput_flat.csv", index=False)
print("throughput rows:", len(tdf),
      "| ok:", int((tdf.status == "ok").sum()) if not tdf.empty else 0)
print("policy rows:", len(pol_df), "| feasible:", int(pol_df.feasible.sum()))
tdf.head() if not tdf.empty else "no throughput rows yet"
""")

# ------------------------------------------------------ viz: throughput
md(r"""
## 5) 시각화

### 5.1 처리량 toks/s vs 배치 (모델별 subplot, S별 라인, 정책 α)
""")

code(r"""
def _ok(df):
    return df[df.status == "ok"].copy() if not df.empty else df

def _ydec(d):
    # decode-only throughput (timed from the prefill/decode boundary, so
    # decode_toks_s is present on any run with a decode step). A row missing
    # it (gen 1, so no boundary) is left as NaN and simply drops from the
    # decode plot rather than being replaced by end-to-end toks_s, which would
    # place a prefill-dominated point on a "decode toks/s" axis.
    return d["decode_toks_s"]

if not tdf.empty:
    pol = _ok(tdf[tdf.variant == "policy"])
    models = [m for m in MODELS_THRU if m in pol.model.unique()]
    fig, axes = plt.subplots(1, max(1, len(models)),
                             figsize=(5 * max(1, len(models)), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = pol[pol.model == m]
        for S in sorted(sub.S.unique()):
            d = sub[sub.S == S].sort_values("B")
            ax.plot(d.B, _ydec(d), marker="o", label=f"S={S}")
        ax.set_title(f"{m} (offload, 정책 α)")
        ax.set_xlabel("batch"); ax.set_ylabel("decode toks/s")
        ax.set_xscale("log", base=2); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "throughput_vs_batch.png", dpi=120)
    plt.show()
else:
    print("no throughput rows")
""")

md(r"""
### 5.2 처리량 toks/s vs 시퀀스 (모델별 subplot, B별 라인, 정책 α)
""")

code(r"""
if not tdf.empty:
    pol = _ok(tdf[tdf.variant == "policy"])
    models = [m for m in MODELS_THRU if m in pol.model.unique()]
    fig, axes = plt.subplots(1, max(1, len(models)),
                             figsize=(5 * max(1, len(models)), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = pol[pol.model == m]
        for B in sorted(sub.B.unique()):
            d = sub[sub.B == B].sort_values("S")
            ax.plot(d.S, _ydec(d), marker="o", label=f"B={B}")
        ax.set_title(f"{m} (offload, 정책 α)")
        ax.set_xlabel("seq len S"); ax.set_ylabel("decode toks/s")
        ax.set_xscale("log", base=2); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "throughput_vs_seq.png", dpi=120)
    plt.show()
else:
    print("no throughput rows")
""")

# -------------------------------------- viz: throughput vs activation ratio α
md(r"""
### 5.2b 처리량 vs 활성화 비율 α (모델별 subplot, 셀별 라인)

x축 = **α value = 활성화 비율**. 셀 (model, B, S)마다 α=0(전부 KV 블록) ·
α=policy(레이어별 곡선의 mean α 위치) · α=1(전부 활성화 블록)을 이어 KV/활성화
하이브리드 캐싱을 본다. 정책 점이 두 극단 대비 어디서 이기는지(하이브리드 이득)를
한눈에. 3점은 `SHOWCASE_MODEL` 셀에만 있고, 나머지 셀은 policy 단점이다.
""")

code(r"""
# mean policy α per (model, B, S) — the x-position of the 'policy' point
# (Σα / num_layers from the policy solve).
mean_alpha = {}
if not pol_df.empty:
    for _, r in pol_df.iterrows():
        if bool(r.feasible) and pd.notna(r.get("alpha_sum")):
            nl = NUM_LAYERS.get(r.model)
            if nl:
                mean_alpha[(r.model, int(r.B), int(r.S))] = float(r.alpha_sum) / nl

def _alpha_x(row):
    v = row["variant"]
    if v == "a0":  return 0.0
    if v == "a1":  return 1.0
    if v == "policy":
        return mean_alpha.get((row["model"], int(row["B"]), int(row["S"])))
    return None

if not tdf.empty:
    d3 = _ok(tdf).copy()
    d3["alpha_x"] = d3.apply(_alpha_x, axis=1)
    d3["ydec"] = _ydec(d3)
    d3 = d3[d3.alpha_x.notna()]
    models = [m for m in MODELS_THRU if m in d3.model.unique()]
    fig, axes = plt.subplots(1, max(1, len(models)),
                             figsize=(5 * max(1, len(models)), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = d3[d3.model == m]
        for (B, S), cell in sub.groupby(["B", "S"]):
            cell = cell.sort_values("alpha_x")
            ax.plot(cell.alpha_x, cell.ydec, marker="o", label=f"B{B} S{S}")
            # mark the policy point so it's distinguishable from the α=0/1 ends.
            pol_pt = cell[cell.variant == "policy"]
            if not pol_pt.empty:
                ax.scatter(pol_pt.alpha_x, pol_pt.ydec, marker="*", s=140,
                           zorder=5, edgecolor="k", linewidth=0.4)
        ax.set_title(f"{m} (offload)")
        ax.set_xlabel("ACT ratio α  (★ = policy mean α)")
        ax.set_ylabel("decode toks/s")
        ax.set_xlim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "throughput_vs_alpha.png", dpi=120)
    plt.show()
else:
    print("no throughput rows")
""")

md(r"""
### 5.2c 실측 vs pred (채점 플롯)

셀·변형별 실측 decode toks/s 대 §2b pred. y=x 위가 정상이고, 아래 점은
비용 모델이 예측한 스텝시간보다 느리게 돈 셀이다.
""")

code(r"""
if not tdf.empty and "pred_toks_s" in tdf.columns:
    d = _ok(tdf)
    d = d[d.pred_toks_s.notna() & d.decode_toks_s.notna()]
    if len(d):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for m in sorted(d.model.unique()):
            s = d[d.model == m]
            ax.scatter(s.pred_toks_s, s.decode_toks_s, s=50, label=m)
        under = d[d.decode_toks_s < d.pred_toks_s]
        if len(under):
            ax.scatter(under.pred_toks_s, under.decode_toks_s, s=160,
                       facecolors="none", edgecolors="r", linewidths=2,
                       label="below pred", zorder=5)
        lim = [0, max(d.decode_toks_s.max(), d.pred_toks_s.max()) * 1.1]
        ax.plot(lim, lim, "k--", alpha=0.5, label="y = x (pred)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted decode toks/s")
        ax.set_ylabel("measured decode toks/s")
        ax.set_title(f"실측 vs pred — 실측이 아래인 셀 {len(under)}건")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "figs" / "pred_check.png", dpi=120)
        plt.show()
        print(under[["tag", "decode_toks_s", "pred_toks_s"]]
              if len(under) else "pred 아래인 셀 0건")
    else:
        print("no scorable rows yet")
else:
    print("no throughput rows / no pred column")
""")

md(r"""
### 5.3 시스템 RAM peak (UMA, 정책 α)

메모리 사용량만 수집한다. RAM peak =
MemTotal − min(MemAvailable) — cudaHostRegister된 슬롯의 K 버퍼와 V 버퍼까지 잡히는
UMA-정직 지표. 예산 초과 조짐(min_avail 바닥)은 여기서 본다.
""")

code(r"""
if not tdf.empty:
    pol = _ok(tdf[(tdf.variant == "policy") & (tdf.model.isin(MODELS_THRU))]).copy()
    pol["ram_peak"] = pol["mem.ram_peak_used_gib"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in sorted(pol.model.unique()):
        for S in sorted(pol[pol.model == m].S.unique()):
            d = pol[(pol.model == m) & (pol.S == S)].sort_values("B")
            ax.plot(d.B, d.ram_peak, marker="o", label=f"{m} S={S}")
    ax.set_title("RAM peak (UMA)"); ax.set_xlabel("batch"); ax.set_ylabel("GiB")
    ax.set_xscale("log", base=2); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "ram_peak.png", dpi=120)
    plt.show()
else:
    print("no throughput rows")
""")

md(r"""
### 5.4 정책 trajectory (레이어별 T_ahead / ΔT / α, B=max)

셀별로 `plot_trajectory`(정책 솔버의 표준 플롯)를 재사용 — x=레이어, 3단:
1) **T_ahead^i** (앞섬 시간) + 앞섬 시간 상·하한선(상한 = C_mem으로 제한된
   C_burn^i = min(C_mem, remaining_burn^i), 하한 = C_floor^i), 2) **ΔT^i**
   레이어별로 얻거나 잃는 시간(bank↑ / burn↓), 3) **α^i** 활성화 비율.
   units가 있는 전 모델을 B=max에서 §2와 같은 가격으로 다시 풀어 그린다.
""")

code(r"""
from elastic.policy.plot_policy import plot_trajectory
from IPython.display import Image, display

Bmax = max(B_GRID)
for m in MODELS_POLICY:
    for S in S_GRID:
        up = units_index.get((m, S))
        if up is None:
            continue
        res = solve(priced_inputs(m, Bmax, S, up))
        print(f"{m:11s} B={Bmax} S={S:5d}: {res.status}")
        if not res.feasible:
            continue
        p = OUT / "figs" / f"trajectory_{m}_B{Bmax}_S{S}.png"
        plot_trajectory(res, p)          # T_ahead / ΔT / α with the bound lines
        display(Image(str(p)))
""")

md(r"""
### 5.5 feasibility / regime 맵 (모델 × B, S별)
""")

code(r"""
order = {m: i for i, m in enumerate(MODELS_POLICY)}
# solver regimes: ok=2 (green); compute-bound/load-bound=1 (yellow, feasible
# with a caveat — curve produced); infeasible (neg-mem)=0 (red).
def _regime_code(status):
    st = str(status)
    if st.startswith("ok"):
        return 2
    if "compute-bound" in st or "load-bound" in st:
        return 1
    return 0
Ssel = sorted(pol_df.S.unique())
fig, axes = plt.subplots(1, len(Ssel), figsize=(4 * len(Ssel), 0.5 * len(MODELS_POLICY) + 1.5),
                         squeeze=False)
for ax, S in zip(axes[0], Ssel):
    M = np.full((len(MODELS_POLICY), len(B_GRID)), np.nan)
    for _, r in pol_df[pol_df.S == S].iterrows():
        if r.model in order and r.B in B_GRID:
            M[order[r.model], B_GRID.index(r.B)] = _regime_code(r.status)
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_xticks(range(len(B_GRID))); ax.set_xticklabels(B_GRID)
    ax.set_yticks(range(len(MODELS_POLICY))); ax.set_yticklabels(MODELS_POLICY)
    ax.set_title(f"regime S={S}\n(green=ok, yellow=compute/load-bound, red=infeasible)")
    ax.set_xlabel("batch")
fig.tight_layout(); fig.savefig(OUT / "figs" / "feasibility_map.png", dpi=120)
plt.show()
""")

md(r"""
### 5.6 cost-model 예측 toks/s + 실측 오버레이

정책의 예측 처리량(B / Σ_i max(T_NVMe, T_comp))을 정책 모델 전부에 대해
표시하고, staged 모델은 실측(정책 α)을 점으로 겹쳐 검증. 예측은 디코드
정상상태 1-스텝 기준, 실측 toks_s는 prefill 포함 wall 기준이라 정확히 같진 않고
경향 비교용.
""")

code(r"""
Ssel = sorted(pol_df.S.unique())
fig, axes = plt.subplots(1, len(Ssel), figsize=(5.5 * len(Ssel), 4), squeeze=False)
meas = _ok(tdf[tdf.variant == "policy"]) if not tdf.empty else pd.DataFrame()
for ax, S in zip(axes[0], Ssel):
    sub = pol_df[(pol_df.S == S) & pol_df.pred_toks_s.notna()]
    for m in MODELS_POLICY:
        d = sub[sub.model == m].sort_values("B")
        if len(d):
            ax.plot(d.B, d.pred_toks_s, marker="o", alpha=0.8, label=f"{m} (pred)")
    if not meas.empty:
        md_ = meas[meas.S == S]
        for m in sorted(md_.model.unique()):
            d = md_[md_.model == m].sort_values("B")
            ax.scatter(d.B, d.toks_s, marker="x", s=80, color="k", zorder=5,
                       label=f"{m} (meas)")
    ax.set_title(f"예측 toks/s  S={S}"); ax.set_xlabel("batch")
    ax.set_ylabel("toks/s"); ax.set_xscale("log", base=2)
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "figs" / "predicted_throughput.png", dpi=120)
plt.show()
""")

# ------------------------------------------------------------ summary
md(r"""
## 6) 요약 저장
""")

code(r"""
summary = dict(
    timestamp=TS, repo=str(REPO), out=str(OUT),
    grid=dict(models_policy=MODELS_POLICY, models_thru=MODELS_THRU,
              S=S_GRID, B=B_GRID, gen_len=GEN_LEN,
              variants=f"{SHOWCASE_MODEL}:{{a0,policy,a1}} others:{{policy}}",
              slots_mem_gib=SLOTS_MEM_GIB),
    policy_cells=len(pol_df), policy_feasible=int(pol_df.feasible.sum()),
    throughput_runs=len(tdf),
    throughput_ok=int((tdf.status == "ok").sum()) if not tdf.empty else 0,
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\nfigs:", sorted(p.name for p in (OUT / "figs").glob("*.png")))
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}  ({len(cells)} cells)")
