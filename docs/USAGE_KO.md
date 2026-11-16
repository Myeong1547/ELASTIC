# ELASTIC 사용법

[English](USAGE.md) · [README](../README_KO.md) · [구현](DESIGN_KO.md)

OPT 모델을 스테이징하고 NVMe 오프로딩 런타임을 실행하는 방법과, 장치 비용을 바탕으로
KV/활성화 하이브리드 캐싱을 설정하는 과정을 설명합니다. `run_bench`는 생성된
프롬프트로 고정 배치를 실행하는 벤치마크 진입점입니다.

## 실행 준비

먼저 [설치 안내](../README_KO.md#설치)에 따라 지정된 vLLM 버전과 로컬 패치를
설치하고 네이티브 로더를 빌드합니다. 아래 명령은 모두 저장소 루트에서 같은
Python 환경을 활성화한 상태로 실행합니다. 명령 형식은 Linux 셸 기준입니다.

`staged/`는 추론에 사용할 NVMe 파일시스템에 둡니다. 필요하면 해당 디렉터리를
NVMe에 마운트하거나 링크합니다. 예제의 `/path/on/nvme/kv.blob`도 그 파일시스템의
쓰기 가능한 경로로 바꿉니다. 다운로드한 체크포인트, 스테이징된 가중치,
KV/활성화 파일을 위한 여유 공간을 확보합니다.

예제의 슬롯 메모리 예산 96 GiB는 메모리 128 GB인 개발 장비에서 사용한 값입니다.
운영체제, 활성화 텐서, 프레임워크 할당에 사용할 시스템 RAM을 남기도록 조정합니다.
슬롯 메모리 예산은 프로세스 전체 메모리의 상한이 아닙니다.

## 빠른 시작

이 OPT-125M 예제는 고정 활성화 비율을 지정해 비용 측정이나 솔버 실행 없이
전체 스트리밍 경로를 실행합니다. `--alpha 0.5`는 모든 레이어에 활성화 비율 0.5를
적용합니다. `0`은 KV 블록만, `1`은 활성화 블록만 사용합니다.

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

스테이징하면 `model.bin`과 `manifest.json`이 생성됩니다. 출력 디렉터리는 없거나
비어 있어야 하며, 이후 실행에서는 만들어 둔 파일을 재사용합니다. 스테이징 시
`--verify`를 추가하면 모든 텐서를 다시 읽어 원본 바이트와 비교합니다.

벤치마크가 성공하면 마지막 결과를 JSON으로 출력하고 `bench.csv`에 한 행을
추가합니다. `--repeats`를 지정하면 초기화한 vLLM 엔진에서 생성 배치를 여러 번
실행합니다. 기본값은 `1`입니다.

## 측정 기반 하이브리드 설정

**스테이징 → 측정 → 솔버 → 벤치마크** 순서로 워크로드에 사용할 활성화 비율을
구합니다. 아래 OPT-66B 예제는 시퀀스 수(N_seq) 64, 프롬프트 길이 1024, 생성 길이
128을 사용합니다. 첫 예제보다 훨씬 많은 디스크 공간과 실행 시간이 필요합니다.

### 1. 가중치 스테이징

```bash
python -m elastic.cli.stage_opt \
    --model facebook/opt-66b --output staged/opt-66b --verify
```

### 2. 장치 비용 측정

```bash
python -m elastic.cost.cli_measure_units \
    --model opt-66b --s-op 1024 --gen-len 128 \
    --output units/units_opt-66b_S1024_g128.json
```

이 단계에는 GPU가 필요하며 `staged/opt-66b/model.bin`을 읽습니다.
측정 CLI는 스테이징·벤치마크 명령과 달리 짧은 모델 이름(`opt-66b`)을
사용합니다. blob은 `staged/<model>/model.bin`에서 찾으며, 이 명령에는
`--blob-dir` 옵션이 없습니다.

레코드에는 실측한 읽기 속도와 GPU 커널 비용, 설정된 지속 쓰기 속도가 들어갑니다.
지속 쓰기 속도는 `ELASTIC_UNITS_WRITE_GBPS`의 값(기본 `1.82` GB/s)을 사용하므로
측정 전에 사용할 드라이브에 맞는 값으로 설정합니다. 이 명령이 쓰기 속도를
실측하는 것은 아닙니다.

출력 파일이 이미 있으면 측정을 건너뜁니다. 장치나 측정 조건을 바꾼 뒤에는
`--remeasure`를 지정해 덮어씁니다. 같은 레코드를 여러 시퀀스 수에 재사용할 수
있지만 프롬프트 길이와 생성 길이는 일치해야 합니다. 측정·솔버·벤치마크의
런타임 설정도 일관되게 유지합니다.

### 3. 하이브리드 설정 생성

```bash
python -m elastic.policy.run_solve \
    --from-units units/units_opt-66b_S1024_g128.json \
    --B 64 --S 1024 --gen-len 128 --d-model 9216 --num-layers 64 \
    --slots-mem-bytes 103079215104 --block-size 16 \
    --out-dir policy_out/opt-66b_B64_S1024_g128
```

`--d-model`은 모델의 hidden size, `--num-layers`는 전체 디코더 레이어 수입니다.
여기서는 슬롯 메모리 예산 **96 GiB를 바이트 단위로** 지정합니다. 벤치마크에서도 같은
예산과 블록 크기를 사용합니다.

진행하기 전에 `summary.json`의 `feasible`이 `true`인지 확인합니다. 실행 가능한
해를 구하지 못하면 솔버는 종료 코드 `3`을 반환합니다. `status`와 `sanity.txt`를
확인하고 워크로드나 슬롯 메모리 예산을 조정한 뒤 다시 풉니다.

### 4. 벤치마크 실행

```bash
python -m elastic.cli.run_bench \
    --model facebook/opt-66b --blob-dir staged/opt-66b \
    --alpha-curve policy_out/opt-66b_B64_S1024_g128/alpha_curve.csv \
    --kv-save-blob /path/on/nvme/kv.blob --slots-mem-gib 96 \
    --n-prompts 64 --prompt-len 1024 --gen-len 128 --block-size 16 \
    --csv-out bench.csv --tag opt-66b_B64_S1024_g128
```

`summary.json`은 `alpha_curve.csv`와 같은 디렉터리에 둡니다. 실행기는 이 파일의
시퀀스 수·프롬프트 길이·생성 길이를 벤치마크 인자와 비교하며, 다르면 실행을
거부합니다. 이 값들이나 슬롯 메모리 예산을 바꾸면 솔버를 다시 실행합니다.

## 입력·출력과 메모리

| 항목 | 용도 |
|---|---|
| `staged/<model>/model.bin` | NVMe 엔진이 읽는 패킹된 4 KiB 정렬 모델 가중치. |
| `staged/<model>/manifest.json` | 가중치 오프셋, 텐서 형상, 모델 설정, dtype. blob과 함께 보관합니다. |
| `units/*.json` | 솔버가 사용하는 장치 비용 레코드. |
| `policy_out/<run>/alpha_curve.csv` | 디코더 레이어별 활성화 비율. |
| `policy_out/<run>/summary.json` | 실행 가능 여부, 워크로드 조건, 슬롯 수. 곡선 옆에 있어야 합니다. |
| 그 밖의 솔버 출력 | `lead_profile.csv`, `sanity.txt`, 실행 가능한 해가 있을 때의 `alpha_plot.png`와 `trajectory_plot.png`. |
| `--kv-save-blob` | 실행 중 KV 블록과 활성화 블록을 저장하는 KV/활성화 파일. 동시 실행마다 별도 파일을 사용합니다. |
| `--csv-out` | 반복마다 한 행씩 추가하는 처리량 CSV. |

`--slots-mem-gib`는 가중치 버퍼, K 버퍼와 V 버퍼, 활성화 버퍼를 포함한 순환 레이어
슬롯의 슬롯 메모리 예산입니다. 일시적인 활성화 텐서와 다른 프로세스
할당량까지 제한하지는 않습니다. UMA에서는 모두 시스템 RAM을 사용하므로 메모리가
부족하면 슬롯 메모리 예산이나 워크로드 크기를 줄입니다.

KV/활성화 파일은 `posix_fallocate`로 전체 저장 레이아웃에 필요한 공간을
예약하며, 희소 파일이 아닙니다. 모델·시퀀스 수·최대 시퀀스 길이에 따라
커집니다. `--alpha 0`도 KV 블록을 저장하고 다시 읽으므로 이 파일이 필요합니다.
파일 재사용은 순차 실행에서만 하며, 실행기는 종료 후 파일을 삭제하지 않습니다.

벤치마크는 정확히 `--prompt-len`개의 토큰으로 된 프롬프트를 만들고, 전체 배치를
하나의 프리필 스텝에 수용합니다. 생성은 탐욕 방식이며 EOS를 무시하여 모든
시퀀스가 `--gen-len`만큼 실행되도록 합니다. 프롬프트와 생성 길이의 합은 모델에
설정된 위치 길이 제한을 넘지 않도록 합니다.

주요 CSV 필드는 다음과 같습니다.

| 필드 | 의미 |
|---|---|
| `init_s` | vLLM 엔진 초기화 시간. 첫 반복에 기록됩니다. |
| `wall_s` | 생성 호출의 전체 경과 시간. |
| `toks_s` | 생성 토큰 수를 `wall_s`로 나눈 값. 프리필 시간이 포함됩니다. |
| `decode_toks_s` | 첫 생성 토큰 이후의 디코드 스텝 처리량. CUDA 이벤트로 측정하며, 디코드 스텝이 없으면 비어 있습니다. |
| `prefill_wall_s` | `wall_s`에서 디코드 시간을 뺀 값. 프롬프트 처리와 프리필을 포함합니다. |
| `prompt_toks`, `gen_toks` | 실제 프롬프트 토큰 수와 생성 토큰 수. |

행에는 모델, 워크로드, 활성화 비율 설정, 슬롯 메모리 예산, 태그, 반복 번호도 기록됩니다.
전체 스키마는 [실행기 코드](../elastic/cli/run_bench.py), 나머지 옵션은 `--help`를
참고하세요.

사용할 수 있는 가중치가 없는 모델은 `stage_opt --dummy`로 스테이징할 수 있습니다.
이때 `--model`에는 `configs/opt-175b`처럼 `config.json`이 있는 로컬 디렉터리를
전달하고, `run_bench --model`에도 같은 디렉터리를 전달합니다. 전체 모델 크기의
합성 가중치를 사용하므로 처리량 측정용이며, 생성 텍스트에는 의미가 없습니다.

## 테스트와 실험 캠페인

설치한 환경에서 저장소 테스트를 실행합니다.

```bash
make test
```

일부 테스트에는 CUDA가 필요합니다. 이 테스트들은 구현 구성요소를 검사하며,
전체 NVMe/GPU 경로의 벤치마크는 대상 장비에서 실행해야 합니다.
메모리에 들어가는 모델의 생성 토큰을 비교할 때는 `run_bench`의 `--dump-tokens`와
메모리 상주 vLLM 실행인 `--reference`도 사용할 수 있습니다. 두 실행의 모델,
시퀀스 수, 프롬프트 길이, 생성 길이를 같게 맞춥니다.

여러 워크로드를 실행하는 실험 노트북은 다음과 같이 생성합니다.

```bash
python benchmarks/build_campaign_nb.py
```

[생성기](../benchmarks/build_campaign_nb.py)를 수정한 뒤
`benchmarks/elastic_campaign.ipynb`를 다시 생성하고, 설치한 환경에서 셀을
실행합니다. 설정 셀에서 모델·워크로드 그리드, 슬롯 메모리 예산, NVMe 장치를
지정합니다. 다른 시스템에서 캠페인을 실행하기 전에는 장비별 SMART 검사와
`fstrim /` 명령을 확인합니다.

캠페인은 모델 스테이징, 비용 레코드 확보, 워크로드별 솔버 실행을 거쳐
각 벤치마크를 별도 프로세스로 실행합니다. 결과는
`benchmarks/results/campaign_<timestamp>/`에 저장하며, `policy_master.csv`,
`thru.csv`, `throughput.jsonl`, 실행별 로그와 그림을 포함합니다. 비용 레코드는
`benchmarks/results/units_cache/`에 캐시합니다. `ELASTIC_CAMPAIGN_DIR`을 기존
캠페인 디렉터리로 지정하면 재개할 수 있으며, 성공한 벤치마크 태그는 건너뜁니다.
