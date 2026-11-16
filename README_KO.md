# ELASTIC

[English](README.md)

*Edge LLM Architecture for Storage-Aware, Throughput-Oriented Inference and Caching*

ELASTIC은 메모리가 제한된 엣지 시스템에서 오프라인 LLM 추론의 처리량을 높이기
위한 실행 아키텍처입니다. 캐시 일관성이 유지되는 통합 메모리 환경에서 모델
가중치와 KV 캐시를 모두 NVMe로 오프로딩합니다.

핵심인 **ELASTIC Engine**은 캐시 레이아웃, 로딩, 어텐션에서의 소비, 라이트백을
함께 조정합니다. 디코딩 중 캐시가 커져도 데이터를 꾸준히 공급하고 NVMe I/O와
GPU 연산을 겹치는 실행 구조입니다.

- **저장 레이아웃:** 디코딩에서 갱신하는 블록과 바뀌지 않는 블록을 분리하고,
  매 스텝의 쓰기를 각 블록 영역 안의 연속된 범위로 모읍니다.
- **스트리밍 실행:** 순환 레이어 슬롯에 데이터를 미리 읽고, 준비된 블록부터
  부분 어텐션을 계산한 뒤 결과를 병합합니다.
- **KV/활성화 하이브리드 캐싱:** 이 엔진 위에서 캐시 일부를 활성화로
  저장하고 GPU에서 K/V를 다시 계산해 NVMe 데이터 이동량을 줄입니다.

이 저장소는 ELASTIC을 vLLM 위에 구현하며, 현재 OPT 모델을 지원합니다.

## 요구 환경

- io_uring과 `O_DIRECT`를 지원하는 Linux.
- CPU와 NVIDIA GPU가 물리 메모리를 공유하며 캐시 일관성이 유지되는 UMA 시스템.
- Python 3.11 이상과 CUDA용 PyTorch.
- 스테이징한 가중치와 KV/활성화 파일을 둘 수 있는 NVMe 여유 공간.

DGX Spark(NVIDIA GB10, 통합 메모리 128 GB)에서 개발했습니다.
UMA가 아닌 시스템은 지원하지 않습니다.

## 설치

ELASTIC에 사용할 Python 환경을 활성화한 뒤 저장소 루트에서 실행합니다.
vLLM에는 아래의 지정 커밋과 저장소 패치가 필요합니다.

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

같은 환경에 패치된 vLLM을 유지해야 합니다. 일반 `vllm` 패키지를 덮어 설치하면
필요한 연동 부분이 사라집니다.

## 빠른 시작

작은 OPT-125M 모델과 고정 활성화 비율 0.5를 사용하는 예제입니다.
저장소 루트에서 실행하며, `staged/`는 NVMe에 둡니다.
아래 저장 경로도 실제 NVMe 파일 시스템의 경로로 바꿉니다.

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

처리량 결과는 `bench.csv`에 행으로 추가됩니다. `--slots-mem-gib`는 레이어
슬롯의 슬롯 메모리 예산이며, 시스템의 가용 메모리에 맞게 조정합니다.

비용을 바탕으로 비율을 정하는 하이브리드 캐싱을 실행하려면
[상세 사용법](docs/USAGE_KO.md#측정-기반-하이브리드-설정)의
**스테이징 → 측정 → 정책 계산 → 벤치마크** 순서를 따릅니다.

## 테스트

```bash
make test
```

## 문서

- [상세 사용법](docs/USAGE_KO.md): 첫 실행, 하이브리드 캐싱, 벤치마크 출력과 캠페인.
- [구현](docs/DESIGN_KO.md): ELASTIC Engine의 데이터 흐름, 저장 레이아웃, 어텐션과 슬롯 재사용.
- [NVMe 엔진](native_loader/README.md): 네이티브 로더 빌드와 API.

## 라이선스

[MIT](LICENSE). [patches/vllm-elastic.patch](patches/vllm-elastic.patch)로
수정하는 vLLM 파일에는 vLLM의 Apache-2.0 라이선스가 유지됩니다.
