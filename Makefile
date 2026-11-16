# ELASTIC — top-level build orchestration.
#
#   make build      build the native NVMe loader + pip install
#   make loader     build only libnvme_loader.so (native_loader/)
#   make install    pip install -e . (assumes container/pip env)
#   make test       pytest elastic/
#   make clean      remove build artefacts + pytest cache + pycache
#   make check-deps probe Python deps + CUDA + .so libs

.PHONY: all build loader install test clean check-deps

all: build

build: loader install

loader:
	$(MAKE) -C native_loader

install:
	pip install -e .

test:
	python -m pytest elastic/ -v

clean:
	$(MAKE) -C native_loader clean || true
	find elastic benchmarks -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

check-deps:
	@echo "=== Python ==="
	@python -c "import numpy, torch, matplotlib, transformers; \
	            print(' OK numpy/torch/matplotlib/transformers')"
	@python -c "import pandas, nbformat; print(' OK pandas/nbformat')" 2>/dev/null \
	    || echo " pandas/nbformat not installed (campaign notebook, run_bench --alpha-curve)"
	@echo "=== CUDA ==="
	@python -c "import torch; print(' available:', torch.cuda.is_available()); \
	            print(' device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
	@echo "=== Native libs ==="
	@test -f native_loader/build/libnvme_loader.so   && echo " loader  OK"   || echo " loader  MISSING (make loader)"
	@echo "=== vLLM (patched editable checkout) ==="
	@python -c "import vllm; print(' vllm', vllm.__version__)" 2>/dev/null || echo " vllm not installed"
