# Transformer Surgery Lab — developer commands
# Requires: Docker with compose plugin (or docker-compose standalone)

COMPOSE := docker compose -f docker/docker-compose.yml
IMAGE_DEV := tsl-dev:latest
IMAGE_CPU := tsl-dev-cpu:latest

.PHONY: help build build-cpu shell shell-cpu test lint format prepare-data prepare-data-smoke smoke train train-mini evaluate benchmark compare-runs export-dashboard dashboard clean

help:
	@echo "Transformer Surgery Lab — available targets:"
	@echo "  build         Build GPU dev image"
	@echo "  build-cpu     Build CPU-only dev image"
	@echo "  shell         Open interactive shell (GPU image)"
	@echo "  shell-cpu     Open interactive shell (CPU image)"
	@echo "  test          Run pytest in CPU container"
	@echo "  lint          Run ruff check"
	@echo "  format        Run black + ruff format"
	@echo "  prepare-data  Prepare TinyStories dataset"
	@echo "  prepare-data-smoke  Prepare local smoke corpus"
	@echo "  smoke         Run smoke test pipeline"
	@echo "  train         Launch baseline training"
	@echo "  train-mini    Launch mini CPU gate training"
	@echo "  evaluate      Evaluate a run (RUN_DIR=runs/<name>)"
	@echo "  benchmark     Benchmark a run (RUN_DIR=runs/<name>)"
	@echo "  compare-runs  Compare runs (RUNS='runs/a runs/b')"
	@echo "  export-dashboard  Bundle runs into dashboard_export.json"
	@echo "  dashboard     Launch Streamlit dashboard"
	@echo "  clean         Remove local Python caches"

build:
	$(COMPOSE) build dev

build-cpu:
	$(COMPOSE) build dev-cpu

shell:
	$(COMPOSE) run --rm dev bash

shell-cpu:
	$(COMPOSE) run --rm dev-cpu bash

test:
	$(COMPOSE) run --rm dev-cpu pytest tests/ -v

lint:
	$(COMPOSE) run --rm dev-cpu ruff check src tests scripts dashboard

format:
	$(COMPOSE) run --rm dev-cpu black src tests scripts dashboard
	$(COMPOSE) run --rm dev-cpu ruff check --fix src tests scripts dashboard

prepare-data:
	$(COMPOSE) run --rm dev-cpu python scripts/prepare_data.py --config configs/default.yaml configs/data/tinystories.yaml

prepare-data-smoke:
	$(COMPOSE) run --rm dev-cpu python scripts/prepare_data.py --config configs/default.yaml configs/data/smoke.yaml

smoke:
	$(COMPOSE) run --rm dev-cpu python scripts/smoke_test.py --config configs/train/smoke.yaml

train:
	$(COMPOSE) run --rm dev python scripts/train.py --config configs/experiments/baseline_layernorm_sinusoidal_relu_mha.yaml

train-mini:
	$(COMPOSE) run --rm dev-cpu python scripts/train.py --config configs/experiments/baseline_mini_gate.yaml --device cpu

evaluate:
	$(COMPOSE) run --rm dev-cpu python scripts/evaluate.py --run-dir $(RUN_DIR) --device cpu --generate

benchmark:
	$(COMPOSE) run --rm dev-cpu python scripts/benchmark.py --run-dir $(RUN_DIR) --device cpu

compare-runs:
	$(COMPOSE) run --rm dev-cpu python scripts/compare_runs.py --runs $(RUNS)

export-dashboard:
	$(COMPOSE) run --rm dev-cpu python scripts/export_dashboard_data.py $(if $(RUNS),--runs $(RUNS),)

dashboard:
	$(COMPOSE) --profile dashboard up dashboard

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
