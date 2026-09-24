PY ?= python
VENV = backend/.venv
PYBIN = $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
  PYBIN = $(VENV)/Scripts/python.exe
endif

.PHONY: help setup backend frontend samples test lint typecheck eval build demo docker-up docker-down clean train-bundle train-notebook

help:
	@echo "AQUA-SENTINEL targets:"
	@echo "  make setup          create venv + install backend deps"
	@echo "  make samples        prepare real-data subset + render synthetic SSS surveys"
	@echo "  make lint           ruff check + format --check (backend), eslint (frontend)"
	@echo "  make typecheck      mypy (backend), tsc -b --noEmit (frontend)"
	@echo "  make test           run backend pytest suite"
	@echo "  make build          build the frontend bundle"
	@echo "  make eval           run the heuristic-baseline evaluation harness"
	@echo "  make demo           single-port demo: build frontend + run backend on :8000"
	@echo "  make docker-up      docker compose up --build (backend :8000, frontend :5173)"
	@echo "  make train-bundle   build aqua_train_bundle.zip for Colab training"
	@echo "  train-notebook      open training/aqua_training.ipynb in Colab (manual upload)"

setup:
	cd backend && $(PY) -m venv .venv
	cd backend && $(PYBIN) -m pip install -e ".[dev]"

samples:
	cd backend && $(PYBIN) scripts/prepare_sample_data.py
	cd backend && $(PYBIN) scripts/generate_synthetic_sss.py

test:
	cd backend && $(PYBIN) -m pytest tests/

lint:
	cd backend && $(PYBIN) -m ruff check app scripts tests
	cd backend && $(PYBIN) -m ruff format --check app scripts tests
	cd frontend && npm run lint

typecheck:
	cd backend && $(PYBIN) -m mypy app
	cd frontend && npm run typecheck

eval:
	cd backend && $(PYBIN) scripts/evaluate.py

build:
	cd frontend && npm install --no-audit --no-fund
	cd frontend && npm run build

demo: build
	cd backend && $(PYBIN) -m uvicorn app.main:app --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf frontend/node_modules frontend/dist backend/.venv backend/.pytest_cache
	find backend frontend -type d -name __pycache__ -prune -exec rm -rf {} +

# --- GPU training (Colab) ---------------------------------------------------
# Stage 1 runs locally (CPU, packaging only). Stages 2-6 live in the notebook.
train-bundle:
	cd backend && $(PYBIN) ../training/package_for_colab.py --out ../aqua_train_bundle.zip
	@echo "Bundle ready: aqua_train_bundle.zip — upload to Colab per training/README.md"
