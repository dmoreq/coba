# Makefile for coba — run from the coba/ directory

UV := $(shell which uv 2>/dev/null || echo "/Users/quy.doan/anaconda3/bin/uv")

.PHONY: run-web coverage lint lint-all format check-types check-types-web test test-web test-all help

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║              COBA — Monorepo Commands                   ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  make run-web      Start the Dash simulation web app    ║"
	@echo "╠──────────────────────────────────────────────────────────╣"
	@echo "║  make lint         Ruff linter (whole workspace)        ║"
	@echo "║  make format       Black formatter (whole workspace)     ║"
	@echo "║  make check-types  Mypy strict check on src/coba        ║"
	@echo "║  make check-types-web  Mypy relaxed check on web/       ║"
	@echo "╠──────────────────────────────────────────────────────────╣"
	@echo "║  make test         Core library tests (src/)            ║"
	@echo "║  make test-web     Dash web app tests (web/)            ║"
	@echo "║  make test-all     All 594 tests (src/ + web/)          ║"
	@echo "║  make coverage     Core tests with 90% cov threshold    ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""

# ── Web App ──────────────────────────────────────────────────────────────────
run-web:
	$(UV) run --package coba-web python web/app.py

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	$(UV) run ruff check .

format:
	$(UV) run black .

# Strict type-checking on the pure-Python core library
check-types:
	$(UV) run mypy src/coba --ignore-missing-imports

# Relaxed type-checking on the Dash web application (callbacks are untyped by design)
check-types-web:
	$(UV) run mypy web --ignore-missing-imports \
		--exclude 'web/(\.venv|tests)' \
		--disable-error-code no-untyped-def \
		--disable-error-code no-untyped-call

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(UV) run pytest tests/ --ignore=tests/test_shared_sim.py -v -p no:asyncio

test-web:
	$(UV) run --package coba-web python -m pytest web/tests/ -v -p no:asyncio

test-all: test test-web

coverage:
	$(UV) run pytest tests/ --ignore=tests/test_shared_sim.py -v -p no:asyncio \
		--cov=. --cov-report=term-missing --cov-fail-under=90
