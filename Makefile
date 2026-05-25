# Makefile for coba — run from the coba/ directory

UV := $(shell which uv 2>/dev/null || echo "/Users/quy.doan/anaconda3/bin/uv")

.PHONY: lint format check-types test test-all coverage help

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║              COBA — Core Library Commands              ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  make lint         Ruff linter                         ║"
	@echo "║  make format       Ruff formatter                       ║"
	@echo "║  make check-types  Mypy on src/coba                    ║"
	@echo "║  make test         Core library tests                   ║"
	@echo "║  make test-all     Alias for test                       ║"
	@echo "║  make coverage     Core coverage                        ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""

lint:
	$(UV) run ruff check src/coba tests

format:
	$(UV) run ruff format src/coba tests

check-types:
	$(UV) run mypy src/coba --ignore-missing-imports

test:
	$(UV) run pytest tests/ --ignore=tests/test_shared_sim.py -v -p no:asyncio

test-all: test

coverage:
	PYTHONPATH=src $(UV) run pytest tests/ -v -p no:asyncio \
		--ignore=tests/test_shared_sim.py \
		--cov=src/coba --cov-report=term-missing
