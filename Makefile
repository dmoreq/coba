# Makefile for coba — run from the coba/ directory

UV := $(shell which uv 2>/dev/null || echo "/Users/quy.doan/anaconda3/bin/uv")

.PHONY: run-web lint format check-types check-types-web test test-web test-all coverage help

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║              COBA — Monorepo Commands                   ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  make run-web      Start the Flet bandit lab            ║"
	@echo "╠──────────────────────────────────────────────────────────╣"
	@echo "║  make lint         Ruff linter (src + tests)            ║"
	@echo "║  make format       Ruff formatter (src + tests)          ║"
	@echo "║  make check-types  Mypy on src/coba                    ║"
	@echo "║  make check-types-web  Mypy on src/web                 ║"
	@echo "╠──────────────────────────────────────────────────────────╣"
	@echo "║  make test         Core library tests                   ║"
	@echo "║  make test-web     Flet redesign tests (142 tests)     ║"
	@echo "║  make test-all     All tests                            ║"
	@echo "║  make coverage     Core + web coverage                  ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""

# ── Web App ──────────────────────────────────────────────────────────────────
run-web:
	$(UV) sync --extra flet --quiet
	PYTHONPATH=src $(UV) run python -c "from web.main import run; run()"

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	$(UV) run ruff check src/web tests/flet_redesign

format:
	$(UV) run ruff format src/web tests/flet_redesign

check-types:
	$(UV) run mypy src/coba --ignore-missing-imports

check-types-web:
	$(UV) run mypy src/web --ignore-missing-imports \
		--disable-error-code no-untyped-def \
		--disable-error-code no-untyped-call \
		--disable-error-code annotation-unchecked \
		--disable-error-code union-attr \
		--disable-error-code name-defined \
		--disable-error-code attr-defined

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(UV) run pytest tests/ --ignore=tests/flet_redesign --ignore=tests/test_shared_sim.py -v -p no:asyncio

test-web:
	PYTHONPATH=src $(UV) run pytest tests/flet_redesign -q -p no:asyncio

test-all: test test-web

coverage:
	PYTHONPATH=src $(UV) run pytest tests/ -v -p no:asyncio \
		--cov=src/coba --cov=src/web --cov-report=term-missing
