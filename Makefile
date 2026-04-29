# Makefile for coban — run from the coban/ directory

.PHONY: coverage lint format test help

help:
	@echo "Available targets:"
	@echo "  make lint      — Run Ruff linter"
	@echo "  make format    — Run Black formatter"
	@echo "  make test      — Run pytest (no coverage)"
	@echo "  make coverage  — Run pytest with 90% coverage threshold"

lint:
	ruff check .

format:
	black .

test:
	pytest tests/ -v -p no:asyncio

coverage:
	pytest tests/ -v -p no:asyncio --cov=. --cov-report=term-missing --cov-fail-under=90
