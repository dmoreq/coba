# Contributing to COBA

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/coba.git
cd coba
uv sync --frozen
```

## Development

```bash
make test          # Core library tests
make lint          # Ruff check
make format        # Ruff format
make check-types   # Mypy on src/coba
make coverage      # Coverage report
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(coba): add policy constraint support
fix(coba): validate batch context shapes
test(coba): cover offline evaluation weights
docs: update architecture guide
```

### Scopes

- `coba`: Core bandit library (`src/coba/`)
- `test`: Tests
- `docs`: Documentation

## Pull Request Process

1. Rebase on main: `git rebase origin/main`
2. Run tests: `uv run pytest tests/ -p no:asyncio --ignore=tests/test_shared_sim.py`
3. Lint: `uv run ruff check src/coba tests`
4. Format: `uv run ruff format src/coba tests`
5. Update docs if changing public APIs or behavior

## Code Style

- Python 3.10+ type annotations (`from __future__ import annotations`)
- Deterministic tests with seeded RNGs
- Small public APIs backed by focused tests
- Domain-agnostic core code that operates on numpy arrays and Python data structures

## Project Structure

```text
src/coba/    # Core bandit library
tests/       # Core library tests
docs/        # Documentation
```
