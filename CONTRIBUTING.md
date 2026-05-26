# Contributing to COBA

## Setup

```bash
git clone https://github.com/dmoreq/coba.git
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
2. Run tests: `make test`
3. Lint: `make lint`
4. Format: `make format`
5. Type check: `make check-types`
6. Update docs if changing public APIs or behavior

## Code Style

- Python 3.10+ type annotations (`from __future__ import annotations`)
- Deterministic tests with seeded RNGs
- Small public APIs backed by focused tests
- Domain-agnostic core code that operates on numpy arrays and Python data structures
