# Contributing to COBA

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/coba.git
cd coba
uv sync --frozen
```

## Development

```bash
make run-web        # Launch Flet web app (localhost:8550)
make test           # Core library tests
make test-web       # Web app tests
make lint           # ruff check
make format         # ruff format
make coverage       # Coverage report
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(web): add environment zone components
fix(web): replace page.session with page.data for Flet 0.85.1
refactor(web): merge arena and comparison into analysis
test(web): add event bus edge case tests
docs: update architecture guide
```

### Scopes
- `web`: Flet web application (`src/web/`)
- `coba`: Core bandit library (`src/coba/`)
- `test`: Tests
- `docs`: Documentation

## Pull Request Process

1. Rebase on main: `git rebase origin/main`
2. Run full test suite: `uv run pytest tests/ -p no:asyncio --ignore=tests/test_shared_sim.py`
3. Lint: `uv run ruff check src/`
4. Format: `uv run ruff format src/`
5. Update docs if changing public API or adding features

## Code Style

- Python 3.10+ type annotations (`from __future__ import annotations`)
- Frozen dataclasses for immutable view-models (`@dataclass(frozen=True)`)
- Protocols for interfaces (`typing.Protocol`)
- No third-party UI libraries — pure Flet 0.85.1
- Theme colors via `ThemeManager.get_tokens(page)`, never hardcoded hex

## Project Structure

```
src/web/             # Flet web app (not next.js)
src/coba/            # Core bandit library
tests/flet_redesign/ # 198 web tests
tests/web/           # 53 component/theme/state tests
tests/               # 537 core library tests
```
