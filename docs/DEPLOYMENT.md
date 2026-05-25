# Deployment Guide

## Local Development

```bash
# Install dependencies
uv sync --frozen

# Launch web app
uv run python -c "from web.main import run; run()"
```

Open `http://localhost:8550` in your browser.

## Makefile Targets

```bash
make run-web     # uv run python -c "from web.main import run; run()"
make test        # Core library tests
make test-web    # Web app tests
make coverage    # Coverage report
make lint        # Ruff + mypy
make format      # Ruff format
```

## Production Deployment

The app runs as a Flet web server. Deploy as a long-running process:

```bash
# Install deps, start on port 8550
uv sync --frozen
uv run python -c "from web.main import run; run()"
```

Environment considerations:
- Set `FLET_WEB=true` for web mode (default)
- Expose port 8550
- Use a reverse proxy (nginx/caddy) for TLS termination

## Testing

```bash
uv run pytest tests/ -p no:asyncio --ignore=tests/test_shared_sim.py
```
