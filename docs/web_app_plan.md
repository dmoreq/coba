# Web Application Plan — Local Dev Browser Mode

Last updated: 2026-05-24

## Problem

`src/web/main.py` launches as a native desktop window via `ft.app(target=main)`. The codebase is named `web` and the docs describe a web application, but it renders as a desktop app.

## Change

One-line change + Makefile update. No architectural refactor needed for local dev.

### 1) `src/web/main.py` — Flet launch target

Change the `run()` function from desktop to browser mode:

```python
def run() -> None:
    if ft is None:
        raise RuntimeError("Flet is not installed.")
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8550)
```

**What this does:**
- `view=ft.AppView.WEB_BROWSER` — Flet starts its built-in web server and opens the browser, instead of a native OS window.
- `port=8550` — pins the port so `make run-web` always opens at `http://localhost:8550`.

### 2) `Makefile` — run-web target

```makefile
run-web:
	$(UV) sync --extra flet --quiet
	PYTHONPATH=src $(UV) run python -c "from web.main import run; run()"
```

No change needed — it already works. The `run()` function now starts a web server instead of a desktop window.

### 3) `pyproject.toml` — no change needed

Flet's built-in web server is part of the `flet` package. No FastAPI, uvicorn, or `flet_fastapi` required for local dev mode.

## What stays the same

- Global `_session` singleton — fine for single-user local dev. One browser tab at a time.
- `PreferencesStore` → `~/.coba_flet_preferences.json` — unchanged.
- All routes, policies, worlds, lessons, comparison, sandbox — no code changes.
- All 142 tests continue to pass — tests mock `ft = None` and don't invoke `run()`.

## Why not flet_fastapi / production mode?

For local dev, Flet's built-in web server is sufficient. It handles:
- Static file serving
- WebSocket for real-time UI updates
- Single-session state (which is what the app has)

Production mode (flet_fastapi + uvicorn + per-session isolation) would require:
- `_SimSession` scope change from global → per-request
- `_page`, `_pref_store` scope change
- FastAPI + flet_fastapi + uvicorn dependencies
- Session lifecycle management (create on connect, destroy on disconnect)

That's scope creep for local dev only. It can be done as a follow-up phase if needed.

## Verification

```bash
make run-web
# Opens http://localhost:8550 in browser
# App renders in browser tab instead of desktop window
```

```bash
make test-web
# All 142 tests pass
```

```bash
make lint
# Clean
```
