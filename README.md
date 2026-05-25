# COBA — Contextual Bandits Educational Platform

[![CI Pipeline](https://github.com/yourusername/coba/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/coba/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-775%20passing-brightgreen)](.)

An interactive educational platform for teaching **17 contextual bandit algorithms** through hands-on, browser-based simulations, built with [Flet](https://flet.dev).

## Quick Start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --frozen

# Launch the web app
uv run python -c "from web.main import run; run()"
```

Open `http://localhost:8550` in your browser.

## Features

### 🎓 17 Algorithms
Context-free: Random, Epsilon-Greedy, UCB1, Thompson Sampling, Softmax
Contextual linear: LinUCB, LinUCB-SW, LinTS, Logistic UCB, GP-UCB
Ensemble: Bootstrapped Ensemble, Tree UCB, Tree TS
Hybrid: LinUCB Hybrid
Continuous: CATS

### 🎮 Interactive Simulation
- Step-by-step mode with 4-phase interaction visualization
- Auto-play with configurable speed
- 7 real-world narrative scenarios (RidePilot, Rural Clinic, MovieMatch, etc.)
- Guided lesson progression with staged objectives

### 🎨 Full Theme Support
- Dark/light mode toggle
- Semantic color tokens (environment teal, agent amber)
- Responsive 3-zone dashboard layout

## Project Structure

```
coba/
├── src/
│   ├── coba/              # Core bandit library (17 algorithms)
│   │   ├── bandit.py       # ClusterBandit public API
│   │   ├── policies/       # Algorithm implementations
│   │   └── drift.py        # Drift detection
│   │
│   └── web/                # Flet web application
│       ├── main.py          # Entry point
│       ├── app.py           # AppShell: navigation, theme, routing
│       ├── components/      # Reusable UI widgets
│       ├── layouts/         # Dashboard layouts
│       ├── theme/           # Color tokens, theme manager
│       ├── statemgmt/       # Event bus, interaction phases
│       ├── analysis/        # Metrics, comparison, diagnostics
│       ├── policies/        # Web-facing policy wrappers
│       ├── worlds/          # 7 narrative simulation worlds
│       ├── curriculum/      # 14 lesson configurations
│       └── ui/              # View models (dataclass layer)
│
├── tests/
│   ├── flet_redesign/       # 198 web app tests
│   │   ├── test_e2e.py      # Session lifecycle E2E tests
│   │   ├── test_edge_cases.py  # 38 edge case tests
│   │   └── ...
│   ├── web/                 # 53 component/theme/state tests
│   └── ...                  # 537 core library tests
│
└── docs/                    # Documentation
```

## Testing

```bash
# Full test suite
uv run pytest tests/ -p no:asyncio --ignore=tests/test_shared_sim.py

# Web-only tests
uv run pytest tests/flet_redesign/ tests/web/ -p no:asyncio

# With coverage
uv run pytest tests/flet_redesign/ tests/web/ --cov=src/web --cov-report=term-missing -p no:asyncio
```

**775 tests passing** (537 core + 238 web), all deterministic (seeded RNG, no async).

## Documentation

- [Quick Start for Learners](./docs/QUICK_START_LEARNER.md)
- [Architecture Guide](./docs/ARCHITECTURE.md)
- [Adding New Lessons](./docs/ADDING_LESSONS.md)
- [Algorithm Reference](./docs/algorithms/)
- [Contributing Guide](./CONTRIBUTING.md)

## License

MIT
