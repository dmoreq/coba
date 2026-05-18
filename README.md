# COBA Web — Contextual Bandits Educational Platform

[![CI Pipeline](https://github.com/yourusername/coba/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/coba/actions/workflows/ci.yml)
[![Deploy Status](https://github.com/yourusername/coba/actions/workflows/deploy.yml/badge.svg)](https://github.com/yourusername/coba/actions/workflows/deploy.yml)
[![Frontend Tests](https://img.shields.io/badge/frontend%20tests-103%2F103%20passing-brightgreen)](web/frontend)
[![Backend Tests](https://img.shields.io/badge/backend%20tests-63%2F63%20passing%20%2890%25%20coverage%29-brightgreen)](web/backend)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict%20mode-blue)](web/frontend)

An interactive educational platform for teaching **17 contextual bandit algorithms** through hands-on, browser-based simulations.

[🎮 Try it live](https://coba-web.vercel.app) | [📚 Reference](https://coba-web.vercel.app/reference) | [📖 Docs](./docs/index.md)

## Features

### 🎓 17 Interactive Lessons

**Beginner:**
- Explore vs Exploit (Epsilon-Greedy)
- UCB1 (Upper Confidence Bound)
- Thompson Sampling

**Intermediate:**
- LinUCB (Contextual Linear)
- Linear Thompson Sampling
- Logistic Bandits
- Cluster Routing
- LinUCB-Hybrid

**Advanced:**
- Neural Linear
- Random Forest
- Gaussian Process UCB
- Softmax Exploration
- Sliding-Window LinUCB
- Drift Detection
- Offline Evaluation

**Specialist:**
- CATS (Real-Time Bidding)
- Production Features

### 🎯 Key Capabilities

- **Interactive Simulations** — Adjust parameters and watch algorithms in real-time
- **Real-time Visualizations** — Reward curves, regret analysis, arm scores, distributions
- **Theory Cards** — Collapsible algorithm explanations with formulas
- **Progress Tracking** — Mark lessons complete, localStorage persistence
- **Keyboard Navigation** — Space (play/pause), Arrows (step), Numbers (speed), Ctrl+N/P (next/prev)
- **Algorithm Reference** — Comprehensive guide with papers and complexity analysis
- **Responsive Design** — Works on desktop, tablet, mobile
- **Dark Mode** — Full theme support
- **Production Ready** — 166 tests, zero TypeScript errors, 90% backend coverage

## Quick Start

### Frontend

```bash
cd web/frontend
npm install
npm run dev
```

Visit `http://localhost:3000` and click any lesson to start learning!

### Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

## Project Structure

```
coba/
├── web/
│   ├── frontend/           # Next.js 16 + React 19 + TypeScript
│   │   ├── components/     # 50+ reusable components
│   │   ├── app/            # Pages (landing, lessons, reference)
│   │   ├── lib/            # Hooks, API client, lesson registry
│   │   └── tests/          # 103 unit tests
│   │
│   └── backend/            # FastAPI + Python
│       ├── app/
│       │   ├── models/     # Pydantic schemas
│       │   ├── routers/    # REST endpoints
│       │   └── services/   # Business logic
│       └── tests/          # 63 tests (90% coverage)
│
├── .github/workflows/      # CI/CD pipelines
├── docs/                   # Documentation
└── README.md              # This file
```

## Architecture

### Frontend Stack
- **Framework:** Next.js 16 (Turbopack)
- **Language:** TypeScript (strict mode)
- **Styling:** TailwindCSS + dark mode
- **UI:** Custom components + Recharts
- **Testing:** Vitest + React Testing Library
- **State:** React hooks + localStorage

### Backend Stack
- **Framework:** FastAPI
- **Language:** Python 3.10+
- **Validation:** Pydantic
- **Testing:** Pytest (90% coverage)
- **Features:** 6 REST endpoints, session management, trace building

### Design Principles
- ✅ **DRY** — Lesson configs in single registry
- ✅ **SOLID** — Clear interfaces, minimal coupling
- ✅ **Clean Code** — Self-documenting, well-tested
- ✅ **Performance** — Fast builds, efficient rendering
- ✅ **Accessibility** — Semantic HTML, ARIA labels

## Testing

### Frontend

```bash
cd web/frontend
npm test                 # Run all tests
npm test -- --coverage  # With coverage
npm run build           # Type check + build
```

**Test Coverage:** 103 tests, all passing

### Backend

```bash
cd web/backend
pytest                  # Run all tests
pytest --cov=app       # With coverage
```

**Test Coverage:** 63 tests, 90% coverage

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `→` | Step forward |
| `←` | Previous step |
| `1` / `2` / `3` | Speed 1x / 10x / 100x |
| `Ctrl+N` | Next lesson |
| `Ctrl+P` | Previous lesson |
| `?` | Show help |

## API Endpoints

### Sessions
- `POST /sessions` — Create session
- `POST /sessions/{id}/step` — Step the bandit
- `POST /sessions/{id}/update` — Update with reward
- `GET /sessions/{id}` — Get stats
- `DELETE /sessions/{id}` — Delete session

### Lesson Extras
- `POST /sessions/{id}/arm` — Add/remove arm
- `POST /sessions/{id}/drift` — Inject drift
- `POST /sessions/{id}/offline-eval` — Offline evaluation
- `GET /sessions/{id}/cluster-map` — Cluster visualization
- `GET /sessions/{id}/leaf-scores` — CATS scores

[Full API docs](http://localhost:8000/docs)

## Deployment

### Frontend (Vercel)

```bash
# Automatic via GitHub Actions on push to main
# Or manual:
npm run build
vercel deploy --prod
```

### Backend

Recommended platforms:
- **Railway** — Easy PostgreSQL integration
- **Render** — Free tier available
- **Fly.io** — Global deployment
- **Heroku** — Classic option

See [deployment guide](./docs/deployment.md) for details.

## CI/CD

GitHub Actions workflows:
- **CI Pipeline** — Tests + builds on every push/PR
- **Deploy** — Auto-deploys frontend to Vercel on merge to main

Status badges at top of README.

## Documentation

Full documentation available in [`docs/`](./docs/) including:

- [Architecture Guide](./docs/ARCHITECTURE.md) — System design & component structure
- [Deployment Guide](./docs/DEPLOYMENT.md) — Production deployment instructions
- [Quick Start for Learners](./docs/QUICK_START_LEARNER.md) — Using the platform
- [Adding New Lessons](./docs/ADDING_LESSONS.md) — Developer guide for new algorithms
- [Algorithm Reference](./docs/algorithms/) — Deep dives (LinUCB, Neural Linear, GP-UCB, etc.)
- [API Documentation](http://localhost:8000/docs) — Interactive API reference

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make changes with tests
4. Push and open a PR
5. GitHub Actions will run tests automatically

## Metrics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | ~7,500 |
| **Frontend Tests** | 103 / 103 ✅ |
| **Backend Tests** | 63 / 63 (90% coverage) ✅ |
| **TypeScript Errors** | 0 ✅ |
| **Build Time** | ~1.2s ✅ |
| **Bundle Size** | ~45KB (gzipped) |
| **Components** | 50+ |
| **Lessons** | 17 |
| **Papers Referenced** | 14+ |

## License

MIT

## Citation

If you use COBA Web in your research or teaching, please cite:

```bibtex
@software{coba_web_2026,
  author = {Your Name},
  title = {COBA Web: Interactive Contextual Bandits Educational Platform},
  year = {2026},
  url = {https://github.com/yourusername/coba}
}
```

## Questions?

See [docs/](./docs/) for comprehensive documentation, or open an issue!

---

**Status:** 🎉 Production-ready! All 17 lessons fully interactive.
