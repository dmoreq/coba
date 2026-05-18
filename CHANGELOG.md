# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Changed
- Removed internal process and AI-agent markdown files from repository

---

## [1.0.0] — 2026-05-18

### Web Platform

#### Added
- Interactive learning platform (`web/`) with Next.js 15 frontend and FastAPI backend
- 16 lesson components covering all bandit algorithms with live simulation
- `LessonShell` two-column layout — theory + controls left, charts right
- `useSimulator` / `useSession` / `useProgress` / `useRewardRegret` hooks
- Typed API client with full camelCase↔snake_case conversion
- Component registry replacing per-route ternary chains
- Reusable lesson UI section components (`ArmBar`, `PullHistogram`, `RewardChart`, `RegretChart`, `BetaDistribution`, `TreeDiagram`, `ConfidenceEllipse`, `DriftTimeline`)
- Reference page with algorithm comparison table and keyboard shortcuts
- CI/CD pipeline (GitHub Actions) for backend and frontend
- Integration test suite with session lifecycle fixtures

#### Fixed
- Full Tailwind v4 class-based dark mode (`.dark` class, not OS media query)
- Mean=0 bug in trace builder for context-free policies
- `chosen_action` vs `chosen_arm` field mismatch for `ContinuousBandit`
- camelCase→snake_case conversion for nested config objects
- Hydration warning from browser extension attribute injection
- Lesson route serialization error (`rewardFn` not JSON-serializable)
- `nFeatures` not forwarded from `useSession` to `useSimulator`
- Button enabled state not gated on session readiness
- UCB1Lesson reward function and unused rates maps

#### Refactored
- All 16 lesson components to unified `LessonShell` layout (57% vertical scroll reduction)
- Lesson simulation utilities centralized from per-lesson duplicates
- Shared SSE streaming helper extracted in backend
- Policy metadata centralized; lesson extras router helpers extracted

---

## [0.9.0] — 2026-04 (Continuous Actions & Tree Ensembles)

### Added
- `ContinuousBandit` public façade with `CATSPolicy` (Continuous Action Tree Search)
- `BinaryActionTree` with comprehensive unit tests
- `CATSLeafModel` extending `LinUCBArmModel`
- Continuous action schemas (`ContinuousDecision`) with validation
- Data generators for continuous action scenarios
- **Tree Ensemble Bandits** (arXiv 2402.06963):
  - Random Forest UCB and TS arm models
  - Tree ensemble uncertainty estimator
  - Integration with `ClusterBandit`
- Continuous bid optimizer API routes and async simulator
- Frontend bid optimizer page and navigation entry

---

## [0.8.0] — 2026-03 (Policy Expansion)

### Added
- `GP-UCB` policy — Gaussian Process with RBF kernel and Cholesky inference (Srinivas et al., ICML 2010)
- `NeuralLinear` bandit — MLP backbone + per-arm LinTS on learned embeddings
- `LinUCB-Hybrid` policy — shared cross-arm feature learning via `SharedRidge`
- `RewardNormalizer` — minmax and zscore modes with exponential moving statistics
- `decide_top_k()` — ranked top-k arm selection for recommendation lists
- Confidence-based abstention: `decide()` returns `abstained=True` on near-tie scores
- Per-arm `gamma` override in `add_arm()` for faster cold-start adaptation
- `min_pull_rates` constraint on `ClusterBandit` for guaranteed arm exploration floors
- `PageHinkleyDetector` — two-sided sequential test for reward distribution shift

### Fixed
- Sherman-Morrison denominator clamped to `1e-10` to prevent NaN/inf in long-running streams
- Cold-start explore scores changed from magic constants (`1e3`) to `float("inf")`
- `IPSEstimator.compute_weights()` unused `rewards` parameter removed
- `router.update()` now correctly exits cold-start after `n_clusters` online updates

---

## [0.7.0] — 2026-02 (Domain Refactor & Examples)

### Changed
- All domain-specific language (rideshare, pricing) removed from source code, docstrings, and metadata
- `fit_from_logs` renamed to `fit_offline` across codebase and all documentation
- Vietnamese and English documentation clarified and import paths corrected

### Added
- Runnable examples for all library features (quickstart, offline bootstrap, arm management, monitoring)
- Streamlit interactive apps replacing 16 static examples
- Optional `streamlit` extra with `plotly` dependency

---

## [0.6.0] — 2026-01 (src Layout & Tooling)

### Changed
- Package moved to `src/coba/` layout (prevents accidental working-directory imports)
- `routers/`, `offpolicy/`, `evaluation/` single-file subpackages flattened to `router.py`, `offpolicy.py`, `evaluation.py`

### Added
- Pre-commit hooks: `ruff` (with `--fix`), `ruff-format`, `black`, `check-yaml`, `check-toml`, `trailing-whitespace`

---

## [0.1.0] — 2025-12 (Initial Release)

### Added
- `ClusterBandit` — KMeans cluster routing with per-cluster arm models
- `ClusterRouter` — online KMeans with Sherman-Morrison updates
- `LinUCBArmModel`, `LinTSArmModel`, `LogisticBanditsArmModel`, `SoftmaxArmModel`
- `SlidingWindowLinUCBArmModel` for non-stationary reward streams
- IPS and Doubly Robust off-policy bootstrapping (`fit_offline`)
- Offline evaluation metrics (`BanditEvaluator`)
- `BanditDecision` / `BanditConfig` schemas
- English and Vietnamese algorithm documentation
