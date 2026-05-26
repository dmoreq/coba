# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Added LICENSE file (MIT) and pyproject.toml metadata (GitHub URLs, classifiers, README, author) for pip-install-from-GitHub compatibility.
- Added `dist/` to `.gitignore`.

### Changed
- CI workflow now uses `uv` instead of `pip install -e ".[dev]"`.
- Husky pre-commit hook runs `pre-commit` instead of stale `lint-staged`.
- README now includes `pip install git+...` and `uv add git+...` instructions.

### Removed
- Removed the legacy UI package and its associated tests, workflows, and documentation.
- Removed stale `lint-staged` configuration from `package.json` (referenced deleted `dashboard/`).
- Removed Black pre-commit hook (Ruff handles formatting).
- Removed stale `dashboard/frontend` references from `package.json`.

---

## [0.1.0]

### Added
- Core contextual bandit library.
- Discrete and continuous-action policies.
- Offline policy evaluation utilities.
- Drift detection, reward normalization, constraints, and persistence helpers.
