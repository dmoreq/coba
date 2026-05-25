# COBA Web Documentation

Welcome to COBA Web — an **interactive educational platform** for learning **17 contextual bandit algorithms** through hands-on simulations.

👉 **New here?** Start with [Quick Start for Learners](./QUICK_START_LEARNER.md)

---

## Documentation Structure

### Getting Started
- **[Quick Start for Learners](./QUICK_START_LEARNER.md)** — How to use the platform
- **[Architecture Guide](./ARCHITECTURE.md)** — System design & technical decisions
- **[Roadmap](./ROADMAP.md)** — Future features & enhancement plans

### For Educators
- **[Adding New Lessons](./ADDING_LESSONS.md)** — Implement a new algorithm
- **[Contributing Guide](../CONTRIBUTING.md)** — Code standards, testing, accessibility

### Reference
- **[Algorithm Library](./algorithms/)** — Deep dives on all algorithms
- **[Evaluation Methods](./evaluation.md)** — Offline policy evaluation (IPS, DR, NCIS)
- **[Policy Reference](./policies.md)** — Algorithm comparisons & complexity analysis
- **[Advanced Features](./advanced_features.md)** — Drift detection, arm management, CATS

### Deployment
- **[Deployment Guide](./DEPLOYMENT.md)** — Local dev and production setup

### Contributing
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — Code standards, PR process

---

## Quick Navigation by Role

### Learner / Student
1. Run the app: `uv run python -c "from web.main import run; run()"`
2. Open http://localhost:8550
3. Start with a lesson or free-play in the Arena
4. See [Quick Start for Learners](./QUICK_START_LEARNER.md) for details

### Instructor
1. Read [Quick Start for Learners](./QUICK_START_LEARNER.md) to understand the platform
2. See [Adding New Lessons](./ADDING_LESSONS.md) to create custom algorithms
3. Review [Roadmap](./ROADMAP.md) for planned classroom features

### 👨‍💻 Developer
1. Fork the repo: `git clone https://github.com/yourusername/coba.git`
2. Read [Architecture Guide](./ARCHITECTURE.md) for system design
3. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for dev setup
4. See [Adding New Lessons](./ADDING_LESSONS.md) to add an algorithm
5. Read [Algorithm Library](./algorithms/) for deep technical context

### 🚀 DevOps / Deployment
1. See [Deployment Guide](./DEPLOYMENT.md) for frontend (Vercel) & backend (Railway/Render)
2. Review [Architecture Guide](./ARCHITECTURE.md) for infrastructure decisions
3. Check [CONTRIBUTING.md](../CONTRIBUTING.md) for CI/CD workflows

---

## 🔍 COBA Python Library Reference

This documentation focuses on **COBA Web** (the educational platform). For the underlying **COBA Python library**, see:

- **Core Concepts:**
  - `coba.bandit.ClusterBandit` — Main entry point
  - `coba.policies.*` — Learning algorithms (LinUCB, Thompson Sampling, etc.)
  - `coba.router.ClusterRouter` — KMeans context clustering
  - `coba.evaluation.*` — Offline policy evaluation

- **Key Principle:** Domain-agnostic (operates on numpy arrays, zero coupling to your business logic)

- **Example:**
```python
from coba.bandit import ClusterBandit

bandit = ClusterBandit(policy='lin_ucb')
arm = bandit.decide(context=features)       # Recommend arm
bandit.update(context=features, arm=arm, reward=0.95)  # Observe reward
```

For full COBA library docs, see the [main README](../README.md).

---

## Project Stats

| Metric | Value |
|--------|-------|
| Tests | 775 ✅ |
| Algorithms | 17 |
| Lessons | 14 |
| Worlds | 7 |
| LOC | ~8,000 |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Reporting bugs
- Proposing features
- Opening PRs
- Code review process

---

## 📝 License

MIT — See [LICENSE](../LICENSE)

---

**Status:** 🎉 Production-ready (v1.0)
**Questions?** Open an issue or email us!
