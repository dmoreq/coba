# Git History Rewrite Plan

## Goals

1. **Remove `Co-Authored-By: Claude`** from all 19 commits that contain it
2. **Squash noise commits** — process logs, summaries, plans, phase reports, and quick-ref docs that were committed as files but should not exist as individual commits
3. **Normalize commit messages** — consistent conventional-commit style, no "Step A/B/C", no "Phase N complete" titles, no em-dashes, no redundant prefixes
4. **Result** — a clean linear history of ~30 meaningful commits that a new contributor can read to understand how the project was built

---

## Current State

| Metric | Value |
|--------|-------|
| Total commits | 224 |
| Commits with `Co-Authored-By: Claude` | 19 |
| Pure docs-noise commits (plans, reports, summaries) | ~30 |
| Target commit count (estimate) | ~30 |

---

## Commits to Remove (squash into neighbour or drop entirely)

These commits added files that have since been deleted or are internal AI process artefacts.
They should be squashed into the nearest meaningful commit or dropped outright.

### Internal AI process docs (drop / squash)

| Hash | Subject | Action |
|------|---------|--------|
| `2012428` | docs: Phases 6-8 complete — missing module docs… | squash → `docs: update algorithm reference and Vietnamese translations` |
| `57a13f2` | docs: add comprehensive final report — all 5 phases | drop |
| `8b16339` | docs: Phase 4-5 completion — polish, variant notes | squash → above |
| `002b692` | docs: Phase 1-3 documentation overhaul | squash → above |
| `4582c67` | chore: remove historic documentation | drop (files gone) |
| `45db044` | docs: update core documentation for v1.0 status | squash → above |
| `3bc4ed3` | docs: Theme diagnostic and manual testing guide | drop (file removed) |
| `1219be1` | docs: Comprehensive guide to complete theme fix | drop (file removed) |
| `a5b7934` | docs: Quick reference guide for Steps A, B, C | drop (file removed) |
| `cad9013` | docs: Complete summary of Steps A, B, C fixes | drop (file removed) |
| `cbd829b` | docs: Final summary — UI/UX Redesign complete | drop (file removed) |
| `6dc5b26` | docs(ui): Phase 3 complete — Final redesign docs | drop |
| `9edc905` | docs: comprehensive UI/UX redesign summary | drop |
| `d6df2d7` | docs(ui): Phase 1-2 complete status and pattern guide | drop |
| `5e00431` | docs: Add lesson refactoring pattern guide Phase 2b | drop (file removed) |
| `4665a02` | docs: add bandit decision test execution summary | drop |
| `e499eb2` | docs: add bandit decision test plan | drop |
| `9e13d73` | docs: add web cleanup execution summary | drop |
| `d5d4c69` | docs: add web cleanup plan | drop |
| `133aa14` | docs: test execution results summary - 17/17 | drop |
| `42a3889` | docs: comprehensive integration test execution report | drop |
| `ae9ec0f` | docs: integration tests quick start and reference | squash → `test: add integration test suite` |
| `8cb6965` | docs: add deployment ready checklist | drop (file removed) |
| `693adf8` | docs: add final completion summary — COBA Web 100% | drop |
| `712ac42` | docs: add comprehensive QA checklist | drop |
| `100a246` | docs: add final status report — 90% complete | drop |
| `542e476` | docs: add comprehensive progress checkpoint 50% | drop |
| `0069a36` | docs: add session summary for comprehensive review | drop |
| `0fa0816` | docs: add comprehensive implementation guides | drop |
| `bd33f2d` | docs: add web education platform design and plan | drop |
| `12cd5c5` | docs: remove non-core markdown files | drop (redundant) |
| `fa537f1` | docs: deep analysis and 12-commit cleanup plan | drop |
| `3ed36d1` | docs: add final implementation summary | drop |
| `1d7c061` | docs: add comprehensive architecture decision record | drop |
| `e10a3de` | docs: add comprehensive accessibility/perf testing | drop |
| `d5a79c0` | docs: add deployment, lesson extension… guides | squash → `docs: add deployment and contribution guides` |
| `4d608aa` | docs: add comprehensive plan for phases 1-6 | drop |
| `8a109c0` | docs: consolidate documentation into 8 focused docs | drop |
| `9b9da3b` | docs: add startup checklist and verification guide | drop |
| `fd8453b` | docs: add comprehensive startup guide | drop |
| `3ec5bab` | docs: add comprehensive codebase cleanup analysis | drop |
| `68bd7f7` | docs: add session summary and final status report | drop |
| `e59c74e` | docs: add comprehensive plan for Tree Ensemble | drop |
| `c10bac8` | docs: add final comprehensive implementation report | drop |
| `fd8e937` | docs: add comprehensive implementation summary | drop |
| `f1e850f` | docs: add comprehensive CATS implementation progress | drop |
| `b46be2f` | docs: record coba cleanup plan | drop |
| `09fe2ad` | docs: record chart explanation validation checklist | drop |
| `ad78a1e` | docs: plan chart explanations and slow-run updates | drop |
| `4319ac1` | docs(web): replace scaffold README with project runbooks | squash → `docs: add web backend and frontend runbooks` |

---

## Co-Authored-By Commits (strip trailer only, keep commit)

| Hash | Subject |
|------|---------|
| `3cac721` | Initial commit — COBA contextual bandit engine |
| `eb70bb8` | refactor: adopt src/ layout and flatten single-file subpackages |
| `cae2f3b` | chore: add pre-commit with ruff, black, and standard file hooks |
| `25df8d6` | refactor: remove domain-specific language from source code |
| `00b5548` | docs: improve clarity of English documentation |
| `9e0203a` | docs(vi): improve clarity of Vietnamese documentation |
| `26330795` | style: apply ruff-format to all remaining Python files |
| `774f707` | refactor: rename fit_from_logs to fit_offline across codebase |
| `9d7404f` | docs: add runnable examples for all library features |
| `2bc90dc` | fix(core): guard SM denominator drift, fix explore scores |
| `d195e59` | feat(bandit): add decide_top_k for ranked arm selection |
| `ccc91aa` | feat(normalizer): add RewardNormalizer for running reward scaling |
| `f4c50fc` | feat(bandit): add confidence-based abstention to decide() |
| `5ca92e9` | feat(router): add per-arm gamma override in add_arm() |
| `d9a921a` | feat(drift): add PageHinkleyDetector for reward distribution shift |
| `3907ef5` | feat(bandit): add min_pull_rates constraint for guaranteed arm exploration |
| `25402d9` | feat(policies): add LinUCB-Hybrid policy |
| `7c96b46` | feat(policies): add NeuralLinear bandit |
| `fa71038` | feat(policies): add GP-UCB policy with RBF kernel |

---

## Commit Message Normalization Rules

| Pattern | Replace with |
|---------|-------------|
| `Step A —`, `Step B —`, `Step C —` | Describe what changed: `fix(ui): activate theme toggle`, etc. |
| `Phase N complete —` | Remove phase language, describe the outcome |
| `Phases N-M complete —` | Same |
| `— ` (em-dash) in subject | `:` or plain description |
| `comprehensive` in subject | Remove; be specific |
| `add comprehensive …` | `add …` |
| `test(phases-2-3)` | `test(backend)` |
| `fix(phase-1)` | `fix(backend)` |
| `fix(coba)` | `fix(router)` |

---

## Target History (~30 commits, chronological oldest→newest)

```
1.  feat: initial COBA contextual bandit engine
2.  refactor: adopt src/ layout and flatten subpackages
3.  chore: add pre-commit hooks (ruff, black)
4.  style: apply ruff-format to all Python files
5.  refactor: remove domain-specific language from source
6.  docs: clarify English and Vietnamese documentation
7.  feat(bandit): add decide_top_k for ranked arm selection
8.  feat(bandit): add confidence-based abstention to decide()
9.  feat(normalizer): add RewardNormalizer for running reward scaling
10. feat(router): add per-arm gamma override in add_arm()
11. feat(drift): add PageHinkleyDetector for reward shift detection
12. feat(bandit): add min_pull_rates for guaranteed arm exploration
13. fix(core): guard Sherman-Morrison drift, fix explore scores
14. feat(policies): add GP-UCB with RBF kernel and Cholesky inference
15. feat(policies): add NeuralLinear bandit (MLP backbone + LinTS)
16. feat(policies): add LinUCB-Hybrid with shared cross-arm learning
17. feat(policies): add tree ensemble bandits (Random Forest UCB/TS)
18. feat(continuous): add CATS policy and ContinuousBandit facade
19. docs: add runnable examples for all library features
20. build: add optional streamlit extra with plotly
21. feat(web): scaffold interactive learning platform
22. feat(web): add core interactive lesson modules
23. feat(web): add advanced educational modules
24. feat(web): add production utility endpoints and theme support
25. feat(web): expose tree ensemble policies in frontend
26. feat(web): add continuous bid optimizer API and simulator
27. feat(frontend): implement lesson registry and all lesson components
28. feat(frontend): implement typed API client, hooks, and session management
29. feat(ci): add CI/CD pipeline
30. refactor(frontend): centralize simulation, hooks, and lesson utilities
31. fix(backend): align policy metadata, session routing, and trace builder
32. fix(frontend): fix hydration, camelCase conversion, and error handling
33. refactor(ui): full LessonShell layout redesign across all 16 lessons
34. fix(theme): complete Tailwind v4 class-based dark mode separation
35. test(backend): add context-free and contextual policy decision tests
36. test(frontend): add integration test suite with fixtures
37. docs: update algorithm reference, deployment guide, and Vietnamese translations
38. chore: remove non-standard markdown files
```

---

## Execution Steps

### Prerequisites
```bash
# Ensure clean working tree
git status   # should be clean
git stash list   # should be empty

# Create a backup branch before any rebase
git branch backup/pre-rewrite main
```

### Step 1 — Interactive rebase from root
```bash
git rebase -i --root
```

In the editor:
- Mark `pick` → `drop` for all noise doc commits listed above
- Mark `pick` → `squash` / `fixup` for commits that belong together
- Rewrite subjects following the normalization rules above

### Step 2 — Strip Co-Authored-By trailers
For the 19 commits with `Co-Authored-By: Claude`, during the rebase:
- Use `reword` action to edit the commit message
- Delete the `Co-Authored-By:` line from the trailer

Or after the rebase, run a filter-branch / filter-repo pass:
```bash
git filter-repo --message-callback '
import re
return re.sub(rb"\nCo-Authored-By: Claude.*\n?", b"\n", message).rstrip() + b"\n"
'
```

### Step 3 — Force push (coordinate if others have cloned)
```bash
git push origin main --force-with-lease
```

---

## Risk & Rollback

- `backup/pre-rewrite` branch preserves the original 224-commit history
- All file content is preserved — only commit metadata and groupings change
- `git diff backup/pre-rewrite main` should show only the deleted noise-doc files
- Restore with: `git checkout backup/pre-rewrite -b main-restored`
