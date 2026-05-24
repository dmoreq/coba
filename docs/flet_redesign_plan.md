# COBA Flet Redesign: Architecture & Implementation Plan

**Status:** Planning Phase | **Created:** 2026-05-24
**Target Audience:** Absolute beginners | **Tech:** Flet (UI) + coba (ML) + uv (pkg)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Narrative Worlds — Seven Real-World Scenarios](#2-narrative-worlds--seven-real-world-scenarios)
3. [Full Algorithm & Feature Curriculum](#3-full-algorithm--feature-curriculum)
4. [Theory Pedagogy — How Each Algorithm Is Taught](#4-theory-pedagogy--how-each-algorithm-is-taught)
5. [Algorithm Debugger — Step-by-Step Formula Visualization](#5-algorithm-debugger--step-by-step-formula-visualization)
6. [Configuration Control System — Tooltips & Hover Help](#6-configuration-control-system--tooltips--hover-help)
7. [UI/UX Design System](#7-uiux-design-system)
8. [Component Architecture](#8-component-architecture)
9. [State Management](#9-state-management)
10. [Data Visualization](#10-data-visualization)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Testing Strategy](#12-testing-strategy)
13. [Dependencies & Configuration](#13-dependencies--configuration)

---

## 1. Project Structure

```
coba/
├── src/
│   ├── coba/                         # ML Engine (EXISTING — read-only unless an adapter needs a public hook)
│   │   ├── bandit.py                 #   ClusterBandit for discrete arms
│   │   ├── config.py                 #   BanditConfig — 30+ hyperparams
│   │   ├── router.py                 #   ClusterRouter (KMeans context routing)
│   │   ├── schemas.py                #   BanditDecision, BanditStats (Pydantic)
│   │   ├── types.py                  #   Arm, PolicyType (17 variants)
│   │   ├── drift.py                  #   PageHinkleyDetector
│   │   ├── evaluation.py             #   Offline eval: Rejection, DR, NCIS
│   │   ├── offpolicy.py              #   IPS/DR off-policy estimators
│   │   ├── normalizer.py             #   Reward normalization utilities
│   │   ├── persistence.py            #   Model save/load
│   │   ├── simulation/               #   Pure reward functions
│   │   │   └── reward_fns.py         #     categorical, linear, context_free
│   │   ├── continuous/               #   Continuous-action bandits (CATS path)
│   │   │   ├── action_tree.py
│   │   │   ├── bandit.py             #     ContinuousBandit
│   │   │   ├── policy.py             #     CATSPolicy
│   │   │   └── schemas.py
│   │   └── policies/                 #   Discrete arm model implementations
│   │       ├── base.py, ucb1.py, thompson.py, linucb.py, lin_ts.py
│   │       ├── linucb_sw.py, lin_ucb_hybrid.py, gp_ucb.py, logistic.py
│   │       ├── softmax.py, sklearn_models.py, cats.py  # cats.py = CATS leaf model
│   │       ├── neural_linear.py, tree_ensemble.py, tree_ensemble_base.py
│   │       └── ridge.py
│   │
│   └── coba_web/                     # Flet frontend package (NEW; avoids generic top-level `web` import)
│       ├── __init__.py
│       ├── main.py                   # ft.app(target=main)
│       ├── app_config.py             # Theme, colors, spacing, typography
│       │
│       ├── narrative/                # Seven narrative worlds
│       │   ├── __init__.py
│       │   ├── _base.py              #  WorldConfig, FeatureDef, ArmDef
│       │   ├── worlds.py             #  All 7 world definitions
│       │   ├── scene.py              #  Scene renderer (dynamic env panel)
│       │   └── characters.py         #  Animated avatars, feature badges
│       │
│       ├── engine/                   # Simulation bridge (coba ↔ Flet)
│       │   ├── __init__.py
│       │   ├── discrete_session.py   #  Wraps ClusterBandit for discrete-arm lessons
│       │   ├── continuous_session.py #  Wraps ContinuousBandit/CATSPolicy for CATS
│       │   ├── step.py               #  StepRunner, context generation, debug capture
│       │   └── features.py           #  Feature engine (drift, offline eval)
│       │
│       ├── state/                    # Reactive state management
│       │   ├── __init__.py
│       │   ├── store.py              #  AppStore (dict-based observable)
│       │   └── actions.py            #  step, run_n, reset, update_config
│       │
│       ├── components/               # Reusable Flet UI components
│       │   ├── __init__.py
│       │   ├── layout.py             #  AppShell, TopBar, BottomControls
│       │   ├── arena.py              #  Arena split-pane (env | agent)
│       │   ├── scene_panel.py        #  Narrative scene (replaces static patient card)
│       │   ├── treatment.py          #  TreatmentPanel, TreatmentCard (animated)
│       │   ├── controls.py           #  ControlDrawer (collapsible, right panel)
│       │   ├── tooltips.py           #  Tooltip system for every config control
│       │   ├── charts.py             #  RewardChart, RegretChart, PullBar, MeanBar
│       │   ├── beliefs.py            #  ConfidenceBounds (UCB), BetaPosteriors (TS)
│       │   ├── trace.py              #  TracePanel (DataTable, collapsible)
│       │   ├── debugger.py           #  Decision-faithful algorithm debugger
│       │   ├── theory_card.py        #  Structured theory card (§4)
│       │   ├── lesson_card.py        #  LessonCard (curriculum map tile)
│       │   ├── world_selector.py     #  Narrative world picker
│       │   └── theme_switch.py       #  Dark/light mode toggle
│       │
│       ├── lessons/                  # One page per lesson (17 lessons + comparison/sandbox)
│       │   ├── __init__.py
│       │   ├── _base.py              #  BaseLesson scaffold
│       │   ├── epsilon_greedy.py
│       │   ├── ucb1.py
│       │   ├── thompson.py
│       │   ├── linucb.py
│       │   ├── lints.py
│       │   ├── softmax.py
│       │   ├── logistic.py           #  LogisticUCB + LogisticTS
│       │   ├── gp_ucb.py
│       │   ├── linucb_sw.py          #  Sliding-window LinUCB
│       │   ├── linucb_hybrid.py
│       │   ├── bootstrapped.py       #  BootstrappedTS + BootstrappedUCB
│       │   ├── neural_linear.py
│       │   ├── tree_ensemble.py      #  RandomForestUCB + RandomForestTS
│       │   ├── cats.py               #  Continuous Action Tree Sampling (continuous session)
│       │   ├── drift.py              #  Drift detection lesson
│       │   ├── offline_eval.py       #  Offline evaluation lesson
│       │   └── comparison.py         #  Side-by-side comparison mode
│       │
│       ├── views/                    # Route-level pages
│       │   ├── __init__.py
│       │   ├── home.py               #  Curriculum map with world selector
│       │   ├── sandbox.py            #  Free-form experiment arena
│       │   └── about.py              #  Credits / methodology
│       │
│       └── utils/
│           ├── __init__.py
│           ├── animation.py          #  Animation helpers (pulse, slide, fade)
│           └── colors.py             #  Theme-aware color helpers
│
├── tests/
│   ├── coba/                         # Existing ML tests
│   └── coba_web/                     # NEW: Flet UI tests
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_discrete_session.py
│       ├── test_continuous_session.py
│       ├── test_store.py
│       ├── test_step.py
│       ├── test_narrative.py
│       ├── test_components.py
│       └── test_lessons/
│           ├── test_ucb1.py
│           ├── test_thompson.py
│           ├── test_linucb.py
│           └── ...
```

### 1.1 Architectural Constraints

- **Discrete lessons** (`UCB1`, `Thompson`, `LinUCB`, etc.) use `ClusterBandit` + `ClusterRouter`.
- **CATS** uses the continuous-action stack (`ContinuousBandit` / `CATSPolicy`) and must not be routed through `ClusterBandit` or `ClusterRouter`.
- **Debugger snapshots are captured during decision execution**, not recomputed by the UI after the fact. This preserves stochastic samples for Thompson/TS and the routed cluster for contextual policies.
- If debug support requires engine changes, add narrow public hooks such as `ClusterRouter.predict_with_debug()` rather than accessing private attributes from the UI package.
- The Flet package is `coba_web` to avoid colliding with the existing top-level `web` workspace configuration.

---

## 2. Narrative Worlds — Seven Real-World Scenarios

Each lesson can be experienced through **any** narrative world. The user selects a world on the home page. The world changes the visual skin, not the algorithm — the same UCB1 runs identically under the hood.

### World 1: The Rural Clinic (Healthcare)
> **Real application:** Adaptive clinical trials (FDA-endorsed), personalized treatment assignment

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Treatments: Pill A, Injection B, Therapy C | Color-coded treatment cards |
| Context | Patient vitals: Age, Blood Pressure, Temperature | Vital-sign gauge bars |
| Reward | Recovery: Healthy / Not recovered | Animated health bar |
| Ground truth | Each treatment works better for different patient profiles | Hidden weight vectors per treatment |

### World 2: MovieMatch — Streaming Recommendations
> **Real application:** Netflix homepage, YouTube Up Next, Spotify Discover Weekly

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Movie genres: Drama, Comedy, Sci-Fi | Genre poster cards |
| Context | Viewer profile: Watch time, Rating history, Time of day | Profile badge row |
| Reward | Engagement: Watched full / Skipped | Thumbs up/down animation |
| Ground truth | Some viewers prefer certain genres at certain times | Hidden preference weights |

### World 3: NewsFeed — Article Personalization
> **Real application:** Yahoo! News (original LinUCB paper), Google News, Twitter timeline

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Article topics: Politics, Sports, Science | Headline cards |
| Context | Reader: Location, Device, Browsing history | Reader profile panel |
| Reward | Click: Clicked / Ignored | Click animation (pulse on article) |
| Ground truth | Reader interests vary by time/day/device | Hidden click probability per topic |

### World 4: ShopSmart — E-Commerce Product Display
> **Real application:** Amazon product ranking, Alibaba homepage, booking.com

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Products: Sneakers, Watch, Headphones | Product card thumbnails |
| Context | Shopper: Previous purchases, Cart value, Session time | Shopping cart indicator |
| Reward | Purchase: Bought / Walked away | Cart-add animation |
| Ground truth | Different shoppers want different products | Hidden purchase probability |

### World 5: RidePilot — Dynamic Pricing
> **Real application:** Uber surge pricing, Lyft dynamic pricing, DoorDash delivery fees

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Price multipliers: 1.0x, 1.3x, 1.6x, 2.0x | Price tag cards |
| Context | Conditions: Rain, Rush hour, Event nearby | Weather + traffic indicators |
| Reward | Ride accepted: Accepted / Declined | Checkmark/X animation |
| Ground truth | Riders accept different prices under different conditions | Hidden acceptance curves |

### World 6: GameBot — Difficulty Adaptation
> **Real application:** Duolingo lesson difficulty, gaming adaptive AI, educational software

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Difficulty: Easy, Medium, Hard | Difficulty badge cards |
| Context | Player: Skill level, Streak, Session duration | Player stat bars |
| Reward | Engagement: Continued playing / Quit | Continue/quit animation |
| Ground truth | Players stay engaged at optimal challenge level | Hidden engagement function |

### World 7: LabTrial — Adaptive Experiment Design
> **Real application:** A/B testing, multi-arm trials, hyperparameter optimization

| Bandit Concept | Skin | Visual |
|---------------|------|--------|
| Arms | Experiment variants: A, B, C, D | Variant label cards |
| Context | Experiment conditions: Sample size, Budget, Time | Lab instrument panel |
| Reward | Metric improvement: Improved / No change | Metric bar animation |
| Ground truth | Variants have different true effect sizes | Hidden effect size bars |

### World Configuration Schema

```python
# src/coba_web/narrative/_base.py
@dataclass
class WorldConfig:
    world_id: str              # "clinic", "moviematch", "newsfeed", ...
    name: str                  # "The Rural Clinic"
    emoji: str                 # "" — no longer used in UI
    description: str           # One-sentence summary
    arms_label: str            # "Treatments", "Genres", "Articles", ...
    context_features: list[FeatureDef]  # Named features with ranges
    reward_positive: str       # "Recovered", "Watched", "Clicked", ...
    reward_negative: str       # "Not recovered", "Skipped", "Ignored", ...
    arms: list[ArmDef]         # Default arms with icons, names, colors

@dataclass
class FeatureDef:
    key: str                   # "age_z", "bp_z", "temp_z"
    label: str                 # "Age", "Blood Pressure", "Temperature"
    range_min: float           # -2.0
    range_max: float           # +2.0
    unit: str                  # "(z-score)", "mmHg", "°C"
    low_label: str             # "Young", "Low", "Cool"
    high_label: str            # "Elderly", "High", "Fever"

@dataclass
class ArmDef:
    id: str                    # "pill_a"
    name: str                  # "Pill A"
    emoji: str                 # "" — no longer used in UI
    color: str                 # "#6366F1"
```

---

## 3. Full Algorithm & Feature Curriculum

### 3.1 Complete Inventory — 17 Policy Variants

| # | PolicyType Enum | Algorithm | Type | Context | File |
|---|----------------|-----------|------|---------|------|
| 1 | `UCB1` | Upper Confidence Bound | Optimistic | No | `ucb1.py` |
| 2 | `THOMPSON` | Thompson Sampling (Beta) | Bayesian | No | `thompson.py` |
| 3 | `EPSILON_GREEDY` | ε-Greedy | Heuristic | No | `sklearn_models.py` |
| 4 | `LIN_UCB` | Linear UCB | Optimistic | Yes | `linucb.py` |
| 5 | `LIN_TS` | Linear Thompson | Bayesian | Yes | `lin_ts.py` |
| 6 | `SOFTMAX` | Boltzmann Exploration | Probabilistic | Yes | `softmax.py` |
| 7 | `LOGISTIC_UCB` | Logistic Regression UCB | Optimistic | Yes | `logistic.py` |
| 8 | `LOGISTIC_TS` | Logistic Thompson | Bayesian | Yes | `logistic.py` |
| 9 | `LIN_UCB_SW` | Sliding-Window LinUCB | Optimistic | Yes | `linucb_sw.py` |
| 10 | `LIN_UCB_HYBRID` | Hybrid LinUCB | Optimistic | Yes | `lin_ucb_hybrid.py` |
| 11 | `BOOTSTRAPPED_TS` | Bootstrap Ensemble TS | Bayesian | Yes | `sklearn_models.py` |
| 12 | `BOOTSTRAPPED_UCB` | Bootstrap Ensemble UCB | Optimistic | Yes | `sklearn_models.py` |
| 13 | `GP_UCB` | Gaussian Process UCB | Optimistic | Yes | `gp_ucb.py` |
| 14 | `NEURAL_LINEAR` | Neural Network + LinTS | Bayesian | Yes | `neural_linear.py` |
| 15 | `RANDOM_FOREST_UCB` | Random Forest UCB | Optimistic | Yes | `tree_ensemble.py` |
| 16 | `RANDOM_FOREST_TS` | Random Forest TS | Bayesian | Yes | `tree_ensemble.py` |
| 17 | `CATS` | Continuous Action Tree | Optimistic | Yes | `continuous/policy.py` + `policies/cats.py` leaf model |

### 3.2 Additional Features (beyond algorithms)

| Feature | What It Does | Config Params | File |
|---------|-------------|--------------|------|
| **Drift Detection** | Auto-detects when reward patterns change; resets models | `enable_drift_detection`, `drift_delta`, `drift_lambda` | `drift.py` |
| **Offline Evaluation** | Evaluate a policy using logged data (no live interaction) | `method`, `n_bootstrap` | `evaluation.py` |
| **Off-Policy Correction** | IPS/DR weighting to correct biased data | (via `weight` param in `update()`) | `offpolicy.py` |
| **Cluster Routing** | Groups similar contexts; learns per-group models | `n_clusters`, `use_minibatch`, `scale_contexts` | `router.py` |
| **Min Pull Constraints** | Guarantees minimum traffic to specific arms | `min_pull_rates` | `bandit.py` |
| **CATS Action Space** | Continuous actions (not discrete arms); separate `ContinuousBandit` path | `cats_a_min`, `cats_a_max`, `cats_depth` | `continuous/` |

### 3.3 Lesson Map — 17 Lessons + Sandbox

**MVP boundary:** Ship a first usable app with 3 worlds (Clinic, MovieMatch, ShopSmart), 3 foundation lessons (ε-Greedy, UCB1, Thompson), core charts, trace table, and basic tooltips. Treat contextual, advanced, CATS, debugger, comparison, and sandbox work as staged expansions after the MVP is stable.

```
Phase 1: FOUNDATIONS (Context-Free MAB)
 ┌─────────────────────────────────────────────────────┐
 │ Lesson 1: ε-Greedy — "The Curious Doctor"           │
 │   Available in ALL 7 worlds                          │
 │   Teaches: Explore vs Exploit tradeoff               │
 │   Interact: ε slider (0.0–0.5)                      │
 │   Visuals: Random choice animation, regret counter   │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 2: UCB1 — "The Optimistic Doctor"             │
 │   Available in ALL 7 worlds                          │
 │   Teaches: Confidence bounds, optimism principle     │
 │   Interact: α slider (0.1–5.0)                       │
 │   Visuals: Confidence interval bars, squish on fail  │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 3: Thompson Sampling — "The Bayesian Doctor"  │
 │   Available in ALL 7 worlds                          │
 │   Teaches: Bayesian updating, Beta posterior         │
 │   Interact: Prior strength slider (α=β from 0.5–10)  │
 │   Visuals: Beta distribution curves narrowing        │
 └─────────────────────────────────────────────────────┘

Phase 2: CONTEXTUAL BANDITS (With Observable Features)
 ┌─────────────────────────────────────────────────────┐
 │ Lesson 4: LinUCB — "The Pattern-Spotting Doctor"     │
 │   Teaches: Linear model, context features matter     │
 │   Interact: α, l2_lambda sliders, toggle context     │
 │   Visuals: Weight vector arrows, per-feature impact  │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 5: LinTS — "The Cautious Bayesian"            │
 │   Teaches: Bayesian linear model, posterior sampling  │
 │   Interact: v_sq slider, compare vs LinUCB           │
 │   Visuals: Gaussian posterior ellipses               │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 6: Softmax — "The Diplomat"                   │
 │   Teaches: Probabilistic selection, temperature       │
 │   Interact: τ slider (0.01–10.0)                     │
 │   Visuals: Selection probability pie chart            │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 7: Logistic Bandits — "The Conversion Expert" │
 │   Teaches: Binary rewards (0/1), sigmoid, Laplace     │
 │   Covers: LogisticUCB + LogisticTS (toggle)          │
 │   Interact: α (UCB), v_sq (TS)                       │
 │   Visuals: Sigmoid decision boundary animation        │
 └─────────────────────────────────────────────────────┘

Phase 3: ADVANCED PATTERNS
 ┌─────────────────────────────────────────────────────┐
 │ Lesson 8: Sliding-Window LinUCB — "The Adaptable One"│
 │   Teaches: Non-stationary environments, forgetting    │
 │   Interact: Window size (50–500), inject drift event  │
 │   Visuals: Window highlight, old data fading out     │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 9: GP-UCB — "The Smooth Operator"             │
 │   Teaches: Gaussian Processes, uncertainty surfaces  │
 │   Interact: β, length_scale, noise_var sliders       │
 │   Visuals: 2D heatmap of GP posterior mean + CI      │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 10: Bootstrapped Bandits — "The Committee"    │
 │   Teaches: Ensemble uncertainty, bootstrap principle  │
 │   Covers: BootstrappedTS + BootstrappedUCB           │
 │   Interact: n_bootstraps (2–50)                      │
 │   Visuals: Ensemble member predictions as dots       │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 11: LinUCB Hybrid — "The Team Player"         │
 │   Teaches: Shared vs per-arm features                │
 │   Interact: n_shared_features slider                 │
 │   Visuals: Split context vector (shared | per-arm)   │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 12: Neural Linear — "The Deep Learner"        │
 │   Teaches: Neural feature extraction + linear head   │
 │   Interact: hidden_sizes, retrain_freq               │
 │   Visuals: Embedding space projection (2D)           │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 13: Tree Ensembles — "The Forest Ranger"      │
 │   Teaches: Decision trees for non-linear patterns    │
 │   Covers: RandomForestUCB + RandomForestTS           │
 │   Interact: n_estimators, max_depth                  │
 │   Visuals: Single decision tree visualization        │
 └─────────────────────────────────────────────────────┘

Phase 4: CONTINUOUS ACTIONS
 ┌─────────────────────────────────────────────────────┐
 │ Lesson 14: CATS — "The Fine-Tuner"                   │
 │   Teaches: Continuous action spaces (not just A/B/C) │
 │   Interact: Action range [a_min, a_max], depth       │
 │   Visuals: Action tree, leaf selection animation     │
 └─────────────────────────────────────────────────────┘

Phase 5: PRODUCTION FEATURES
 ┌─────────────────────────────────────────────────────┐
 │ Lesson 15: Drift Detection — "The Vigilant One"      │
 │   Teaches: Detecting when the world changes          │
 │   Interact: Inject drift event, delta/lambda sliders │
 │   Visuals: Cumulative deviation tracker, alarm flash  │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 16: Offline Evaluation — "The Historian"      │
 │   Teaches: Evaluating policies from logged data       │
 │   Interact: Upload CSV (or pre-loaded data)           │
 │   Visuals: IPS vs DR vs NCIS comparison bars         │
 ├─────────────────────────────────────────────────────┤
 │ Lesson 17: Policy Comparison — "The Tournament"      │
 │   Two policies, same world, side-by-side             │
 │   Interact: Select any 2 policies, run head-to-head  │
 │   Visuals: Split arena, diff charts (reward delta)   │
 └─────────────────────────────────────────────────────┘

Phase 6: SANDBOX
 ┌─────────────────────────────────────────────────────┐
 │ Free-form Experiment — All 30+ params, any world     │
 │   Export traces as CSV, save/load scenarios          │
 └─────────────────────────────────────────────────────┘
```

---

## 4. Theory Pedagogy — How Each Algorithm Is Taught

Each lesson follows a strict **5-stage pedagogy template**. Every stage has a dedicated UI section.

### 4.1 The 5-Stage Theory Card

```
┌──────────────────────────────────────────────────────┐
│  Understanding UCB1                      [Collapse]  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Stage 1: Intuition — "The Big Picture"             │
│  ┌────────────────────────────────────────────────┐ │
│  │ Imagine you're a doctor with 3 treatments.     │ │
│  │ You've tried Pill A 50 times (70% success).    │ │
│  │ You've tried Injection B only 2 times.         │ │
│  │ Should you stick with Pill A or try B more?    │ │
│  │                                                │ │
│  │ UCB1 says: "Be optimistic. Give B the benefit  │ │
│  │ of the doubt. If B is truly worse, more data   │ │
│  │ will confirm it. If B is actually better,      │ │
│  │ being optimistic discovers it faster."         │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  Stage 2: Interactive Visual — "See It Work"       │
│  ┌────────────────────────────────────────────────┐ │
│  │  Arm A: [████████████░░░░]  Mean: 0.70         │ │
│  │          [──CI──]  Width: narrow (50 pulls)    │ │
│  │                                                │ │
│  │  Arm B: [██░░░░░░░░░░░░░░]  Mean: 0.50         │ │
│  │          [────────CI────────]  Width: wide      │ │
│  │          (only 2 pulls)                         │ │
│  │                                                │ │
│  │  UCB1 picks B: its upper confidence bound       │ │
│  │  (0.50 + wide CI) > Arm A's (0.70 + narrow)    │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  Stage 3: Formula — "The Math, Made Simple"        │
│  ┌────────────────────────────────────────────────┐ │
│  │  score_i = mean_i + [alpha] * sqrt(2*ln(N)/n_i)│ │
│  │                                                │ │
│  │  mean_i = average reward for arm i             │ │
│  │  N      = total pulls across all arms          │ │
│  │  n_i    = pulls for this arm only              │ │
│  │  alpha  = how optimistic you are               │ │
│  │                                                │ │
│  │  The sqrt term SHRINKS as n_i grows:           │ │
│  │  more pulls -> tighter confidence -> less bonus│ │
│  │  The ln(N) term GROWS slowly: unpulled arms    │ │
│  │  accumulate a growing bonus over time.         │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  Stage 4: Interactive Controls — "Try It"          │
│  ┌────────────────────────────────────────────────┐ │
│  │  alpha (Exploration): [══════●═══════] 1.00    │ │
│  │  Higher -> more optimistic, try uncertain arms │ │
│  │  Lower  -> stick to known best arms            │ │
│  │                                                │ │
│  │  [Step]  [Run 50]  [Reset]                     │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  Stage 5: Live Metrics — "What's Happening"        │
│  ┌────────────────────────────────────────────────┐ │
│  │  Cumulative Reward: 42.3    Regret: 8.7        │ │
│  │  Arm A: 23 pulls (0.72) [████████░░]           │ │
│  │  Arm B: 15 pulls (0.81) [██████████]           │ │
│  │  Arm C: 12 pulls (0.45) [████░░░░░░]           │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 4.2 Theory Content Per Lesson

Each lesson has a `TheoryContent` dataclass:

```python
@dataclass
class TheoryContent:
    intuition: str           # Plain English, 2-3 sentences. No math.
    metaphor: str            # Real-world analogy (1 sentence)
    visual_explanation: str  # What the animation/chart shows
    formula_human: str       # Formula in words, not symbols
    formula_math: str        # The actual formula (for curious learners)
    formula_parts: dict[str, str]  # Variable → plain-English explanation
    key_insight: str         # The one thing to remember
    when_to_use: str         # "Use this when..." guidance
    pitfalls: str            # Common gotchas
    comparison_to: str | None  # "How this differs from X" (for related algorithms)
```

### 4.3 Visual Pedagogy Per Algorithm

| Algorithm | Signature Visualization | What The User Sees |
|-----------|----------------------|-------------------|
| **UCB1** | Confidence interval bars per arm | Bars with shaded "maybe zone" shrinking over time |
| **Thompson** | Beta distribution curves | Bell-like curves per arm narrowing with data |
| **Epsilon-Greedy** | Dice-roll animation | Random arm highlighted with "Exploring" or "Best choice" label |
| **LinUCB** | Feature impact arrows + weight vectors | Each patient feature shows impact direction/size per treatment |
| **LinTS** | Elliptical confidence regions | 2D ellipse (if 2 features) shrinking toward true weight |
| **Softmax** | Probability pie chart | Dynamic pie where slices change size with τ slider |
| **Logistic** | Sigmoid decision boundary | S-curve shifting with each observation |
| **GP-UCB** | 2D heatmap of uncertainty | Color surface showing where the GP is confident vs uncertain |
| **Sliding Window** | Window highlight animation | Old data fading, new data bright — window slides forward |
| **Bootstrapped** | Dot cloud of ensemble predictions | 10+ dots per arm showing disagreement → uncertainty |
| **LinUCB Hybrid** | Split context bar | Color-coded: yellow (shared features), blue (per-arm features) |
| **Neural Linear** | Embedding projection | 2D scatter of context embeddings colored by reward |
| **Tree Ensemble** | Single decision tree | Interactive tree where nodes light up during prediction |
| **CATS** | Search tree with zoom | Binary tree with highlighted path to selected leaf |
| **Drift Detection** | Cumulative deviation tracker | Rising line with alarm threshold — flashes red on detection |
| **Offline Eval** | Method comparison bars | 3 bars (Rejection, DR, NCIS) with utilization rate labels |

---

## 5. Algorithm Debugger — Step-by-Step Formula Visualization

### 5.0 Concept

Every algorithm has a unique "computation flow" — a sequence of formula terms that transform inputs (pull counts, context vector, posterior parameters) into a decision (which arm to pick). The **Algorithm Debugger** is a per-step panel that exposes this internal computation visually, like a code debugger but for math.

**Design principle:** Each algorithm type gets its own debug layout because the internal state and formula structure are fundamentally different. The debugger lives in a collapsible panel below the TracePanel.

### 5.1 Debug Data Schema

Every `StepRecord` carries an optional `debug_snapshot: AlgorithmDebugSnapshot` captured inside the decision path before the model update. The UI only renders this snapshot; it never calls `model.score()` again to reconstruct stochastic values.

```python
# src/coba_web/engine/debug.py

@dataclass
class AlgorithmDebugSnapshot:
    """Per-algorithm internal state at decision time."""
    algorithm: str                         # "ucb1", "linucb", "thompson", ...
    selected: str | float                  # Chosen arm or continuous action
    arm_names: list[str]                   # ["A", "B", "C"] for discrete policies
    context: np.ndarray | None             # Context vector (None for context-free)
    per_arm: dict[str, PerArmDebugState]   # Keyed by arm name
    scores: dict[str, float]               # Exact scores/samples used for the decision
    total_pulls: int                       # N (for UCB1)
    routed_cluster: int | None             # ClusterRouter result for discrete contextual policies
    continuous: ContinuousDebugState | None # CATS-only details
    extra: dict[str, Any] | None           # e.g., softmax probabilities, GP kernel size

@dataclass
class ContinuousDebugState:
    action: float
    propensity: float
    leaf_index: int
    leaf_lo: float
    leaf_hi: float
    leaf_scores: dict[int, float]
    mean_estimate: float
    confidence_width: float

@dataclass
class PerArmDebugState:
    n_pulls: int
    # Context-free UCB
    mean_reward: float | None
    exploration_bonus: float | None
    # Thompson (Beta) — sample is the exact draw used for the decision
    alpha: float | None
    beta: float | None
    sample: float | None
    # Linear models (UCB, TS, Softmax, Logistic)
    coefficients: np.ndarray | None
    mean_estimate: float | None
    confidence_width: float | None
    # GP-UCB
    gp_mean: float | None
    gp_std: float | None
    # Bootstrapped
    ensemble_predictions: list[float] | None
    # Softmax
    softmax_probability: float | None
    # Tree Ensemble
    tree_predictions: list[float] | None
```

### 5.2 Debugger Layout Per Algorithm Group

#### Group A: Context-Free UCB (UCB1)

```
Algorithm Debugger: UCB1 — Step 42
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  INPUT STATE                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Total pulls (N) = 42                                │    │
│  │                                                      │    │
│  │  Arm │ Pulls (n) │  Sum   │ Mean (mu) │ Bonus Term │    │
│  │  ----|-----------|--------|-----------|------------│    │
│  │  A   │    20     │  14.0  │  0.700   │ a*sqrt(2ln42/20)│  │
│  │      │           │        │          │ =1.0*0.612  │    │
│  │      │           │        │          │ = 0.612     │    │
│  │  ----|-----------|--------|-----------|------------│    │
│  │  B   │    15     │  12.0  │  0.800   │ =1.0*0.706  │    │
│  │  ----|-----------|--------|-----------|------------│    │
│  │  C   │     7     │   3.0  │  0.429   │ =1.0*1.033  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  COMPUTATION (animated, highlights as it computes)           │
│                                                              │
│  Step 1: bonus = alpha * sqrt(2*ln(N) / n_i)                │
│  Step 2: score = mu + bonus                                 │
│                                                              │
│  Arm A:   0.700 + 0.612 = 1.312                             │
│  Arm B:   0.800 + 0.706 = 1.506  << WINNER                  │
│  Arm C:   0.429 + 1.033 = 1.462                             │
│                                                              │
│  DECISION                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Arm B wins (score 1.506 > 1.462 > 1.312)           │    │
│  │  Despite lower mean, B wins because its bonus       │    │
│  │  (0.71) is larger — fewer pulls means more optimism. │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  AFTER REWARD (if reward = 0)                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Arm B: pulls 15->16, sum stays 12.0                │    │
│  │         mu: 0.800 -> 0.750                          │    │
│  │  NEXT bonus: a*sqrt(2ln43/16) = 1.0*sqrt(7.52/16)   │    │
│  │             = 0.686  <-- bonus SHRANK (more data)   │    │
│  │  The bonus shrinks because n_i grew --              │    │
│  │  confidence increased, less optimism needed.         │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Group B: Context-Free Bayesian (Thompson Sampling)

```
Algorithm Debugger: Thompson Sampling — Step 42
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  POSTERIOR STATE (per arm)                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Arm │ Success (a)│ Failure (b)│ Beta(a,b)          │    │
│  │  ----|------------|------------|--------------------|    │
│  │  A   │     15     │      5     │ [████░░░░]  75%   │    │
│  │  B   │     12     │      3     │ [██████░░]  80%   │    │
│  │  C   │      4     │      5     │ [███░░░░░]  44%   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  SAMPLE FROM EACH POSTERIOR (random draw)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Arm A: Beta(15,5) -> sample = 0.72                 │    │
│  │  Arm B: Beta(12,3) -> sample = 0.91  <-- high draw │    │
│  │  Arm C: Beta(4,5)  -> sample = 0.38                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  DECISION: Arm B (highest sample: 0.91)                      │
│                                                              │
│  Even though Arm A has similar mean (75% vs 80%),            │
│  Arm B's Beta(12,3) has more variance — sometimes            │
│  samples higher. Thompson naturally explores.                │
│                                                              │
│  AFTER REWARD (if reward = 1)                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Arm B: alpha: 12->13, beta: 3->3 (success!)        │    │
│  │  Posterior mean: 80.0% -> 81.25%                    │    │
│  │  [██████░░]  ->  [██████░░]  (narrower, higher)    │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Group C: Linear UCB (LinUCB, LinUCB-SW)

```
Algorithm Debugger: LinUCB — Step 42
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  CONTEXT VECTOR (Patient #42)                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  x = [age_z: 0.52,  bp_z: -1.20,  temp_z: 0.83]    │    │
│  │       Middle-aged        Low BP          Fever      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  PER-ARM COMPUTATION                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │  Pill A (beta = [1.20, -0.50, 0.30])              │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │  mean  = x @ beta                           │   │    │
│  │  │        = 0.52*1.20 + (-1.20)*(-0.50)       │   │    │
│  │  │          + 0.83*0.30                        │   │    │
│  │  │        = 0.624 + 0.600 + 0.249              │   │    │
│  │  │        = 1.473                              │   │    │
│  │  │                                            │   │    │
│  │  │  width = alpha * sqrt(x @ A_inv @ x)       │   │    │
│  │  │        = 1.0 * sqrt(0.187)                  │   │    │
│  │  │        = 0.432                              │   │    │
│  │  │  UCB   = 1.473 + 0.432 = 1.905             │   │    │
│  │  └─────────────────────────────────────────────┘   │    │
│  │                                                     │    │
│  │  Injection B (beta = [-0.30, 1.50, -0.10])        │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │  mean  = 0.52*(-0.30) + (-1.20)*1.50       │   │    │
│  │  │          + 0.83*(-0.10)                     │   │    │
│  │  │        = -2.039                             │   │    │
│  │  │  width = 1.0 * sqrt(0.215) = 0.464          │   │    │
│  │  │  UCB   = -2.039 + 0.464 = -1.575            │   │    │
│  │  └─────────────────────────────────────────────┘   │    │
│  │                                                     │    │
│  │  Therapy C (beta = [0.50, -0.80, -0.60])          │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │  mean  = ... = 0.722                        │   │    │
│  │  │  width = 1.0 * sqrt(0.312) = 0.559           │   │    │
│  │  │  UCB   = 0.722 + 0.559 = 1.281              │   │    │
│  │  └─────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  DECISION: Pill A wins (UCB 1.905 > 1.281 > -1.575)         │
│                                                              │
│  Pill A won because this patient (middle age, low BP,        │
│  fever) matches Pill A's weight pattern. beta shows A        │
│  rewards middle-aged (+1.20), low-BP patients (+0.60).       │
│                                                              │
│  AFTER REWARD                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  A_inv updated via Sherman-Morrison (x @ A_inv @ x  │    │
│  │  shrinks -> confidence width decreases next step)   │    │
│  │  beta updated toward observed reward                │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Group D: Softmax

```
Algorithm Debugger: Softmax — Step 42
┌──────────────────────────────────────────────────────────────┐
│  MEAN ESTIMATES (same beta*x as LinUCB)                      │
│  Arm A: 1.473    Arm B: -2.039    Arm C: 0.722              │
│                                                              │
│  SOFTMAX TRANSFORM (tau = 1.0)                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  exp(mean / tau):                                   │    │
│  │  Arm A: exp(1.473) = 4.36                          │    │
│  │  Arm B: exp(-2.039) = 0.13                         │    │
│  │  Arm C: exp(0.722) = 2.06                          │    │
│  │  Sum = 6.55                                        │    │
│  │  Probabilities:                                     │    │
│  │  Arm A: 66.6%  [████████████████░░]                │    │
│  │  Arm B:  2.0%  [█░░░░░░░░░░░░░░░░░]                │    │
│  │  Arm C: 31.4%  [████████░░░░░░░░░░]                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  SAMPLED CHOICE: Arm A (rolled 0.52 of [0,1])               │
│                                                              │
│  tau=1.0 gives proportional probabilities.                   │
│  Try tau=0.1 -> almost always picks A (greedy).              │
│  Try tau=5.0 -> nearly uniform random.                       │
│  As user drags tau slider, probability bars animate live.    │
└──────────────────────────────────────────────────────────────┘
```

#### Group E: GP-UCB

```
Algorithm Debugger: GP-UCB — Step 42
┌──────────────────────────────────────────────────────────────┐
│  GAUSSIAN PROCESS STATE                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Kernel: RBF(length_scale=1.0)                      │    │
│  │  Stored observations: 120 (of max 500)              │    │
│  │  Kernel Matrix K (120x120) — Cholesky-decomposed   │    │
│  │  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  CONTEXT: x = [0.52, -1.20, 0.83]                           │
│                                                              │
│  POSTERIOR COMPUTATION (Arm A)                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  k(x) = [RBF(x, x_1), ... RBF(x, x_120)]            │    │
│  │       = 120 kernel similarities to stored points    │    │
│  │  mu(x)  = k(x) @ K_inv @ y = 0.624                 │    │
│  │  sigma2 = RBF(x,x) - k(x) @ K_inv @ k(x)           │    │
│  │         = 1.0 - 0.813 = 0.187                      │    │
│  │  sigma  = 0.432                                    │    │
│  │  UCB    = mu(x) + beta * sigma(x)                   │    │
│  │         = 0.624 + 2.0 * 0.432 = 1.488              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  UNCERTAINTY HEATMAP (context feature space, 2D slice)       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │       age_z ->                                      │    │
│  │  -2    -1     0     1     2     (bp_z rows)        │    │
│  │  [light][light][dark][light][    ]  -2              │    │
│  │  [light][dark][dark][dark][light]  -1              │    │
│  │  [dark][dark[x][dark][dark][dark]   0  <-- you     │    │
│  │  [light][dark][dark][dark][light]   1     here     │    │
│  │  [    ][light][dark][light][light]   2              │    │
│  │  dark = explored (low var)  light = uncertain        │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Group F: Bootstrapped Ensemble

```
Algorithm Debugger: BootstrappedTS — Step 42
┌──────────────────────────────────────────────────────────────┐
│  CONTEXT: x = [0.52, -1.20, 0.83]                           │
│                                                              │
│  ENSEMBLE VOTES (10 models per arm)                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Pill A                                             │    │
│  │  Predictions: 0.62 0.58 0.71 0.59 0.66 0.63        │    │
│  │               0.55 0.68 0.60 0.64                   │    │
│  │  Mean: 0.626    Std: 0.049                          │    │
│  │  BootstrappedTS: sample ~ N(0.626, 0.049^2)          │    │
│  │  -> sampled 0.641                                   │    │
│  │  BootstrappedUCB: 0.626 + alpha * 0.049 = 0.675     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Uncertainty comes from model DISAGREEMENT.                  │
│  All models agree -> narrow CI -> less exploration.          │
│  Models disagree -> wide CI -> more exploration.             │
└──────────────────────────────────────────────────────────────┘
```

#### Group G: Neural Linear

```
Algorithm Debugger: NeuralLinear — Step 42
┌──────────────────────────────────────────────────────────────┐
│  RAW CONTEXT: x = [0.52, -1.20, 0.83]  (3 features)        │
│                                                              │
│  NEURAL BACKBONE (MLP: 3->64->32->16)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Input(3) -> Hidden(64,ReLU) -> Hidden(32,ReLU)     │    │
│  │           -> Embedding(16)                           │    │
│  │  z = [0.12, -0.45, 0.33, -0.08, 0.67, -0.22,       │    │
│  │       0.51, -0.19, 0.41, -0.03, 0.28, -0.55,       │    │
│  │       0.18, 0.72, -0.31, 0.09]                      │    │
│  │  [Visual: 16 colored boxes by value magnitude]      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  THEN LinTS ON EMBEDDING (same as LinTS, but on z)          │
│  Per-arm: mean = z @ beta_z, width = v * sqrt(z @ A_inv @ z)│
│                                                              │
│  RETRAIN COUNTER: 184/200 pulls until next retrain           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Progress: [████████████████████░░]  92%             │    │
│  │  At 200: backbone retrains, LinTS models rebuild.   │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Group H: Logistic Bandits

```
Algorithm Debugger: LogisticUCB — Step 42
┌──────────────────────────────────────────────────────────────┐
│  CONTEXT: x = [0.52, -1.20, 0.83]                           │
│                                                              │
│  LOGISTIC MODEL (per arm)                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  w = [1.20, -0.50, 0.30]  (weight vector)           │    │
│  │  H_inv = 3x3 inverse Hessian (covariance approx)    │    │
│  │  logit = x @ w = 1.473                             │    │
│  │  p = sigmoid(1.473) = 0.813                        │    │
│  │  UCB = 0.813 + alpha*sqrt(x@H_inv@x * sigmoid')    │    │
│  └─────────────────────────────────────────────────────┘    │
│  Logistic is for binary (0/1) rewards. Sigmoid keeps         │
│  predictions in [0,1]. H_inv approximates the posterior.     │
└──────────────────────────────────────────────────────────────┘
```

#### Group I: LinUCB Hybrid

```
Algorithm Debugger: LinUCB Hybrid — Step 42
┌──────────────────────────────────────────────────────────────┐
│  SPLIT CONTEXT: n_shared=2, n_per_arm=1                      │
│  ┌──────────────┬──────────┐                                 │
│  │ SHARED (z)   │ PER-ARM  │                                 │
│  │ age:0.52     │ temp:0.83│                                 │
│  │ bp:-1.20     │          │                                 │
│  ├──────────────┼──────────┤                                 │
│  │ Learned by   │ Learned  │                                 │
│  │ ALL arms     │ per arm  │                                 │
│  └──────────────┴──────────┘                                 │
│  SCORE = z@beta_shared + x_arm@theta_arm + bonus             │
│  Shared features pool data across arms -> faster learning.    │
└──────────────────────────────────────────────────────────────┘
```

#### Group J: Tree Ensemble (RandomForestUCB/TS)

```
Algorithm Debugger: RandomForestUCB — Step 42
┌──────────────────────────────────────────────────────────────┐
│  CONTEXT: x = [0.52, -1.20, 0.83]                           │
│                                                              │
│  TREE #3 (of 50), highlighted path:                          │
│  age_z<=0? -> NO -> temp_z<=-0.3? -> YES -> bp_z<=0.2?     │
│  -> YES -> Leaf: reward=0.72                                │
│                                                              │
│  All 50 trees: Mean=0.626, Std=0.049                        │
│  UCB = 0.626 + alpha*0.049 = 0.675                           │
│                                                              │
│  Uncertainty = how much trees disagree. No math needed.      │
└──────────────────────────────────────────────────────────────┘
```

#### Group K: CATS (Continuous Action Tree Sampling)

```
Algorithm Debugger: CATS — Step 42
┌──────────────────────────────────────────────────────────────┐
│  ACTION TREE (depth=3, 8 leaves, range [0, 100])            │
│  [0 -- 100] -> [25 -- 50] -> [25 -- 37.5]                   │
│  -> Leaf 2 selected -> action = 37.5                        │
│                                                              │
│  Per-leaf LinUCB scores: L2 wins (UCB=0.93)                 │
│                                                              │
│  CATS finds the optimal CONTINUOUS value (e.g., price        │
│  $37.50), not just discrete choices. The tree zooms into     │
│  promising regions.                                          │
└──────────────────────────────────────────────────────────────┘
```

#### Group L: Sliding-Window LinUCB

```
Algorithm Debugger: SlidingWindow LinUCB — Step 242
┌──────────────────────────────────────────────────────────────┐
│  WINDOW MANAGER (window_size = 200)                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Total observations ever: 242                        │    │
│  │  In window:           200 (steps 43-242)            │    │
│  │  Forgotten:            42 (steps 1-42)              │    │
│  │  [.... forgotten ....|██████ IN WINDOW ██████]     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  SCORE: Same as LinUCB but only uses window data             │
│  A = sum x_i * x_i^T  (only i in window)                    │
│  b = sum r_i * x_i    (only i in window)                    │
│  beta = A_inv * b                                           │
│                                                              │
│  Each new obs evicts oldest. Fast adaptation to change.      │
└──────────────────────────────────────────────────────────────┘
```

#### Group M: Drift Detection

```
Algorithm Debugger: Drift Detector — Step 42
┌──────────────────────────────────────────────────────────────┐
│  PAGE-HINKLEY TRACKER (per arm)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Pill A                                             │    │
│  │  Running mean: 0.70 -> 0.68 -> 0.55 (declining)    │    │
│  │  Cumulative dev (m_t): 12.3 -> 28.7 -> 62.1         │    │
│  │  Threshold (lambda): ------------------------------ 50.0│  │
│  │                                                     │    │
│  │  ALARM! m_t=62.1 > lambda=50.0                      │    │
│  │  -> Drift detected. Resetting arm A models.          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  AFTER RESET: Arm A starts fresh. History forgotten.         │
│  Algorithm re-explores to learn the new pattern.             │
│                                                              │
│  Drift detection notices when "the world changed."           │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 Debugger Component Architecture

```python
# src/coba_web/components/debugger.py

class AlgorithmDebugger(ft.Container):
    """Collapsible panel exposing algorithm internals for the current step.

    Routes to per-algorithm renderer based on snapshot.algorithm.
    Each renderer builds a unique computation-flow visualization.
    """

    def render(self, snapshot: AlgorithmDebugSnapshot) -> None:
        renderers = {
            "ucb1": self._render_ucb1,
            "thompson": self._render_thompson,
            "linucb": self._render_linear_ucb,
            "linucb_sw": self._render_linear_ucb,
            "cats": self._render_cats,  # Separate continuous-action snapshot
            "lints": self._render_linear_ts,
            "logistic_ts": self._render_linear_ts,
            "softmax": self._render_softmax,
            "gp_ucb": self._render_gp_ucb,
            "bootstrapped_ts": self._render_bootstrapped,
            "bootstrapped_ucb": self._render_bootstrapped,
            "neural_linear": self._render_neural_linear,
            "logistic_ucb": self._render_logistic,
            "random_forest_ucb": self._render_tree_ensemble,
            "random_forest_ts": self._render_tree_ensemble,
            "linucb_hybrid": self._render_linucb_hybrid,
        }
        renderer = renderers.get(snapshot.algorithm)
        if renderer:
            self.content = renderer(snapshot)

    def _render_ucb1(self, snap) -> ft.Column:
        """N, n_i, μ_i, bonus_i, score_i table with per-term expansion."""
        rows = []
        for arm_name in snap.arm_names:
            state = snap.per_arm[arm_name]
            bonus = state.exploration_bonus or float("inf")
            score = (state.mean_reward or 0) + (bonus if bonus != float("inf") else bonus)
            rows.append(_FormulaRow(
                arm=arm_name,
                terms=[
                    _Term("n", f"{state.n_pulls}", "Pull count"),
                    _Term("μ", f"{state.mean_reward:.3f}", "Mean reward"),
                    _Term("+", "", ""),
                    _Term("bonus", f"{bonus:.3f}",
                          f"α·√(2·ln({snap.total_pulls})/{state.n_pulls})"),
                    _Term("=", "", ""),
                    _Term("score", f"{score:.3f}", "UCB score", highlight=arm_name == snap.extra["chosen_arm"]),
                ]
            ))
        return ft.Column([ft.Text("UCB1 Formula Breakdown", weight="bold"), *rows])
```

### 5.4 Debugger Data Capture

Debugger capture is an engine adapter responsibility. The discrete adapter calls a small public debug hook on `ClusterRouter` so it records the routed cluster and exact per-arm scores used for selection without reaching into private router fields. The continuous adapter records the `ContinuousDecision` returned by CATS.

```python
# src/coba_web/engine/discrete_session.py

def step(self) -> StepRecord:
    context = self._next_context()
    decision, debug = self._decide_with_debug(context)
    reward = self._reward_fn(context, decision.arm)
    self._bandit.update(context, decision.arm, reward)
    return StepRecord(..., debug_snapshot=debug)


def _decide_with_debug(self, context: np.ndarray) -> tuple[BanditDecision, AlgorithmDebugSnapshot]:
    debug_decision = self._router.predict_with_debug(context)
    # predict_with_debug scores each arm exactly once, so Thompson/TS samples are faithful.
    debug = AlgorithmDebugSnapshot(
        algorithm=self.config.policy.value,
        selected=debug_decision.selected_arm,
        arm_names=[str(a) for a in self.arms],
        context=debug_decision.routed_context,
        per_arm=debug_decision.per_arm_state,
        scores=debug_decision.scores,
        total_pulls=self.total_pulls,
        routed_cluster=debug_decision.cluster_idx,
        continuous=None,
        extra=debug_decision.extra,
    )
    return debug_decision.to_bandit_decision(), debug
```

```python
# src/coba_web/engine/continuous_session.py

def step(self) -> StepRecord:
    context = self._next_context()
    decision = self._continuous_bandit.decide(context)
    reward = self._reward_fn(context, decision.chosen_action)
    self._continuous_bandit.update(context, decision.chosen_action, reward, decision.propensity)
    return StepRecord(
        ...,
        debug_snapshot=AlgorithmDebugSnapshot(
            algorithm="cats",
            selected=decision.chosen_action,
            arm_names=[],
            context=context,
            per_arm={},
            scores={str(k): v for k, v in decision.leaf_scores.items()},
            total_pulls=self.total_pulls,
            routed_cluster=None,
            continuous=ContinuousDebugState.from_decision(decision),
            extra={},
        ),
    )
```

### 5.5 Debugger Placement in UI

```
┌── TRACE (last 20 steps, collapsible) ──────────────────────┐
│  Step │ Context │ Chose │ Reward │ Regret │ Scores          │
├──────────────────────────────────────────────────────────────┤
│  41   │ [0.5,..│  A    │  1.0   │  0.0   │ A:1.01 B:0.89  │
│  42   │ [0.5,..│  B    │  0.0   │ -0.8   │ A:0.95 B:1.15  │ <-- selected
└──────────────────────────────────────────────────────────────┘

┌── ALGORITHM DEBUGGER (collapsible) ────────────────────────┐
│  Step: [< 41] [42 selected] [43 >]    Algorithm: UCB1      │
│                                                              │
│  ANIMATION: [Replay] rewinds computation step-by-step       │
│  Each term highlights as "computed" — 200ms per term       │
│                                                              │
│  INPUT STATE                                                │
│  N=42 | Arm | n | mu   | bonus = a*sqrt(2lnN/n) | score   │
│  -----|-----|---|------|------------------------|-------- │
│       | A   |20 |0.700 | 1.0*sqrt(7.48/20)=0.612 | 1.312  │
│       | B   |15 |0.800 | 1.0*sqrt(7.48/15)=0.706 | 1.506  | <-- winner
│       | C   | 7 |0.429 | 1.0*sqrt(7.48/7) =1.033 | 1.462  │
│                                                              │
│  AFTER REWARD (B got 0): mu_B=0.800->0.750, bonus shrinks  │
└──────────────────────────────────────────────────────────────┘
```

**Animation replay flow:** Click ▶ → (1) Input state rows appear with fade-in (300ms) → (2) Per-arm bonus terms compute left-to-right with staggered delay (200ms each) → (3) Score column pulses on winner (scale 1.0→1.15→1.0, 400ms) → (4) After-reward section slides in (500ms).

---

## 6. Configuration Control System — Tooltips & Hover Help

### 6.1 Design Philosophy

**Every interactive control** (slider, switch, dropdown, button) has a **hover tooltip** that explains:
1. **What it does** — in plain English (max 1 sentence)
2. **What happens if you increase it** — behavioral effect
3. **What happens if you decrease it** — behavioral effect
4. **Recommended starting value** — default
5. **Which algorithms use it** — context badge

### 6.2 Tooltip Component

```python
# src/coba_web/components/tooltips.py

@dataclass
class ParamTooltip:
    param_key: str              # "alpha", "epsilon", "l2_lambda", ...
    label: str                  # "Exploration Bonus (α)"
    short_desc: str             # "Controls how optimistic UCB is about untested arms"
    increase_effect: str        # "Higher → tries uncertain arms more often"
    decrease_effect: str        # "Lower → sticks to known best arms"
    default: float | int | bool
    range_hint: str             # "Typical: 0.5–2.0"
    applies_to: list[str]       # ["UCB1", "LinUCB", "GP-UCB", "LogisticUCB", ...]
    beginner_tip: str           # Extra help for beginners

@dataclass
class ParamControlSpec:
    """Numeric control metadata kept separate from explanatory copy."""
    param_key: str
    min_value: float | int
    max_value: float | int
    divisions: int | None = 100


# Master registry — every config param has an entry
PARAM_TOOLTIPS: dict[str, ParamTooltip] = {
    # ===== Core Exploration =====
    "alpha": ParamTooltip(
        param_key="alpha",
        label="Exploration Bonus (α)",
        short_desc="How much extra credit you give to arms you haven't tried much yet.",
        increase_effect="More exploration — you'll try uncertain arms more often. Good when the best arm isn't obvious yet.",
        decrease_effect="More exploitation — you'll stick to what's working. Good when you're confident.",
        default=1.0,
        range_hint="0.1 (nearly greedy) to 5.0 (very exploratory)",
        applies_to=["UCB1", "LinUCB", "LogisticUCB", "BootstrappedUCB", "LinUCBSW", "CATS"],
        beginner_tip="Start at 1.0. If the algorithm isn't trying new arms, increase α. If it's trying bad arms too often, decrease α.",
    ),

    "epsilon": ParamTooltip(
        param_key="epsilon",
        label="Exploration Rate (ε)",
        short_desc="The chance of picking a completely random arm each round (ignoring what you've learned).",
        increase_effect="More random exploration — like flipping a coin before each decision. Good for discovering hidden gems.",
        decrease_effect="More greedy — almost always picks the best-known arm. Good when you're already confident.",
        default=0.1,
        range_hint="0.01 (1% random) to 0.3 (30% random)",
        applies_to=["EpsilonGreedy"],
        beginner_tip="ε=0.1 means '90% of the time pick best, 10% pick random'. This is the classic starting point.",
    ),

    # ===== Linear Models =====
    "l2_lambda": ParamTooltip(
        param_key="l2_lambda",
        label="Regularization (λ)",
        short_desc="Prevents the model from overfitting to noise. Like saying 'don't trust any single data point too much.'",
        increase_effect="Stronger regularization — smoother, more stable but slower to adapt to patterns.",
        decrease_effect="Weaker regularization — faster to adapt but may overreact to noise.",
        default=1.0,
        range_hint="0.1 (flexible) to 10.0 (conservative)",
        applies_to=["LinUCB", "LinTS", "Softmax", "LogisticUCB", "LogisticTS", "LinUCBHybrid", "LinUCBSW"],
        beginner_tip="1.0 is a safe default. If rewards are very noisy, increase λ. If patterns are strong, decrease λ.",
    ),

    "v_sq": ParamTooltip(
        param_key="v_sq",
        label="Posterior Variance (v²)",
        short_desc="How uncertain the Bayesian model starts out. Higher = more exploration initially.",
        increase_effect="Wider initial uncertainty → more exploration in early steps. Model is 'less sure' at the start.",
        decrease_effect="Narrower initial uncertainty → less exploration early. Model acts more confidently from the start.",
        default=1.0,
        range_hint="0.1 to 5.0",
        applies_to=["LinTS", "LogisticTS", "NeuralLinear"],
        beginner_tip="1.0 is standard. Higher values encourage more early exploration (like a larger α in UCB terms).",
    ),

    # ===== Softmax =====
    "softmax_tau": ParamTooltip(
        param_key="softmax_tau",
        label="Temperature (τ)",
        short_desc="Controls how 'sharp' the selection probabilities are. Like turning up the 'randomness dial.'",
        increase_effect="Higher τ → more uniform random selection. At τ=10, all arms get roughly equal chance.",
        decrease_effect="Lower τ → more greedy. At τ=0.01, almost always picks the best arm.",
        default=1.0,
        range_hint="0.01 (greedy) to 10.0 (uniform random)",
        applies_to=["Softmax"],
        beginner_tip="τ=1.0 gives proportional selection. Try τ=0.1 to see nearly-greedy behavior, τ=5.0 for near-random.",
    ),

    # ===== Sliding Window =====
    "linucb_sw_window": ParamTooltip(
        param_key="linucb_sw_window",
        label="Window Size",
        short_desc="How many recent observations the algorithm remembers. Older data is completely forgotten.",
        increase_effect="Longer memory — adapts slowly. Good when the world changes gradually.",
        decrease_effect="Shorter memory — adapts quickly. Good when the world changes abruptly (like a sudden trend shift).",
        default=200,
        range_hint="50 (fast adaptation) to 500 (stable memory)",
        applies_to=["LinUCBSW"],
        beginner_tip="200 is a good start. If you inject a 'drift event' and the algorithm is slow to notice, shrink the window.",
    ),

    # ===== GP-UCB =====
    "gp_beta": ParamTooltip(
        param_key="gp_beta",
        label="GP Exploration (β)",
        short_desc="Width of the Gaussian Process confidence interval. Higher = wider uncertainty bands.",
        increase_effect="More exploration — the GP will sample more widely across the context space.",
        decrease_effect="More exploitation — the GP focuses on regions it already knows are good.",
        default=2.0,
        range_hint="0.5 to 5.0",
        applies_to=["GPUCB"],
        beginner_tip="β=2.0 gives ~95% confidence intervals. This is mathematically grounded — start here.",
    ),

    "gp_length_scale": ParamTooltip(
        param_key="gp_length_scale",
        label="Smoothness (Length Scale)",
        short_desc="How quickly the GP assumes rewards change when context features change.",
        increase_effect="Smoother predictions — assumes similar patients get similar rewards even far apart in feature space.",
        decrease_effect="Rougher predictions — the GP can change its mind quickly between nearby contexts.",
        default=1.0,
        range_hint="0.1 (wiggly) to 5.0 (very smooth)",
        applies_to=["GPUCB"],
        beginner_tip="1.0 is neutral. If the reward function looks smooth/wavy in the heatmap, increase. If it looks flat, decrease.",
    ),

    "gp_noise_var": ParamTooltip(
        param_key="gp_noise_var",
        label="Observation Noise (σ²)",
        short_desc="How much random noise you expect in rewards. Similar to λ in linear models.",
        increase_effect="Assumes rewards are very noisy → wider confidence intervals, more exploration.",
        decrease_effect="Assumes rewards are clean → narrower intervals, more exploitation.",
        default=0.1,
        range_hint="0.01 (clean signal) to 1.0 (very noisy)",
        applies_to=["GPUCB"],
        beginner_tip="0.1 works well for most simulations. Increase if rewards seem very random.",
    ),

    # ===== Bootstrapped =====
    "n_bootstraps": ParamTooltip(
        param_key="n_bootstraps",
        label="Ensemble Size",
        short_desc="Number of 'sub-committees' that vote on which arm to pick. More = better uncertainty estimate.",
        increase_effect="More ensemble members → smoother, more reliable uncertainty. But slower to compute.",
        decrease_effect="Fewer members → faster but noisier. With 2 members, you barely have a committee.",
        default=10,
        range_hint="5 to 50",
        applies_to=["BootstrappedTS", "BootstrappedUCB"],
        beginner_tip="10 is a good balance. At 50 you'll see very smooth confidence estimates. At 2 you'll see individual member votes.",
    ),

    # ===== LinUCB Hybrid =====
    "n_shared_features": ParamTooltip(
        param_key="n_shared_features",
        label="Shared Features Count",
        short_desc="How many context features are shared across all arms (learned jointly) vs per-arm only.",
        increase_effect="More shared features → faster learning because data from ALL arms helps every arm. Good when arms are related.",
        decrease_effect="Fewer shared features → each arm learns independently. Good when arms are very different.",
        default=0,
        range_hint="0 (all per-arm) to n_features (all shared)",
        applies_to=["LinUCBHybrid"],
        beginner_tip="If features describe the PATIENT (age, BP) → these are shared. If features describe the TREATMENT → these are per-arm.",
    ),

    # ===== Neural Linear =====
    "neural_embedding_dim": ParamTooltip(
        param_key="neural_embedding_dim",
        label="Embedding Size",
        short_desc="How many 'compressed features' the neural network extracts from raw context before the linear model sees them.",
        increase_effect="More expressive representation — can capture more complex patterns. But needs more data to train.",
        decrease_effect="Simpler representation — faster to train but may miss subtle patterns.",
        default=16,
        range_hint="4 to 64",
        applies_to=["NeuralLinear"],
        beginner_tip="16 is standard. Think of this as 'smart feature engineering' done automatically by a neural net.",
    ),

    "neural_retrain_freq": ParamTooltip(
        param_key="neural_retrain_freq",
        label="Retrain Frequency",
        short_desc="How many arm pulls before the neural network backbone is retrained on all accumulated data.",
        increase_effect="Less frequent retraining → faster per-step but model adapts slower. Good when data comes fast.",
        decrease_effect="More frequent retraining → model adapts faster but each retrain is costly. Good when data is scarce.",
        default=200,
        range_hint="50 to 500",
        applies_to=["NeuralLinear"],
        beginner_tip="200 pulls is a good cadence. You'll see a brief 'thinking' pause at each retrain — that's the neural net learning!",
    ),

    # ===== Tree Ensemble =====
    "rf_n_estimators": ParamTooltip(
        param_key="rf_n_estimators",
        label="Number of Trees",
        short_desc="How many decision trees are in the Random Forest ensemble.",
        increase_effect="More trees → smoother, more reliable predictions. But slower to train.",
        decrease_effect="Fewer trees → faster but noisier. With few trees, uncertainty comes from disagreement.",
        default=50,
        range_hint="10 to 100",
        applies_to=["RandomForestUCB", "RandomForestTS"],
        beginner_tip="50 trees gives stable predictions. Watch the tree visualization — each tree 'votes' on the predicted reward!",
    ),

    "rf_max_depth": ParamTooltip(
        param_key="rf_max_depth",
        label="Max Tree Depth",
        short_desc="How deep each decision tree can grow. Limits complexity to prevent overfitting.",
        increase_effect="Deeper trees → can capture more complex patterns but may overfit to noise.",
        decrease_effect="Shallower trees → simpler rules, less likely to overfit. Good when data is noisy.",
        default=6,
        range_hint="3 (simple rules) to 15 (complex patterns)",
        applies_to=["RandomForestUCB", "RandomForestTS"],
        beginner_tip="Depth 6 means each tree asks at most 6 questions about the context. Watch the tree diagram to see the questions!",
    ),

    # ===== CATS =====
    "cats_depth": ParamTooltip(
        param_key="cats_depth",
        label="Tree Depth",
        short_desc="Depth of the binary action tree. Controls how finely the continuous action space is divided.",
        increase_effect="More leaves (2^depth) → finer-grained actions. You can pick more precise values.",
        decrease_effect="Fewer leaves → coarser actions but faster to learn.",
        default=6,
        range_hint="3 (8 actions) to 10 (1024 actions)",
        applies_to=["CATS"],
        beginner_tip="Depth 6 = 64 possible action values. Each leaf is a LinUCB model learning independently. Watch the tree grow!",
    ),

    # ===== Drift Detection =====
    "drift_delta": ParamTooltip(
        param_key="drift_delta",
        label="Sensitivity (δ)",
        short_desc="How big a change must be before the detector calls it 'drift' (not just noise).",
        increase_effect="Only large shifts trigger detection → fewer false alarms but may miss subtle changes.",
        decrease_effect="Even small shifts trigger detection → catches everything but may false-alarm on noise.",
        default=0.005,
        range_hint="0.001 (ultra-sensitive) to 0.05 (only major shifts)",
        applies_to=["DriftDetection"],
        beginner_tip="0.005 means a 0.5% per-step shift will eventually trigger. Try injecting a big drift event and watch the detector spike!",
    ),

    "drift_lambda": ParamTooltip(
        param_key="drift_lambda",
        label="Detection Threshold (λ)",
        short_desc="How much evidence the detector needs before sounding the alarm. Higher = more cautious.",
        increase_effect="Requires more cumulative evidence → fewer false alarms but slower to detect real drift.",
        decrease_effect="Triggers faster → catches drift quickly but may false-alarm on random fluctuations.",
        default=50.0,
        range_hint="10 (trigger-happy) to 200 (very conservative)",
        applies_to=["DriftDetection"],
        beginner_tip="50 is standard. The cumulative tracker line must cross λ before the alarm sounds. Watch the tracker climb!",
    ),

    # ===== Environment =====
    "n_clusters": ParamTooltip(
        param_key="n_clusters",
        label="Context Clusters",
        short_desc="How many 'patient groups' the algorithm divides contexts into. Each group learns independently.",
        increase_effect="More groups → finer-grained learning but each group has less data. Risk of splitting too thin.",
        decrease_effect="Fewer groups → each group has more data but may mix different patient types together.",
        default=5,
        range_hint="1 (no clustering) to 10 (fine-grained)",
        applies_to=["All (via ClusterRouter)"],
        beginner_tip="5 is a safe default. At n_clusters=1, there's no clustering — all patients are treated as one group.",
    ),

    "gamma": ParamTooltip(
        param_key="gamma",
        label="Discount Factor (γ)",
        short_desc="How much the algorithm weights recent observations vs old ones. 1.0 = all equal.",
        increase_effect="Closer to 1.0 → all history matters equally. Good for stable environments.",
        decrease_effect="Closer to 0.9 → recent data matters more. Good when the world is slowly changing.",
        default=1.0,
        range_hint="0.90 to 1.00",
        applies_to=["All (linear/logistic models)"],
        beginner_tip="Leave at 1.0 unless you're in a changing environment. Then try 0.95–0.99 for gentle forgetting.",
    ),

    "min_pull_rates": ParamTooltip(
        param_key="min_pull_rates",
        label="Minimum Pull Guarantee",
        short_desc="Forces the algorithm to try specific arms at least X% of the time, even if they look bad.",
        increase_effect="More guaranteed exploration of specific arms. Useful for fairness or regulatory requirements.",
        decrease_effect="Less forced exploration. The algorithm has more freedom to optimize.",
        default="None (no minimum)",
        range_hint="e.g., {'Arm C': 0.05} = 5% minimum for Arm C",
        applies_to=["All"],
        beginner_tip="This is for fairness! Imagine you MUST show a new treatment to at least 5% of patients for regulatory compliance.",
    ),
}

CONTROL_SPECS: dict[str, ParamControlSpec] = {
    "alpha": ParamControlSpec("alpha", min_value=0.1, max_value=5.0),
    "epsilon": ParamControlSpec("epsilon", min_value=0.0, max_value=0.5),
    "l2_lambda": ParamControlSpec("l2_lambda", min_value=0.1, max_value=10.0),
    "softmax_tau": ParamControlSpec("softmax_tau", min_value=0.01, max_value=10.0),
    "linucb_sw_window": ParamControlSpec("linucb_sw_window", min_value=50, max_value=500, divisions=45),
    # ...same key coverage as PARAM_TOOLTIPS...
}
```

### 6.3 Tooltip Rendering in Flet

```python
class ParamSlider(ft.Column):
    """Slider with integrated tooltip via hover event."""

    def __init__(self, tooltip: ParamTooltip, spec: ParamControlSpec, on_change: Callable):
        self.tooltip_data = tooltip
        self.control_spec = spec
        self.slider = ft.Slider(
            min=spec.min_value,
            max=spec.max_value,
            value=tooltip.default,
            divisions=spec.divisions,
            on_change=on_change,
        )
        self.label = ft.Text(tooltip.label, weight="bold")
        self.hover_card = ft.Container(
            visible=False,
            animate_opacity=ft.Animation(200, "ease"),
            content=ft.Column([
                ft.Text(tooltip.short_desc, size=13),
                ft.Text(f"▲ Increase: {tooltip.increase_effect}", size=12, color=ft.Colors.GREEN_400),
                ft.Text(f"▼ Decrease: {tooltip.decrease_effect}", size=12, color=ft.Colors.RED_400),
                ft.Text(f"💡 {tooltip.beginner_tip}", size=12, italic=True),
                ft.Text(f"📋 Used by: {', '.join(tooltip.applies_to[:3])}", size=11, color=ft.Colors.GREY_400),
            ]),
            bgcolor=ft.Colors.SURFACE,
            border_radius=8,
            padding=12,
            shadow=ft.BoxShadow(blur_radius=8),
        )
        # Show tooltip on hover
        self.slider.on_hover = lambda e: setattr(self.hover_card, 'visible', e.data == 'true')
```

---

## 7. UI/UX Design System

### 7.1 Color Palette

```
Light Theme                    Dark Theme
─────────────────────────────────────────────
Background:   #F8FAFC        Background:   #0F172A
Surface:      #FFFFFF        Surface:      #1E293B
Border:       #E2E8F0        Border:       #334155
Text Primary: #0F172A        Text Primary: #F1F5F9
Text Muted:   #64748B        Text Muted:   #94A3B8
Accent:       #6366F1        Accent:       #818CF8
Success:      #10B981        Success:      #34D399
Danger:       #EF4444        Danger:       #F87171
Warning:      #F59E0B        Warning:      #FBBF24

Arm colors (consistent across themes):
  Arm A: #6366F1 (Indigo)    Arm B: #F59E0B (Amber)
  Arm C: #10B981 (Emerald)   Arm D: #EC4899 (Pink)
  Arm E: #8B5CF6 (Violet)
```

### 7.2 Typography

| Role | Flet Text Style | Size | Weight |
|------|----------------|------|--------|
| Lesson title | `DISPLAY_SMALL` | 32px | Bold |
| Section header | `HEADLINE_MEDIUM` | 24px | SemiBold |
| Body text | `BODY_LARGE` | 16px | Regular |
| Arm name | `TITLE_MEDIUM` | 20px | Bold |
| Metrics (reward) | `BODY_LARGE` | 18px | Bold, Monospace |
| Tooltip | `BODY_SMALL` | 13px | Regular |
| Badge | `LABEL_LARGE` | 14px | Medium |

### 7.3 Spacing Grid

```python
# src/coba_web/app_config.py
SPACING = {
    "xs": 4, "sm": 8, "md": 16, "lg": 24,
    "xl": 32, "2xl": 48, "3xl": 64,
}
```

### 7.4 Animation Patterns

| Interaction | Animation | Duration | Curve |
|------------|-----------|----------|-------|
| Patient/Card arrival | `animate_offset` + `animate_opacity` | 400ms | `EASE_OUT` |
| Arm selection highlight | `animate_scale` (1.0 → 1.05 + glow) | 300ms | `EASE_IN_OUT` |
| Reward reveal | `animate_scale` (0 -> 1.2 -> 1.0) | 500ms | `BOUNCE_OUT` |
| Confidence bound narrowing | `animate` on Container width | 600ms | `EASE_IN_OUT` |
| Chart data update | Built-in Flet chart animation | 300ms | Native |
| Regret increment | `animate_opacity` flash (red, 200ms) | 200ms | `LINEAR` |
| Belief "squish" (failure) | Confidence bar shrinks | 800ms | `EASE_OUT` |
| Theme toggle | Flet's `page.theme_mode` | 400ms | `EASE` |
| Drift alarm | Red border flash + scale pulse | 500ms × 3 | `EASE_IN_OUT` |
| Tree node highlight | Sequential node color change | 100ms/node | `LINEAR` |

### 7.5 Page Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  The Rural Clinic     │ Step 42/200    │ Theme    │ Settings       │ <- TopBar
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌── Theory Card (collapsible) ──────────────────────────────────┐  │
│  │  Stage 1 -> 2 -> 3 -> 4 -> 5                                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── ARENA ──────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  ┌ Environment ───────┐  ┌ Agent (Algorithm) ───────────────┐ │  │
│  │  │                    │  │                                    │ │  │
│  │  │  Patient #42       │  │  Choose Treatment:                │ │  │
│  │  │  Age: 52  BP: 140  │  │                                    │ │  │
│  │  │  Temp: 38.2        │  │  [Pill A] [Injection B] [Therapy] │ │  │
│  │  │                    │  │    0.42       0.81 <-      0.55    │ │  │
│  │  │  [Show Truth]      │  │                                    │ │  │
│  │  └────────────────────┘  │  Reward: Not recovered             │ │  │
│  │                          │  Regret: -0.39                    │ │  │
│  │                          └────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── Live Charts (collapsible) ───────────────────────────────────┐  │
│  │  [Reward] [Regret] [Pulls] [Means]   <-- tabs                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── Trace (last 20 steps, collapsible) ──────────────────────────┐  │
│  │  Step | Patient  | Chose | Reward | Regret | Scores            │  │
│  │  42   | 52,140.. |  B    |  0.0   | -0.39  | A:0.42 B:0.81.. │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  [Step] [Run 50] [Stop] [Reset]    Speed: [======●======] 3x       │ <- Bottom
└──────────────────────────────────────────────────────────────────────┘

  [Settings Drawer — slides in from right]
   ┌──────────────────────────┐
   │  Configuration            │
   │                          │
   │  World: [Dropdown]       │
   │  Algorithm: [Dropdown]   │
   │                          │
   │  alpha (Exploration)     │
   │  [==========●==========] │
   │  [hover -> tooltip]      │
   │                          │
   │  lambda (Regularization) │
   │  [==========●==========] │
   │  [hover -> tooltip]      │
   │                          │
   │  [Apply] [Reset Default] │
   └──────────────────────────┘
```

---

## 8. Component Architecture — Key Components

### TreatmentCard

```python
class TreatmentCard(ft.Container):
    """A single arm card with animated selection/reward states.

    States: idle → selected (scale 1.05, glow) → revealed (green/red tint)
    """
    def __init__(self, arm: ArmDef, on_choose: Callable):
        self.arm = arm
        self.score_text = ft.Text("—", size=24, weight="bold")
        self.confidence_bar = ft.Container(
            width=0,  # Animated via animate=
            height=6,
            bgcolor=arm.color,
            border_radius=3,
            animate=ft.Animation(600, ft.AnimationCurve.EASE_IN_OUT),
        )
        self.reward_emoji = ft.Text("", size=32, animate_scale=...)
```

### ScenePanel (replaces static PatientCard)

```python
class ScenePanel(ft.Container):
    """Dynamic environment panel that renders the current narrative world.

    For Clinic: shows patient vitals.
    For MovieMatch: shows viewer profile + movie poster.
    For ShopSmart: shows shopper cart + product thumbnails.
    """
    def __init__(self, world: WorldConfig):
        self.world = world
        self.feature_badges: list[ft.Container] = []
        self.arrival_animation = ...  # slide-in + fade
```

---

## 9. State Management

Same architecture as original plan — `AppStore` (dict-based observable) with `dispatch(action)` pattern. Added fields:

```python
store = {
    "world": WorldConfig,          # Currently selected narrative
    "config": BanditConfig,        # All 30+ hyperparameters
    "simulation": SimulationSession,
    "trace": list[StepRecord],
    "stats": AggregatedStats,
    "running": bool,
    "speed": int,                  # Steps/sec
    "ui": {
        "theme_mode": "light" | "dark",
        "drawer_open": bool,
        "theory_expanded": bool,
        "truth_revealed": bool,
        "tooltip_visible": dict[str, bool],  # Per-control tooltip state
    },
}
```

---

## 10. Data Visualization

All charts use **Flet's native `LineChart` and `BarChart`**. Custom visualizations for algorithm-specific pedagogy:

| Viz | Implementation | Algorithm |
|-----|---------------|-----------|
| Reward/Regret lines | `ft.LineChart` | All |
| Pull/Mean bars | `ft.BarChart` | All |
| Confidence intervals | `ft.Container` with `animate` width | UCB1, LinUCB, GP-UCB |
| Beta posteriors | `ft.Container` bar with shaded CI overlay | Thompson, BootstrappedTS |
| Probability pie | `ft.PieChart` | Softmax |
| GP heatmap | `ft.GridView` of colored `Container` cells | GP-UCB |
| Decision tree | `ft.Column` of nested `ft.Row` nodes | Tree Ensembles |
| Embedding scatter | `ft.Stack` with positioned dots | Neural Linear |
| CATS tree | Indented `ft.Row` tree with highlight path | CATS |
| Drift tracker | `ft.LineChart` with threshold line | Drift Detection |
| Offline eval bars | `ft.BarChart` with 3 bars + utilization labels | Offline Eval |

---

## 11. Implementation Roadmap

### MVP Phase 0: Scaffold (Day 1)
- Add `flet>=0.24` and `pendulum>=3.0` to deps, create `src/coba_web/` and `tests/coba_web/`
- `main.py` minimal entry, `app_config.py` with theme/colors/spacing
- Wire `coba-web = "coba_web.main:main"`

### MVP Phase 1: Discrete Engine + State (Days 2–4)
- `DiscreteSimulationSession`, `AppStore`, action creators
- Decision-faithful debug data model, but render only basic score tables in MVP
- 8 unit tests

### MVP Phase 2: Core Worlds (Days 5–6)
- Clinic, MovieMatch, and ShopSmart `WorldConfig` definitions
- World selector on home page
- 4 narrative tests

### MVP Phase 3: UI Shell + Controls (Days 7–9)
- `AppShell`, `TopBar`, `BottomControls`, `ControlDrawer`
- Theme toggle and tooltip system with numeric slider bounds
- 5 tests

### MVP Phase 4: Arena + Charts + Trace (Days 10–13)
- `ScenePanel`, `TreatmentPanel`, `TreatmentCard`
- Reward/regret lines, pull/mean bars, trace table, run/stop/reset
- 8 tests

### MVP Phase 5: Foundation Lessons (Days 14–18)
- ε-Greedy, UCB1, Thompson Sampling
- Each with 5-stage theory card and beginner controls
- 9 tests

**MVP Total: 18 days, ~34 tests, ~8 conventional commits**

### Expansion Phase 6: Contextual Lessons (Days 19–25)
- LinUCB, LinTS, Softmax, Logistic
- Add context feature visualizations and routed-cluster debug snapshots
- 12 tests

### Expansion Phase 7: Advanced Discrete Lessons (Days 26–36)
- LinUCB-SW, GP-UCB, Bootstrapped, LinUCB Hybrid, Neural Linear, Tree Ensembles
- Add algorithm-specific debugger renderers only after adapter snapshots are stable
- 18 tests

### Expansion Phase 8: Continuous + Production Features (Days 37–45)
- CATS via `ContinuousSimulationSession` (not `ClusterBandit`)
- Drift Detection and Offline Evaluation lessons
- 10 tests

### Expansion Phase 9: Comparison + Sandbox (Days 46–50)
- Side-by-side policy comparison and free-form experiment arena
- Export traces as CSV, save/load scenarios
- 6 tests

### Expansion Phase 10: Polish + Docs (Days 51–55)
- Animation audit, responsive check, onboarding tutorial, final docs

**Full Expansion Total: 55 days, ~80 tests, ~22 conventional commits**

---

## 12. Testing Strategy

| Layer | Count | Example |
|-------|-------|---------|
| Unit (engine/state) | ~25 | `test_session_step_returns_valid_record` |
| Unit (narrative) | ~8 | `test_world_configs_all_valid` |
| Component (UI) | ~20 | `test_treatment_card_selection_animation` |
| Integration (lessons) | ~25 | `test_ucb1_lesson_full_flow` |
| End-to-end | ~3 | `test_home_to_lesson_to_reset_and_back` |

```python
# Example lesson integration test
def test_ucb1_lesson_with_clinic_world(store, page):
    """Full UCB1 lesson: configure → step → verify charts update."""
    store.dispatch(select_world("clinic"))
    store.dispatch(select_policy("ucb1"))
    store.dispatch(update_config("alpha", 2.0))

    lesson = UCB1Lesson(store, page)
    assert lesson.theory_content.key_insight != ""

    # Run 10 steps
    store.dispatch(run_n(10))
    assert len(store.trace) == 10
    assert store.stats["cumulative_reward"] >= 0
    assert store.charts.reward_data_points == 10
```

---

## 13. Dependencies & Configuration

### 13.1 Library Standards

| Concern | Choice | Rationale |
|---------|--------|-----------|
| UI | `flet` | Best fit for the requested Python-first desktop/web UI without a separate JS frontend. |
| Numeric ML | `numpy`, `scikit-learn` | Already used by the engine; keep as the source of truth for policy math. |
| Data validation | `pydantic` v2 | Strong schemas for config, trace records, uploaded offline-eval rows, and saved scenarios. |
| Logging | `loguru` | Preferred project logging library; already used throughout `src/coba`. Use it in `coba_web` too instead of stdlib `logging`. |
| Time/dates | `pendulum` | Preferred for timezone-aware timestamps, trace exports, run timers, scenario metadata, and user-facing durations. Avoid naive `datetime.now()`. |
| Persistence | `joblib` | Existing model persistence dependency; keep for ML artifacts. |
| CSV/export | stdlib `csv` for MVP; consider `polars` later | Avoid adding a dataframe dependency until offline-eval uploads need larger/faster tabular processing. |
| JSON | stdlib/Pydantic serialization for MVP; consider `orjson` only if profiling shows bottlenecks | Keeps dependency surface small. |

**Conventions:**
- Use `from loguru import logger` for app logging.
- Use `pendulum.now("UTC")` for stored timestamps and convert to local display time only in UI rendering.
- Keep `datetime` only at boundaries where third-party libraries require native `datetime` objects.
- Do not add heavy visualization/dataframe libraries to MVP unless Flet-native charts or stdlib CSV become a measured limitation.

### 13.2 Project Configuration

```toml
[project]
name = "coba"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "pydantic>=2.0",
    "loguru>=0.7",
    "pendulum>=3.0",
    "joblib>=1.3",
    "flet>=0.24.0",
]

[project.scripts]
coba-web = "coba_web.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-p no:asyncio"
pythonpath = ["src", "."]
```

```bash
# Launch
uv run coba-web

# Test
uv run pytest tests/coba_web/ -v

# Lint
uv run ruff check src/coba_web/ && uv run mypy src/coba_web/
```

---

## Summary Table

| Concern | Original Plan | Revised Plan |
|---------|--------------|--------------|
| **Narratives** | 1 (Doctor) | 7 worlds: Clinic, MovieMatch, NewsFeed, ShopSmart, RidePilot, GameBot, LabTrial |
| **Algorithms covered** | 7 of 17 | All 17 + drift detection + offline eval + cluster routing |
| **Theory pedagogy** | Single `theory_md` string | 5-stage card: Intuition → Visual → Formula → Controls → Metrics |
| **Config tooltips** | None | 30+ `ParamTooltip` entries with increase/decrease effects + beginner tips |
| **Lessons** | 9 | 17 lessons + sandbox + comparison mode |
| **Estimated tests** | ~58 | MVP ~34 / Full ~80 |
| **Timeline** | 28 days | MVP 18 days / Full 55 days |
