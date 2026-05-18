# Policies Reference

All 17 policy types supported by COBA Web, mapped to lessons and algorithms.

---

## Context-Free Policies

These policies ignore context and optimize using only arm reward statistics.

### 1. Epsilon-Greedy
**Lesson:** Lesson 0 (Intro: Explore vs Exploit)
**Type:** Deterministic
**Intuition:** Pick the best arm with probability `1-ε`, explore randomly with probability `ε`
**When to use:** Simplest baseline, predictable exploration rate
**Reference:** [docs/algorithms/context_free.md](./algorithms/context_free.md)

### 2. UCB1 (Upper Confidence Bound)
**Lesson:** Lesson 1 (UCB1 Landing Page Testing)
**Type:** Deterministic
**Intuition:** Add an exploration bonus that shrinks as an arm is pulled more
**When to use:** Smooth exploration decay, optimal regret bounds for context-free setting
**Formula:** `score = mean_reward + alpha * sqrt(ln(N) / n_pulls)`
**Reference:** [docs/algorithms/context_free.md](./algorithms/context_free.md)

### 3. Thompson Sampling
**Lesson:** Lesson 2 (Thompson Sampling Email Subject Lines)
**Type:** Stochastic (Bayesian)
**Intuition:** Sample from Beta posterior per arm, pick arm with highest sample
**When to use:** Binary outcomes (conversion), good empirical performance
**Constraint:** Rewards must be in [0, 1]
**Reference:** [docs/algorithms/context_free.md](./algorithms/context_free.md)

### 4. Softmax
**Lesson:** Lesson 11 (Softmax Playlist Generation)
**Type:** Stochastic
**Intuition:** Sample arm with probability proportional to `exp(score / tau)`; temperature `tau` controls exploration
**When to use:** Smooth stochastic exploration, tunable via temperature
**Reference:** See lesson implementation

---

## Linear Contextual Policies

These policies learn a linear model per arm, using context features.

### 5. LinUCB (Linear Upper Confidence Bound)
**Lesson:** Lesson 3 (LinUCB Product Recommendation)
**Type:** Deterministic
**Intuition:** Fit a ridge regression per arm; add exploration bonus proportional to uncertainty
**When to use:** Contextual bandits, linear reward surfaces, interpretable feature weights
**Formula:** `score = x^T beta + alpha * sqrt(x^T A^-1 x)`
**Reference:** [docs/algorithms/linucb.md](./algorithms/linucb.md)

### 6. Linear Thompson Sampling (LinTS)
**Lesson:** Lesson 4 (LinTS Loan Offer Personalisation)
**Type:** Stochastic (Bayesian)
**Intuition:** Maintain Bayesian posterior over arm coefficients, sample and pick best
**When to use:** Contextual, delayed feedback, batch updates
**Reference:** [docs/algorithms/lin_ts.md](./algorithms/lin_ts.md)

### 7. Logistic Bandits (UCB)
**Lesson:** Lesson 5 (Logistic Bandits for Ad CTR)
**Type:** Deterministic
**Intuition:** Model binary outcomes (0/1) with logistic regression per arm; UCB exploration
**When to use:** Click-through rates, conversions, binary rewards
**Constraint:** Rewards must be 0 or 1
**Reference:** [docs/algorithms/logistic.md](./algorithms/logistic.md)

### 8. Logistic Thompson Sampling (LogisticTS)
**Lesson:** Not featured (available policy)
**Type:** Stochastic
**Intuition:** Bayesian logistic regression; sample and pick best arm
**When to use:** Binary outcomes with Bayesian uncertainty quantification
**Constraint:** Rewards must be 0 or 1

---

## Contextual Non-Linear Policies

### 9. LinUCB-Hybrid
**Lesson:** Lesson 7 (LinUCB-Hybrid News Personalisation)
**Type:** Deterministic
**Intuition:** Split context into shared features (learned jointly) + arm-specific features
**When to use:** Shared user/session features that generalize across arms
**Reference:** [docs/algorithms/lin_ucb_hybrid.md](./algorithms/lin_ucb_hybrid.md)

### 10. Cluster Routing
**Lesson:** Lesson 6 (KMeans Cluster Routing Music Demo)
**Type:** Deterministic (clustered)
**Intuition:** Partition context space via K-means; maintain independent LinUCB per cluster
**When to use:** Non-linear/non-stationary environments, heterogeneous user segments
**Reference:** [docs/algorithms/cluster_router.md](./algorithms/cluster_router.md)

### 11. Neural Linear
**Lesson:** Lesson 8 (NeuralLinear Video Recommendation)
**Type:** Hybrid (Deep + Linear)
**Intuition:** Shared MLP backbone extracts non-linear embeddings; per-arm LinTS heads learn on top
**When to use:** Complex non-linear reward surfaces, don't want full deep RL overhead
**Reference:** [docs/algorithms/neural_linear.md](./algorithms/neural_linear.md)

### 12. Random Forest Meta-Learner (UCB)
**Lesson:** Lesson 9 (Random Forest Dynamic Pricing)
**Type:** Ensemble (Tree-based)
**Intuition:** Fit Random Forest regressor per arm; use tree disagreement as uncertainty bonus
**When to use:** Non-linear rewards, robust to outliers, feature interactions
**Reference:** [docs/algorithms/sklearn_meta.md](./algorithms/sklearn_meta.md)

### 13. Random Forest Thompson Sampling
**Lesson:** Not featured (available policy)
**Type:** Ensemble (Stochastic)
**Intuition:** Bootstrap trees per arm; sample from ensemble predictions
**When to use:** Binary outcomes with tree-based non-linearity

### 14. Gaussian Process UCB
**Lesson:** Lesson 10 (GP-UCB Clinical Trial)
**Type:** Probabilistic (Gaussian Process)
**Intuition:** Maintain full GP posterior per arm; UCB leverages posterior variance as uncertainty
**When to use:** Low-volume decisions, complex non-linear surfaces, need uncertainty quantification
**Trade-off:** O(n²) inference — not suitable for high-throughput
**Reference:** [docs/algorithms/gp_ucb.md](./algorithms/gp_ucb.md)

---

## Time-Adaptive & Special Policies

### 15. Sliding-Window LinUCB (LinUCB-SW)
**Lesson:** Lesson 12 (Sliding-Window LinUCB Flash Sale)
**Type:** Deterministic (Windowed)
**Intuition:** LinUCB with a sliding window of recent observations; old data is discarded
**When to use:** Rapidly changing reward distributions (flash sales, seasonal shifts)
**Reference:** See lesson implementation

### 16. Drift Detection (Page-Hinkley + LinUCB)
**Lesson:** Lesson 13 (PageHinkley Drift Detection)
**Type:** Adaptive
**Intuition:** Monitor for distributional shift via Page-Hinkley test; reset arms on drift detection
**When to use:** Non-stationary environments, detect and adapt to concept drift
**Reference:** [docs/advanced_features.md](./advanced_features.md#6-pagehinkleydetector--reward-drift-detection)

### 17. Continuous Action (CATS)
**Lesson:** Lesson 15 (CATS Real-Time Bidding)
**Type:** Tree-based Continuous
**Intuition:** Binary tree partitions continuous action space; per-leaf Thompson Sampling
**When to use:** Real-time bidding, continuous auction prices, parameter tuning
**Arms:** None (continuous range instead)
**Reference:** See lesson implementation

---

## Bootstrapped Policies (Advanced)

### Bootstrapped UCB
**Lesson:** Not featured (available policy)
**Type:** Ensemble (Bootstrapped)
**Intuition:** Train multiple models via bootstrap sampling; pick arm with highest ensemble mean + uncertainty
**When to use:** Robust uncertainty quantification, non-linear via sklearn base estimators

### Bootstrapped Thompson Sampling
**Lesson:** Not featured (available policy)
**Type:** Ensemble (Stochastic)
**Intuition:** Bootstrap ensemble; sample one model, pick best arm per sample
**When to use:** Binary/sparse outcomes with ensemble diversity

---

## Off-Policy Learning

### Inverse Propensity Scoring (IPS)
**Lesson:** Lesson 14 (Offline Evaluation IPS/DR/NCIS)
**Type:** Evaluation method
**Intuition:** Reweight historical rewards by `π(a|x) / p(a|x)` to correct logging policy bias
**When to use:** Evaluating policies on logged data, bootstrapping from biased history
**Reference:** [docs/algorithms/offpolicy_ips.md](./algorithms/offpolicy_ips.md)

### Doubly Robust (DR)
**Lesson:** Lesson 14 (Offline Evaluation IPS/DR/NCIS)
**Type:** Evaluation method (Hybrid)
**Intuition:** Combines reward model (direct method) + IPS; unbiased if either is accurate
**When to use:** Lower variance evaluation than IPS alone, requires reward model
**Reference:** [docs/evaluation.md](./evaluation.md)

### Normalized Capped Importance Sampling (NCIS)
**Lesson:** Lesson 14 (Offline Evaluation IPS/DR/NCIS)
**Type:** Evaluation method
**Intuition:** IPS with capped weights + normalization to prevent variance explosions
**When to use:** Extreme propensity differences (near-deterministic logging policy)
**Reference:** [docs/evaluation.md](./evaluation.md)

---

## Policy Selection Guide

| Scenario | Recommended Policies | Why |
|----------|----------------------|-----|
| **No context** | UCB1, Thompson, Epsilon-Greedy | Context-free optimal |
| **Linear context** | LinUCB, LinTS, Logistic | Fast, interpretable, proven regret bounds |
| **Non-linear context** | Neural Linear, Cluster Routing, Forest | Capture interactions, non-stationary |
| **Binary outcomes** | Logistic, Thompson, Forest | Suit[ed to 0/1 rewards |
| **Drifting rewards** | LinUCB-SW, Drift Detection | Adapt to distribution shifts |
| **Continuous actions** | CATS | Tree-based partitioning of action space |
| **Offline evaluation** | IPS, DR, NCIS | De-bias logged data |
| **Low-volume decisions** | GP-UCB | Full uncertainty, O(n²) okay |
| **High-throughput** | LinUCB, Forest, Neural Linear | O(d²) or O(1) per decision |
| **Binary outcomes** | Logistic, Thompson, Forest | Suited to 0/1 rewards |

---

## Complexity & Performance

| Policy | Time Complexity | Space | Best For |
|--------|-----------------|-------|----------|
| **Epsilon-Greedy** | O(1) | O(n_arms) | Baseline |
| **UCB1** | O(1) | O(n_arms) | Context-free |
| **Thompson** | O(1) | O(n_arms) | Bayesian baseline |
| **LinUCB** | O(d²) | O(d² × n_arms) | Contextual linear |
| **LinTS** | O(d²) | O(d² × n_arms) | Bayesian contextual |
| **Logistic** | O(d) | O(d × n_arms) | Binary outcomes |
| **Neural Linear** | O(hidden²) | O(embedding_dim × hidden × d) | Non-linear + Bayesian |
| **Random Forest** | O(trees × depth) | O(trees × depth × features) | Non-linear ensemble |
| **GP-UCB** | O(n²) | O(n²) | Expensive, low-volume |
| **Cluster Routing** | O(d² × K) | O(d² × K × n_arms) | Non-stationary, partitioned |

---

## Lesson Curriculum Mapped to Policies

| Index | Lesson | Policy | Difficulty |
|-------|--------|--------|------------|
| 0 | Explore vs Exploit | epsilon_greedy | Beginner |
| 1 | UCB1 Landing Page | ucb1 | Beginner |
| 2 | Thompson Sampling Email | thompson | Beginner |
| 3 | LinUCB Product Rec | linucb | Intermediate |
| 4 | LinTS Loan Offer | lints | Intermediate |
| 5 | Logistic Bandits Ad CTR | logistic_ucb | Intermediate |
| 6 | Cluster Routing Music | linucb (clustered) | Intermediate |
| 7 | LinUCB-Hybrid News | linucb_hybrid | Advanced |
| 8 | NeuralLinear Video | neural_linear | Advanced |
| 9 | Random Forest Pricing | random_forest_ucb | Advanced |
| 10 | GP-UCB Clinical Trial | gp_ucb | Advanced |
| 11 | Softmax Playlist | softmax | Intermediate |
| 12 | Sliding-Window Flash Sale | linucb_sw | Advanced |
| 13 | PageHinkley Drift Detection | linucb (drift-aware) | Advanced |
| 14 | Offline Eval IPS/DR/NCIS | linucb | Advanced |
| 15 | CATS Real-Time Bidding | cats | Advanced |
| 16 | Production Constraints and Abstention | linucb | Advanced |

---

## Questions?

See individual algorithm references (in `/algorithms/` directory) for mathematical details, usage examples, and hyperparameter guidance.
