# Policies Reference

The `coba` library provides a comprehensive set of Multi-Armed Bandit (MAB) algorithms, from context-free baselines to advanced non-linear meta-heuristics.

Detailed intuition, mathematical foundations, and code examples for each policy are in their dedicated reference pages below.

## Contextual Bandits

These algorithms analyze a feature vector at each decision step to predict the optimal arm.

1. **[LinUCB (Linear Upper Confidence Bound)](algorithms/linucb.md)**
   A deterministic algorithm that adds an exploration bonus proportional to uncertainty in unexplored context regions.

2. **[LinTS (Linear Thompson Sampling)](algorithms/lin_ts.md)**
   A stochastic Bayesian algorithm. Often the best choice for systems with delayed feedback or batch updates.

3. **[Logistic Bandits (Laplace Approximation)](algorithms/logistic.md)**
   Designed for binary reward environments. Use when the reward is a 0/1 outcome (e.g., conversion, click).

4. **[Meta-Heuristics (Scikit-learn Wrappers)](algorithms/sklearn_meta.md)**
   Wraps any scikit-learn estimator (e.g., LightGBM, Random Forest) in Bootstrapped TS/UCB or Epsilon-Greedy exploration for non-linear reward surfaces.

5. **[LinUCB-Hybrid](algorithms/lin_ucb_hybrid.md)**
   Splits context into shared features (learned jointly across all arms) and arm-specific features. Shared features converge faster through cross-arm data pooling. Best when user/session features generalize across items.

6. **[NeuralLinear](algorithms/neural_linear.md)**
   A shared MLP backbone extracts non-linear embeddings; per-arm LinTS heads learn on top. Combines deep feature extraction with O(d²) Bayesian updates — no GPU required.

7. **[GP-UCB (Gaussian Process UCB)](algorithms/gp_ucb.md)**
   Maintains a full Gaussian Process posterior per arm. Best for low-volume decisions with complex non-linear reward surfaces. O(n²) inference — not suitable for high-throughput serving.

---

## Context-Free Bandits

These algorithms ignore the context vector and rely purely on aggregate arm statistics. A smarter alternative to traditional A/B testing.

8. **[UCB1 & Thompson Sampling](algorithms/context_free.md)**
   Automatically routes traffic toward the best-performing arm as evidence accumulates.

---

## Advanced Topics

1. **[Cluster Routing](algorithms/cluster_router.md)**
   Handle non-linear and non-stationary environments by splitting the context space into sub-clusters, each managed by an independent bandit. Includes warm-start support for adding/removing arms at runtime.

2. **[Off-Policy Learning (IPS & Doubly-Robust)](algorithms/offpolicy_ips.md)**
   Bootstrap a new bandit from biased historical data collected by a prior policy. Uses Inverse Propensity Scoring to de-bias the logs.

3. **[Advanced Features Guide](advanced_features.md)**
   `decide_top_k`, confidence-based abstention, `min_pull_rates` floor constraints, per-arm gamma overrides, `RewardNormalizer`, and `PageHinkleyDetector` drift detection.

4. **Multi-Objective via Reward Scalarization** — see [Architecture & Patterns](index.md#multi-objective-via-reward-scalarization).
