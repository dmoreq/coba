"""
BanditConfig — single source of truth for all hyper-parameters.

Passing a BanditConfig instead of 25 individual keyword arguments eliminates
the param-waterfall that previously existed across ClusterBandit → ClusterRouter
→ _build_arm_models.  All three layers now accept one config object and
destructure only the fields they need.

Typical usage::

    from coba.config import BanditConfig
    from coba.bandit import ClusterBandit
    from coba.types import PolicyType

    cfg = BanditConfig(
        policy=PolicyType.LIN_UCB,
        n_clusters=5,
        alpha=1.0,
    )
    bandit = ClusterBandit(arms=["a", "b", "c"], n_features=7, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coba.types import PolicyType


@dataclass
class BanditConfig:
    """All hyper-parameters for ClusterBandit and its internal components.

    Attributes:
        policy: Learning algorithm to use per arm per cluster.
        n_clusters: Number of KMeans context clusters.
        alpha: Exploration parameter for LinUCB / UCB1 / Logistic-UCB.
        v_sq: Posterior variance multiplier for LinTS / Logistic-TS / NeuralLinear.
        l2_lambda: L2 regularization for linear and logistic models.
        gamma: Exponential discount factor for non-stationarity (1.0 = stationary).
        seed: Global random seed.
        use_minibatch: Use MiniBatchKMeans instead of KMeans for faster online updates.
        scale_contexts: Apply StandardScaler to contexts before clustering.
        epsilon: Exploration probability for EpsilonGreedy.
        n_bootstraps: Ensemble size for Bootstrapped TS/UCB.
        base_estimator: Scikit-learn compatible estimator for non-linear policies.
        n_shared_features: Shared feature count for LinUCBHybrid.
        neural_embedding_dim: Embedding dimension for NeuralLinear backbone.
        neural_hidden_sizes: Hidden layer widths for NeuralLinear backbone MLP.
        neural_retrain_freq: Backbone retrain frequency (total updates) for NeuralLinear.
        gp_beta: UCB exploration coefficient for GP-UCB.
        gp_length_scale: RBF kernel bandwidth for GP-UCB.
        gp_noise_var: Observation noise variance for GP-UCB.
        gp_max_obs: Maximum stored observations for GP-UCB (FIFO eviction).
        enable_drift_detection: Enable per-arm PageHinkley drift detection.
        drift_delta: Minimum detectable change for PageHinkley.
        drift_lambda: Detection threshold for PageHinkley.
        min_pull_rates: Optional per-arm minimum traffic fraction constraints.
        cats_a_min: Lower bound of CATS action space.
        cats_a_max: Upper bound of CATS action space.
        cats_depth: Binary tree depth for CATS (n_leaves = 2^cats_depth).
        rf_n_estimators: Number of Random Forest trees for tree-ensemble bandits.
        rf_max_depth: Maximum tree depth for Random Forest bandit models.
        rf_min_samples_leaf: Minimum samples per RF leaf.
        rf_max_obs: Maximum observations retained per arm model before FIFO eviction.
        rf_min_uncertainty: Lower bound for ensemble-disagreement uncertainty.
    """

    # --- Core ---
    policy: PolicyType = PolicyType.LIN_UCB
    n_clusters: int = 5

    # --- Linear / Logistic exploration ---
    alpha: float = 1.0
    v_sq: float = 1.0
    l2_lambda: float = 1.0
    gamma: float = 1.0

    # --- Infrastructure ---
    seed: int = 42
    use_minibatch: bool = True
    scale_contexts: bool = True

    # --- Softmax ---
    softmax_tau: float = 1.0

    # --- SlidingWindowLinUCB ---
    linucb_sw_window: int = 200

    # --- EpsilonGreedy ---
    epsilon: float = 0.1

    # --- Bootstrapped ---
    n_bootstraps: int = 10
    base_estimator: Any = field(default=None, repr=False)

    # --- LinUCBHybrid ---
    n_shared_features: int = 0

    # --- NeuralLinear ---
    neural_embedding_dim: int = 16
    neural_hidden_sizes: tuple[int, ...] = (64, 32)
    neural_retrain_freq: int = 200

    # --- GP-UCB ---
    gp_beta: float = 2.0
    gp_length_scale: float = 1.0
    gp_noise_var: float = 0.1
    gp_max_obs: int = 500

    # --- Drift detection ---
    enable_drift_detection: bool = False
    drift_delta: float = 0.005
    drift_lambda: float = 50.0

    # --- Traffic constraints ---
    min_pull_rates: dict | None = field(default=None, repr=False)

    # --- CATS (Continuous Action Tree Sampling) ---
    cats_a_min: float = 0.0
    cats_a_max: float = 1.0
    cats_depth: int = 6  # 2^6 = 64 leaves

    # --- Tree ensemble bandits ---
    rf_n_estimators: int = 50
    rf_max_depth: int | None = 6
    rf_min_samples_leaf: int = 1
    rf_max_obs: int = 1000
    rf_min_uncertainty: float = 1e-6

    def __post_init__(self) -> None:
        """Validate constraints that, if violated, produce cryptic downstream errors."""
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.v_sq <= 0:
            raise ValueError(f"v_sq must be > 0, got {self.v_sq}")
        if self.l2_lambda <= 0:
            raise ValueError(f"l2_lambda must be > 0, got {self.l2_lambda}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if self.n_bootstraps < 2:
            raise ValueError(f"n_bootstraps must be >= 2, got {self.n_bootstraps}")
        if self.softmax_tau <= 0:
            raise ValueError(f"softmax_tau must be > 0, got {self.softmax_tau}")
        if self.neural_embedding_dim < 1:
            raise ValueError(f"neural_embedding_dim must be >= 1, got {self.neural_embedding_dim}")
        if self.gp_beta < 0:
            raise ValueError(f"gp_beta must be >= 0, got {self.gp_beta}")
        if self.gp_length_scale <= 0:
            raise ValueError(f"gp_length_scale must be > 0, got {self.gp_length_scale}")
        if self.gp_noise_var <= 0:
            raise ValueError(f"gp_noise_var must be > 0, got {self.gp_noise_var}")
        if self.gp_max_obs < 1:
            raise ValueError(f"gp_max_obs must be >= 1, got {self.gp_max_obs}")
        if self.drift_delta < 0:
            raise ValueError(f"drift_delta must be >= 0, got {self.drift_delta}")
        if self.drift_lambda <= 0:
            raise ValueError(f"drift_lambda must be > 0, got {self.drift_lambda}")
        if self.rf_n_estimators < 2:
            raise ValueError(f"rf_n_estimators must be >= 2, got {self.rf_n_estimators}")
        if self.rf_min_uncertainty < 0:
            raise ValueError(f"rf_min_uncertainty must be >= 0, got {self.rf_min_uncertainty}")
        if self.cats_depth < 1:
            raise ValueError(f"cats_depth must be >= 1, got {self.cats_depth}")
        if self.linucb_sw_window < 1:
            raise ValueError(f"linucb_sw_window must be >= 1, got {self.linucb_sw_window}")
