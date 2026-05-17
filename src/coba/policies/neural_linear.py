"""
NeuralLinear arm model.

Reference:
  Riquelme et al., "Deep Bayesian Bandits Showdown", ICLR 2018.

Architecture:
  A shared MLP backbone extracts a low-dimensional embedding from the raw context.
  Each arm maintains a per-arm LinTS (ridge regression + Thompson Sampling) on top
  of the embedding. The backbone is periodically retrained on buffered observations;
  the per-arm linear heads are rebuilt from the buffer after each backbone retrain.

This gives non-linear feature extraction (backbone) with O(d²) online updates
(per-arm LinTS on embedding), without requiring PyTorch or GPU.

Backbone training schedule:
  The backbone accumulates all observations in a shared buffer. Every
  retrain_freq updates, it refits the MLP on the full buffer (all arms combined,
  using reward as the regression target). After each retrain the backbone
  increments its generation counter and all arm models rebuild their LinTS models
  by replaying the buffer through the new embedding.

Usage (via ClusterBandit):
  bandit = ClusterBandit(
      arms=[...],
      n_features=10,
      policy=PolicyType.NEURAL_LINEAR,
      neural_embedding_dim=16,
      neural_hidden_sizes=(64, 32),
      neural_retrain_freq=100,
  )
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.neural_network import MLPRegressor

from coba.policies.base import BaseArmModel
from coba.policies.lin_ts import LinTSArmModel
from coba.types import Arm

# Maximum buffer size — older observations are dropped to bound memory and
# keep the backbone focused on recent reward distributions.
_DEFAULT_BUFFER_MAXLEN = 10_000


class NeuralLinearBackbone:
    """Shared MLP backbone for all arms in one cluster.

    Maintains a replay buffer and periodically retrains the MLP.
    Exposes the penultimate-layer activations as the arm embedding.

    Args:
        n_features: Input context dimensionality.
        embedding_dim: Dimensionality of the penultimate layer (= LinTS input size).
        hidden_sizes: Tuple of hidden layer widths. Last entry becomes embedding_dim.
        retrain_freq: Number of total updates before triggering a backbone retrain.
        l2_lambda: L2 penalty passed to MLPRegressor (alpha parameter).
        seed: Random seed for reproducibility.
        buffer_maxlen: Maximum replay buffer size (FIFO drop when exceeded).
    """

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 16,
        hidden_sizes: tuple[int, ...] = (64, 32),
        retrain_freq: int = 10,
        l2_lambda: float = 1e-4,
        seed: int = 42,
        buffer_maxlen: int = _DEFAULT_BUFFER_MAXLEN,
    ) -> None:
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.retrain_freq = retrain_freq

        # Build hidden layer spec: user-provided layers + embedding_dim output layer
        layer_sizes = tuple(hidden_sizes) + (embedding_dim,)
        self._mlp = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            activation="relu",
            alpha=l2_lambda,
            max_iter=200,
            random_state=seed,
            warm_start=True,
        )

        # Replay buffer: tuples of (x, arm, reward, weight)
        self._buffer: deque[tuple[np.ndarray, Arm, float, float]] = deque(maxlen=buffer_maxlen)

        self._n_updates: int = 0
        self._generation: int = 0  # incremented on every backbone retrain
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, x: np.ndarray, arm: Arm, reward: float, weight: float = 1.0) -> bool:
        """Add an observation to the buffer and trigger retrain if due.

        Args:
            x: Raw context vector.
            arm: Chosen arm.
            reward: Observed reward.
            weight: IPS importance weight.

        Returns:
            True if a backbone retrain was triggered by this update.
        """
        self._buffer.append((x, arm, reward, weight))
        self._n_updates += 1

        if self._n_updates % self.retrain_freq == 0 and len(self._buffer) >= 3:
            self._retrain()
            return True
        return False

    def get_embedding(self, x: np.ndarray) -> np.ndarray | None:
        """Return the penultimate-layer activation for context x.

        Returns None if the backbone is not yet fitted (cold start).
        Callers must handle None by falling back to raw context or cold-start logic.

        Args:
            x: Raw context vector, shape (n_features,).
        Returns:
            Embedding vector, shape (embedding_dim,), or None if not fitted.
        """
        if not self._is_fitted:
            return None
        return self._forward_to_penultimate(x.reshape(1, -1))[0]

    @property
    def generation(self) -> int:
        """Increments each time the backbone is retrained."""
        return self._generation

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_buffer_for_arm(self, arm: Arm) -> list[tuple[np.ndarray, float, float]]:
        """Return all buffered (x, reward, weight) observations for a specific arm."""
        return [(x, r, w) for x, a, r, w in self._buffer if a == arm]

    def reset(self) -> None:
        """Reset backbone and buffer (full refit from scratch)."""
        self._buffer.clear()
        self._n_updates = 0
        self._generation = 0
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _retrain(self) -> None:
        """Refit the MLP on the full replay buffer."""
        if len(self._buffer) < 3:
            return
        x_mat = np.stack([x for x, _, _, _ in self._buffer])
        y = np.array([r for _, _, r, _ in self._buffer])
        self._mlp.fit(x_mat, y)
        self._is_fitted = True
        self._generation += 1

    def _forward_to_penultimate(self, x_mat: np.ndarray) -> np.ndarray:
        """Manually compute the forward pass up to (and including) the penultimate layer.

        sklearn's MLPRegressor stores all weights in coefs_/intercepts_. We apply
        every layer except the final output layer to get the embedding.
        """
        activations = x_mat.astype(np.float64)
        # All hidden layers (not the final output projection)
        for w, b in zip(self._mlp.coefs_[:-1], self._mlp.intercepts_[:-1]):
            activations = np.maximum(0.0, activations @ w + b)  # ReLU
        return activations


class NeuralLinearArmModel(BaseArmModel):
    """Per-arm NeuralLinear model: LinTS on top of a shared MLP embedding.

    Args:
        arm: Arm identifier.
        backbone: Shared backbone for this cluster. Updated by all arms.
        v_sq: Posterior variance scale for Thompson Sampling.
        l2_lambda: L2 regularisation for the per-arm LinTS.
        gamma: Discount factor for non-stationarity.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        backbone: NeuralLinearBackbone,
        v_sq: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self._backbone = backbone
        self.v_sq = v_sq
        self.l2_lambda = l2_lambda
        self.gamma = gamma

        self._lin_ts = LinTSArmModel(
            arm,
            n_features=backbone.embedding_dim,
            v_sq=v_sq,
            l2_lambda=l2_lambda,
            gamma=gamma,
            rng=self.rng,
        )
        # Generation of the backbone when this arm's LinTS was last rebuilt
        self._synced_generation: int = -1

    def score(self, x: np.ndarray) -> float:
        """LinTS score on the backbone embedding.

        Falls back to Thompson sampling on unexplored arms (score=inf) during
        backbone cold start, ensuring all arms are explored at least once before
        the backbone is ready. This is critical for neural_linear which needs
        diverse arm data before it can meaningfully fit.
        """
        self._sync_if_needed()

        embedding = self._backbone.get_embedding(x)
        if embedding is None:
            # Backbone not ready — return inf to trigger exploration of this arm
            # The router will select arms in order: highest scores win, inf guarantees
            # all arms get tried (UCB-style exploration of untrained arms)
            return float("inf")
        return self._lin_ts.score(embedding)

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update backbone buffer and per-arm LinTS.

        If the backbone retrains as a result of this update, the per-arm LinTS
        is rebuilt from the buffered observations using the new embeddings.
        """
        retrained = self._backbone.add(x, self.arm, reward, weight)

        embedding = self._backbone.get_embedding(x)
        if embedding is not None and not retrained:
            # Backbone unchanged — incremental online update
            self._lin_ts.update(embedding, reward, weight)

        if retrained:
            # Backbone changed — rebuild LinTS from buffer with new embeddings
            self._rebuild_from_buffer()

        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        ws = np.ones(len(y), dtype=np.float64) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self.update(xi, float(yi), float(wi))

    def reset(self) -> None:
        """Reset per-arm LinTS (backbone is owned by the cluster, not reset here)."""
        self._lin_ts.reset()
        self._synced_generation = -1
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sync_if_needed(self) -> None:
        """Rebuild LinTS if the backbone was retrained since last sync."""
        if self._backbone.generation != self._synced_generation:
            self._rebuild_from_buffer()

    def _rebuild_from_buffer(self) -> None:
        """Refit per-arm LinTS from buffered data using the new backbone embeddings."""
        self._lin_ts.reset()
        arm_data = self._backbone.get_buffer_for_arm(self.arm)
        for x, reward, weight in arm_data:
            embedding = self._backbone.get_embedding(x)
            if embedding is not None:
                self._lin_ts.update(embedding, reward, weight)
        self._synced_generation = self._backbone.generation
        if arm_data:
            self.is_fitted = True
