"""Feature normalization and selection.

Provides StandardScaler-based normalization and data-driven feature
selection (remove near-zero variance and highly correlated features).

All thresholds are learned from the training data — no magic numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    """Normalize and select features for the classifier.

    Workflow:
        1. ``fit(X, feature_names)`` on training data.
        2. ``transform(X)`` on train/test data.
        3. ``save(path)`` / ``load(path)`` to persist.

    The fit step:
        - Removes near-zero variance features (var < ``var_threshold``).
        - Removes one feature from each highly correlated pair
          (|r| > ``corr_threshold``).
        - Fits a StandardScaler on the remaining features.

    The transform step:
        - Applies the same column selection and scaling.
    """

    def __init__(
        self,
        var_threshold: float = 1e-7,
        corr_threshold: float = 0.95,
    ) -> None:
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold

        # Set during fit:
        self._scaler: StandardScaler | None = None
        self._input_names: list[str] = []
        self._kept_indices: list[int] = []
        self._kept_names: list[str] = []
        self._removed_low_var: list[str] = []
        self._removed_correlated: list[tuple[str, str, float]] = []

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None

    @property
    def kept_names(self) -> list[str]:
        """Feature names that survived selection."""
        return list(self._kept_names)

    @property
    def n_features_in(self) -> int:
        """Number of features before selection."""
        return len(self._input_names)

    @property
    def n_features_out(self) -> int:
        """Number of features after selection."""
        return len(self._kept_names)

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> "FeatureNormalizer":
        """Learn normalization parameters and feature selection from data.

        Args:
            X: Feature matrix (n_samples, n_features).
            feature_names: Ordered list of feature names matching columns.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If X and feature_names have mismatched dimensions.
        """
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but got "
                f"{len(feature_names)} feature names."
            )

        self._input_names = list(feature_names)
        n_features = X.shape[1]
        keep_mask = np.ones(n_features, dtype=bool)

        # ── Step 1: Remove near-zero variance ────────────────────
        variances = np.var(X, axis=0)
        self._removed_low_var = []
        for i in range(n_features):
            if variances[i] < self.var_threshold:
                keep_mask[i] = False
                self._removed_low_var.append(feature_names[i])

        # ── Step 2: Remove highly correlated features ────────────
        # Work only with features that passed variance check.
        remaining_indices = np.where(keep_mask)[0]
        X_remaining = X[:, remaining_indices]

        if X_remaining.shape[1] > 1:
            corr = np.corrcoef(X_remaining.T)
            # Replace NaN correlations (constant after variance filter) with 0.
            corr = np.nan_to_num(corr, nan=0.0)

            to_drop: set[int] = set()
            self._removed_correlated = []

            for i in range(len(remaining_indices)):
                if i in to_drop:
                    continue
                for j in range(i + 1, len(remaining_indices)):
                    if j in to_drop:
                        continue
                    if abs(corr[i, j]) > self.corr_threshold:
                        # Drop the feature with lower mean absolute
                        # correlation to all other features (keeps
                        # the more broadly informative one).
                        mean_corr_i = np.mean(np.abs(corr[i, :]))
                        mean_corr_j = np.mean(np.abs(corr[j, :]))
                        drop_idx = j if mean_corr_j >= mean_corr_i else i
                        to_drop.add(drop_idx)
                        self._removed_correlated.append((
                            feature_names[remaining_indices[i]],
                            feature_names[remaining_indices[j]],
                            float(corr[i, j]),
                        ))

            # Apply correlation drops.
            for local_idx in to_drop:
                global_idx = remaining_indices[local_idx]
                keep_mask[global_idx] = False

        # ── Step 3: Fit StandardScaler on kept features ──────────
        self._kept_indices = list(np.where(keep_mask)[0])
        self._kept_names = [feature_names[i] for i in self._kept_indices]

        X_selected = X[:, self._kept_indices]
        self._scaler = StandardScaler()
        self._scaler.fit(X_selected)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection and normalization.

        Args:
            X: Feature matrix (n_samples, n_features_in).

        Returns:
            Normalized feature matrix (n_samples, n_features_out).

        Raises:
            RuntimeError: If not fitted.
            ValueError: If X has wrong number of columns.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer is not fitted.")
        if X.shape[1] != len(self._input_names):
            raise ValueError(
                f"Expected {len(self._input_names)} columns, "
                f"got {X.shape[1]}."
            )

        X_selected = X[:, self._kept_indices]
        return self._scaler.transform(X_selected)  # type: ignore[union-attr]

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X, feature_names)
        return self.transform(X)

    def summary(self) -> str:
        """Return a human-readable summary of what was done."""
        if not self.is_fitted:
            return "Not fitted."

        lines = [
            f"Input features:  {self.n_features_in}",
            f"Output features: {self.n_features_out}",
            f"Removed (low variance < {self.var_threshold}): "
            f"{len(self._removed_low_var)}",
        ]
        for name in self._removed_low_var:
            lines.append(f"  - {name}")

        lines.append(
            f"Removed (|correlation| > {self.corr_threshold}): "
            f"{len(self._removed_correlated)}"
        )
        for n1, n2, r in self._removed_correlated:
            lines.append(f"  - {n1} <-> {n2} (r={r:.3f})")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save normalizer state to a JSON file + scaler .npy files.

        Args:
            path: Directory to save into (will be created).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save: not fitted.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON.
        meta: dict[str, Any] = {
            "var_threshold": self.var_threshold,
            "corr_threshold": self.corr_threshold,
            "input_names": self._input_names,
            "kept_indices": [int(i) for i in self._kept_indices],
            "kept_names": self._kept_names,
            "removed_low_var": self._removed_low_var,
            "removed_correlated": [
                {"a": a, "b": b, "r": r}
                for a, b, r in self._removed_correlated
            ],
        }
        with open(save_dir / "normalizer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save scaler parameters.
        assert self._scaler is not None
        np.save(save_dir / "scaler_mean.npy", self._scaler.mean_)
        np.save(save_dir / "scaler_scale.npy", self._scaler.scale_)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Load a saved normalizer.

        Args:
            path: Directory containing saved files.

        Returns:
            Loaded FeatureNormalizer instance.
        """
        load_dir = Path(path)

        with open(load_dir / "normalizer_meta.json") as f:
            meta = json.load(f)

        obj = cls(
            var_threshold=meta["var_threshold"],
            corr_threshold=meta["corr_threshold"],
        )
        obj._input_names = meta["input_names"]
        obj._kept_indices = meta["kept_indices"]
        obj._kept_names = meta["kept_names"]
        obj._removed_low_var = meta["removed_low_var"]
        obj._removed_correlated = [
            (d["a"], d["b"], d["r"]) for d in meta["removed_correlated"]
        ]

        scaler = StandardScaler()
        scaler.mean_ = np.load(load_dir / "scaler_mean.npy")
        scaler.scale_ = np.load(load_dir / "scaler_scale.npy")
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.n_samples_seen_ = 1  # placeholder
        obj._scaler = scaler

        return obj
