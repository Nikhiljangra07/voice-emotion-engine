"""The 'three waves' — independent V/A/D regressors (Layer 2).

Three independent regressors predict valence, arousal, and dominance from the
111 acoustic features (TRAJECTORY_ENGINE.md §2). Design laws enforced here:

  - Law 1: the (V,A,D) output is kept as a TRIPLE/point — never a scalar.
  - Law 10: the FeatureNormalizer is fit on TRAIN ONLY (no leakage).
  - Law 11: no deep learning — SVR / RandomForest / Ridge only.
  - Law 2: evaluation is by CCC (see metrics.dimensional_report).

The model is the swappable Layer 2; the 111-feature engine (Layer 1) is the
constant. Expected reality (Law 7): arousal/dominance CCC will beat valence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from src.dimensional.metrics import DIMENSIONS, dimensional_report
from src.features.feature_vector import feature_names as default_feature_names
from src.features.normalize import FeatureNormalizer

_MODEL_KINDS = ("svr", "rf", "ridge")


def _make_model(kind: str) -> Any:
    """Build one regressor of the given kind (defaults chosen for MVP)."""
    if kind == "svr":
        return SVR(kernel="rbf", C=1.0, gamma="scale")
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    if kind == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unknown model kind '{kind}'. Use one of {_MODEL_KINDS}.")


class DimensionalRegressor:
    """Three independent regressors mapping 111 features -> (V, A, D).

    Workflow:
        reg = DimensionalRegressor(model="ridge")
        reg.fit(X_train, Y_train)          # Y columns = valence, arousal, dominance
        Y_hat = reg.predict(X_test)        # (n, 3)
        report = reg.evaluate(X_test, Y_test)
        reg.save("models/dim_xxx"); DimensionalRegressor.load("models/dim_xxx")
    """

    def __init__(self, model: str = "svr", calibrate: bool = False) -> None:
        if model not in _MODEL_KINDS:
            raise ValueError(f"model must be one of {_MODEL_KINDS}, got '{model}'.")
        self.model_kind = model
        # Output calibration: a leakage-free affine (fit on TRAIN predictions vs
        # TRAIN targets) that rescales predictions to match the target mean/std.
        # Given a fixed correlation this is the CCC-maximising transform — it
        # fixes regressors that hedge toward the mean (RF/Ridge on weak axes).
        # Evidence: lifted MSP-Dev valence CCC 0.110->0.138 (RF), 0.180->0.228
        # (SVR). Off by default to preserve baseline behaviour; opt-in.
        self.calibrate = calibrate
        self.normalizer = FeatureNormalizer()
        self.models: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self._calib: dict[str, tuple[float, float]] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.models) and self.normalizer.is_fitted

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "DimensionalRegressor":
        """Fit one regressor per dimension on train data.

        Args:
            X: Feature matrix (n_samples, n_features).
            Y: Targets (n_samples, 3), columns = valence, arousal, dominance.
            feature_names: Names for X columns. Defaults to the engine's 111.

        Returns:
            self.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or Y.shape[1] != len(DIMENSIONS):
            raise ValueError(f"Y must be (n, {len(DIMENSIONS)}), got {Y.shape}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}.")

        self.feature_names = list(feature_names) if feature_names is not None \
            else default_feature_names()
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but {len(self.feature_names)} "
                "feature names. Pass matching feature_names."
            )

        # No leakage: normalizer learns selection + scaling from TRAIN ONLY.
        Xn = self.normalizer.fit_transform(X, self.feature_names)

        self.models = {}
        self._calib = {}
        for i, dim in enumerate(DIMENSIONS):
            model = _make_model(self.model_kind)
            model.fit(Xn, Y[:, i])
            self.models[dim] = model
            if self.calibrate:
                p = model.predict(Xn)  # TRAIN predictions — no leakage
                a = float(Y[:, i].std() / (p.std() + 1e-9))
                b = float(Y[:, i].mean() - a * p.mean())
                self._calib[dim] = (a, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (n, 3) V/A/D for a feature matrix (calibrated if enabled)."""
        if not self.is_fitted:
            raise RuntimeError("DimensionalRegressor is not fitted.")
        X = np.asarray(X, dtype=float)
        Xn = self.normalizer.transform(X)
        cols = []
        for dim in DIMENSIONS:
            p = self.models[dim].predict(Xn)
            if self.calibrate and dim in self._calib:
                a, b = self._calib[dim]
                p = a * p + b
            cols.append(p)
        return np.column_stack(cols)

    def predict_point(self, x: np.ndarray) -> dict[str, float]:
        """Predict a single (V, A, D) point from one 1-D feature vector."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        v = self.predict(x)[0]
        return {dim: float(v[i]) for i, dim in enumerate(DIMENSIONS)}

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict[str, dict[str, float]]:
        """Per-dimension CCC/RMSE/Pearson + mean CCC on held-out data."""
        Y = np.asarray(Y, dtype=float)
        return dimensional_report(Y, self.predict(X))

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save: not fitted.")
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer.save(save_dir / "normalizer")
        for dim in DIMENSIONS:
            joblib.dump(self.models[dim], save_dir / f"model_{dim}.joblib")
        meta = {
            "model_kind": self.model_kind,
            "dimensions": list(DIMENSIONS),
            "feature_names": self.feature_names,
            "calibrate": self.calibrate,
            "calib": {d: list(self._calib[d]) for d in self._calib},
        }
        with open(save_dir / "dimensional_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DimensionalRegressor":
        load_dir = Path(path)
        with open(load_dir / "dimensional_meta.json") as f:
            meta = json.load(f)
        obj = cls(model=meta["model_kind"], calibrate=meta.get("calibrate", False))
        obj.feature_names = meta["feature_names"]
        obj._calib = {d: tuple(v) for d, v in meta.get("calib", {}).items()}
        obj.normalizer = FeatureNormalizer.load(load_dir / "normalizer")
        obj.models = {
            dim: joblib.load(load_dir / f"model_{dim}.joblib")
            for dim in meta["dimensions"]
        }
        return obj
