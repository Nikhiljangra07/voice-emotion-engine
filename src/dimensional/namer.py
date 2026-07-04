"""Naming layer (Layer 4): a (V,A,D) point -> emotion distribution.

Turns a measured point in PAD space into an emotion *distribution*, grounded
in data. Design laws enforced here:

  - Law 3: emotion positions are DATA centroids learned via ``fit`` — never
    hand-placed. The class holds no emotion coordinates of its own.
  - Law 4: ``predict`` returns a DISTRIBUTION (+ an ``ambiguous`` flag), never
    a forced single label. The nearest emotion is reported for convenience,
    but the distribution is the honest answer.

Distance uses **Mahalanobis** (not Euclidean) so each emotion cluster's shape
and spread are respected — anger/fear/disgust clusters are not equal spheres.
A class with too few samples to estimate a 3x3 covariance falls back to
Euclidean (identity covariance) for that class, logged as a warning.

Intensity = the point's radius from the origin (||V,A,D||) — the Plutchik
"intensity = distance from centre" idea, surfaced alongside the distribution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.dimensional.metrics import DIMENSIONS

logger = logging.getLogger(__name__)

_N_DIMS = len(DIMENSIONS)


class CentroidNamer:
    """Map a (V,A,D) point to an emotion distribution via data centroids.

    Args:
        ambiguity_margin: if the top-1 minus top-2 probability is below this,
            the point is flagged ``ambiguous`` (it sits in a gradient between
            clusters). This is a *reporting* threshold (UX), not an emotion
            coordinate — configurable, documented, not a hidden magic number.
        reg: covariance regularisation (added to the diagonal) for stable
            inversion.
    """

    def __init__(self, ambiguity_margin: float = 0.15, reg: float = 1e-6) -> None:
        self.ambiguity_margin = ambiguity_margin
        self.reg = reg
        self.labels: list[str] = []
        self._centroids: dict[str, np.ndarray] = {}
        self._inv_cov: dict[str, np.ndarray] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._centroids)

    def fit(self, points: np.ndarray, labels: list[str]) -> "CentroidNamer":
        """Learn per-emotion centroid + inverse covariance from labelled points.

        Args:
            points: (n, 3) array of (V, A, D) points.
            labels: length-n list of emotion names.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != _N_DIMS:
            raise ValueError(f"points must be (n, {_N_DIMS}), got {points.shape}.")
        if len(labels) != points.shape[0]:
            raise ValueError("labels length must match number of points.")

        self.labels = sorted(set(labels))
        labels_arr = np.asarray(labels)
        self._centroids = {}
        self._inv_cov = {}

        for emo in self.labels:
            pts = points[labels_arr == emo]
            self._centroids[emo] = pts.mean(axis=0)
            if pts.shape[0] >= _N_DIMS + 1:
                cov = np.cov(pts, rowvar=False) + self.reg * np.eye(_N_DIMS)
                try:
                    self._inv_cov[emo] = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    self._inv_cov[emo] = np.linalg.pinv(cov)
            else:
                # Too few samples for a 3x3 covariance -> Euclidean fallback.
                logger.warning(
                    "Emotion '%s' has %d samples (< %d); using Euclidean "
                    "distance (identity covariance).", emo, pts.shape[0], _N_DIMS + 1
                )
                self._inv_cov[emo] = np.eye(_N_DIMS)
        return self

    def _mahalanobis(self, point: np.ndarray, emo: str) -> float:
        delta = point - self._centroids[emo]
        d2 = float(delta @ self._inv_cov[emo] @ delta)
        return float(np.sqrt(max(d2, 0.0)))

    def predict(self, point: np.ndarray) -> dict[str, object]:
        """Map one (V,A,D) point to an emotion distribution.

        Returns a dict with:
            emotion:      nearest centroid (convenience label)
            distribution: {emotion -> probability}, sums to 1 (Law 4)
            distances:    {emotion -> Mahalanobis distance}
            intensity:    radius ||V,A,D|| from the origin
            ambiguous:    True if top-1/top-2 probability margin is small
        """
        if not self.is_fitted:
            raise RuntimeError("CentroidNamer is not fitted.")
        point = np.asarray(point, dtype=float).ravel()
        if point.size != _N_DIMS:
            raise ValueError(f"point must have {_N_DIMS} values, got {point.size}.")

        distances = {emo: self._mahalanobis(point, emo) for emo in self.labels}

        # Soft assignment: softmax over negative distances (stable form).
        d = np.array([distances[emo] for emo in self.labels])
        logits = -d
        logits -= logits.max()
        w = np.exp(logits)
        probs = w / w.sum()
        distribution = {emo: float(p) for emo, p in zip(self.labels, probs)}

        order = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        top_emotion = order[0][0]
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)

        return {
            "emotion": top_emotion,
            "distribution": distribution,
            "distances": distances,
            "intensity": float(np.linalg.norm(point)),
            "ambiguous": bool(margin < self.ambiguity_margin),
        }

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save: not fitted.")
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "ambiguity_margin": self.ambiguity_margin,
            "reg": self.reg,
            "labels": self.labels,
            "centroids": {e: self._centroids[e].tolist() for e in self.labels},
            "inv_cov": {e: self._inv_cov[e].tolist() for e in self.labels},
        }
        with open(save_dir / "namer.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CentroidNamer":
        with open(Path(path) / "namer.json") as f:
            meta = json.load(f)
        obj = cls(ambiguity_margin=meta["ambiguity_margin"], reg=meta["reg"])
        obj.labels = meta["labels"]
        obj._centroids = {e: np.array(v) for e, v in meta["centroids"].items()}
        obj._inv_cov = {e: np.array(v) for e, v in meta["inv_cov"].items()}
        return obj
