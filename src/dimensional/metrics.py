"""Evaluation metrics for dimensional (V/A/D) regression.

Per TRAJECTORY_ENGINE.md Law 2: predicting valence/arousal/dominance is
REGRESSION, scored by the **Concordance Correlation Coefficient (CCC)** —
the AVEC-challenge standard — NOT "accuracy". Accuracy only applies to the
final point->category step (see namer.py).

CCC (Lin, 1989) combines Pearson correlation with a penalty for any shift in
mean or scale, so a model that is well-correlated but biased is penalised:

    CCC = 2 * cov(y, y_hat)
          ---------------------------------------------------
          var(y) + var(y_hat) + (mean(y) - mean(y_hat))^2

Range [-1, +1]; 1 = perfect agreement on the identity line.
"""

from __future__ import annotations

import numpy as np

# Canonical axis order used everywhere in the dimensional engine.
DIMENSIONS: tuple[str, str, str] = ("valence", "arousal", "dominance")


def _as_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size != yp.size:
        raise ValueError(f"Length mismatch: {yt.size} vs {yp.size}.")
    if yt.size < 2:
        raise ValueError("Need at least 2 samples for these metrics.")
    return yt, yp


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient. Range [-1, 1].

    Returns 0.0 when the denominator is degenerate (both inputs constant
    with equal means), since concordance is undefined there.
    """
    yt, yp = _as_pair(y_true, y_pred)
    mt, mp = float(yt.mean()), float(yp.mean())
    vt, vp = float(yt.var()), float(yp.var())
    cov = float(np.mean((yt - mt) * (yp - mp)))
    denom = vt + vp + (mt - mp) ** 2
    if denom == 0.0:
        return 0.0
    return float(2.0 * cov / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation. Returns 0.0 if either input has zero variance."""
    yt, yp = _as_pair(y_true, y_pred)
    if yt.var() == 0.0 or yp.var() == 0.0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-array report: ccc, rmse, pearson."""
    return {
        "ccc": ccc(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
    }


def dimensional_report(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> dict[str, dict[str, float]]:
    """Per-dimension report for (n, 3) V/A/D matrices, plus the mean CCC.

    Returns a dict keyed by dimension name (each -> ccc/rmse/pearson) with an
    extra ``"mean"`` key holding ``{"ccc": <mean CCC across dimensions>}`` —
    the single headline number used to rank dimensional SER systems.
    """
    Yt = np.asarray(Y_true, dtype=float)
    Yp = np.asarray(Y_pred, dtype=float)
    if Yt.shape != Yp.shape:
        raise ValueError(f"Shape mismatch: {Yt.shape} vs {Yp.shape}.")
    if Yt.ndim != 2 or Yt.shape[1] != len(dimensions):
        raise ValueError(
            f"Expected (n, {len(dimensions)}) arrays, got {Yt.shape}."
        )

    out: dict[str, dict[str, float]] = {}
    cccs: list[float] = []
    for i, dim in enumerate(dimensions):
        rep = regression_report(Yt[:, i], Yp[:, i])
        out[dim] = rep
        cccs.append(rep["ccc"])
    out["mean"] = {"ccc": float(np.mean(cccs))}
    return out
