"""Layer 5 — the emotion TRAJECTORY (P2.4).

Turns an arbitrary-length recording into a timestamped path through PAD space:
window the audio (default 2 s window / 1 s hop), run the engine + the three
V/A/D regressors per window, smooth the sequence, and name each smoothed point
via the data-grounded centroids. The moving point is "the web" (Layer 6 plots
what this produces).

Correctness details (meticulous, not incidental):
  - Each WINDOW is peak-normalized independently, matching how MSP training
    clips were normalized (preprocess() normalizes per clip). Normalizing the
    whole file once would leave quiet windows quiet relative to loud ones —
    a distribution mismatch vs training.
  - The regressors predict in MSP NATIVE 1-7; the namer + centroids live in the
    normalized PAD plane. We convert each predicted point with
    ``normalize_vad_msp`` before naming and before storing trajectory coords.
  - A window that fails extraction (too short / NaN) is recorded as a GAP point
    with an ``error`` code — never silently dropped (TRAJECTORY_ENGINE Law 10).

Validation honesty (Law 18): MSP labels are per-segment, so this trajectory is
a DEMO/visualization on long audio. Per-tick accuracy needs a time-continuous
corpus (RECOLA/SEWA) — do not report per-window accuracy from MSP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.dimensional.loader import normalize_vad_msp
from src.dimensional.metrics import DIMENSIONS
from src.dimensional.namer import CentroidNamer
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import (
    build_feature_vector_from_array,
    feature_names,
    to_array,
)
from src.preprocessing import MIN_DURATION_S, TARGET_SR, load_audio, normalize


@dataclass
class TrajectoryPoint:
    """One window's place on the PAD path. Coords are normalized PAD plane."""

    t_start: float
    t_center: float
    t_end: float
    valence: float          # smoothed, normalized PAD ([-1, 1])
    arousal: float          # smoothed, normalized PAD ([0, 1])
    dominance: float        # smoothed, normalized PAD ([-1, 1])
    emotion: str | None
    distribution: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    ambiguous: bool = False
    error: str | None = None  # None = ok; else a window-gap error code


def window_bounds(
    n_samples: int, sr: int, window_s: float, hop_s: float
) -> list[tuple[int, int]]:
    """Sample-index (start, end) pairs for each window. Pure / testable.

    Includes the final partial window only if it is at least MIN_DURATION_S
    long (shorter tails can't produce reliable features).
    """
    win = max(1, int(round(window_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    min_len = int(MIN_DURATION_S * sr)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(start + win, n_samples)
        if end - start >= min_len:
            bounds.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return bounds


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average down the rows. Pure / testable.

    alpha in (0, 1]; 1.0 = no smoothing. NaN rows are carried over (the EMA
    holds its last value across a gap rather than propagating NaN).
    """
    values = np.asarray(values, dtype=float)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    out = np.empty_like(values)
    prev: np.ndarray | None = None
    for i, row in enumerate(values):
        if np.any(np.isnan(row)):
            out[i] = prev if prev is not None else row
            continue
        prev = row if prev is None else alpha * row + (1.0 - alpha) * prev
        out[i] = prev
    return out


class TrajectoryEngine:
    """Window → V/A/D → smooth → name, over a full recording."""

    def __init__(
        self,
        regressor: DimensionalRegressor,
        namer: CentroidNamer | None = None,
        window_s: float = 2.0,
        hop_s: float = 1.0,
        smoothing_alpha: float = 0.3,
    ) -> None:
        if window_s < MIN_DURATION_S:
            raise ValueError(f"window_s must be >= {MIN_DURATION_S}.")
        if hop_s <= 0:
            raise ValueError("hop_s must be > 0.")
        self.regressor = regressor
        self.namer = namer
        self.window_s = window_s
        self.hop_s = hop_s
        self.smoothing_alpha = smoothing_alpha

    @classmethod
    def from_saved(
        cls,
        regressor_dir: str | Path,
        namer_dir: str | Path | None = None,
        **kwargs,
    ) -> "TrajectoryEngine":
        reg = DimensionalRegressor.load(regressor_dir)
        namer = CentroidNamer.load(namer_dir) if namer_dir else None
        return cls(reg, namer, **kwargs)

    def _extract_windows(self, path: str | Path):
        """Window the audio and extract features + native V/A/D per window.

        Returns (bounds, sr, X, raw, errors): X is the (n, n_features) full
        feature matrix (NaN rows for failed windows — used by diarization);
        raw is native 1-7 V/A/D (clipped); errors[i] is None or a code string.
        """
        y, sr = load_audio(path, sr=TARGET_SR)  # resample only; no trim/normalize
        bounds = window_bounds(len(y), sr, self.window_s, self.hop_s)
        n_feat = len(feature_names())
        X = np.full((len(bounds), n_feat), np.nan)
        raw = np.full((len(bounds), len(DIMENSIONS)), np.nan)
        errors: list[str | None] = [None] * len(bounds)
        for i, (s, e) in enumerate(bounds):
            seg = normalize(y[s:e])  # per-window peak-normalize (matches training)
            try:
                vec = to_array(build_feature_vector_from_array(seg, sr))
                X[i] = vec
                # SAM V/A/D is bounded 1-7 by definition; clip extrapolation.
                raw[i] = np.clip(self.regressor.predict(vec.reshape(1, -1))[0],
                                 1.0, 7.0)
            except Exception as exc:  # noqa: BLE001 — record gap, never drop
                code = getattr(exc, "code", None)
                errors[i] = code.value if code is not None else type(exc).__name__
        return bounds, sr, X, raw, errors

    def _build_points(
        self, bounds, sr, raw, errors, idxs: list[int]
    ) -> list[TrajectoryPoint]:
        """Convert native→PAD, smooth within this index subset, and name.

        Smoothing runs only over ``idxs`` in order — so per-speaker calls smooth
        within each speaker's own timeline, never across another speaker's turn.
        """
        pad = np.full((len(idxs), len(DIMENSIONS)), np.nan)
        for k, i in enumerate(idxs):
            if errors[i] is None and not np.any(np.isnan(raw[i])):
                pad[k] = normalize_vad_msp(*raw[i])
        smoothed = ema_smooth(pad, self.smoothing_alpha)

        points: list[TrajectoryPoint] = []
        for k, i in enumerate(idxs):
            s, e = bounds[i]
            t0, t1 = s / sr, e / sr
            pt = TrajectoryPoint(
                t_start=t0, t_center=(t0 + t1) / 2, t_end=t1,
                valence=float(smoothed[k, 0]),
                arousal=float(smoothed[k, 1]),
                dominance=float(smoothed[k, 2]),
                emotion=None, error=errors[i],
            )
            if errors[i] is None and self.namer is not None \
                    and not math.isnan(smoothed[k, 0]):
                named = self.namer.predict(smoothed[k])
                pt.emotion = named["emotion"]
                pt.distribution = named["distribution"]
                pt.intensity = named["intensity"]
                pt.ambiguous = named["ambiguous"]
            points.append(pt)
        return points

    def analyze(self, path: str | Path) -> list[TrajectoryPoint]:
        """Full pipeline on one audio file → list[TrajectoryPoint]."""
        bounds, sr, _X, raw, errors = self._extract_windows(path)
        if not bounds:
            return []
        return self._build_points(bounds, sr, raw, errors, list(range(len(bounds))))

    def analyze_by_speaker(
        self,
        path: str | Path,
        diarizer=None,
        labels=None,
    ) -> dict[str, list[TrajectoryPoint]]:
        """Diarize, then return one trajectory per speaker ("Speaker A", ...).

        Provide EITHER:
          - ``diarizer``: a SpeakerDiarizer (Path A) or anything with
            ``fit_predict(X, feature_names) -> labels``; OR
          - ``labels``: a precomputed per-window label array (Path B — e.g. from
            the neural diarizer's CSV), length == number of windows.

        Windows labelled -1 (unanalyzable) are dropped from speaker groups.
        """
        if diarizer is None and labels is None:
            raise ValueError("Provide either `diarizer` or `labels`.")
        bounds, sr, X, raw, errors = self._extract_windows(path)
        if not bounds:
            return {}
        if labels is not None:
            labels = np.asarray(labels, dtype=int)
            if len(labels) != len(bounds):
                raise ValueError(
                    f"labels has {len(labels)} entries but there are "
                    f"{len(bounds)} windows — window/hop must match the diarizer."
                )
        else:
            labels = diarizer.fit_predict(X, feature_names())
        result: dict[str, list[TrajectoryPoint]] = {}
        for lab in sorted(set(int(l) for l in labels)):
            if lab < 0:
                continue
            idxs = [i for i in range(len(bounds)) if labels[i] == lab]
            result[f"Speaker {chr(65 + lab)}"] = self._build_points(
                bounds, sr, raw, errors, idxs
            )
        return result


def trajectory_to_rows(points: list[TrajectoryPoint]) -> list[dict]:
    """Flatten to plain dict rows (for DataFrame/CSV/JSON in the demo/viz)."""
    rows = []
    for p in points:
        rows.append({
            "t_start": round(p.t_start, 3),
            "t_center": round(p.t_center, 3),
            "t_end": round(p.t_end, 3),
            "valence": round(p.valence, 4) if not math.isnan(p.valence) else None,
            "arousal": round(p.arousal, 4) if not math.isnan(p.arousal) else None,
            "dominance": round(p.dominance, 4) if not math.isnan(p.dominance) else None,
            "emotion": p.emotion,
            "intensity": round(p.intensity, 4),
            "ambiguous": p.ambiguous,
            "error": p.error,
        })
    return rows
