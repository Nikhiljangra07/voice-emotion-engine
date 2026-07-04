"""Tests for the emotion trajectory engine (Layer 5, P2.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dimensional import (
    CentroidNamer,
    DimensionalRegressor,
    TrajectoryEngine,
    ema_smooth,
    trajectory_to_rows,
    window_bounds,
)
from src.features.feature_vector import feature_names
from src.preprocessing import MIN_DURATION_S

_SHORT_WAV = Path("own_voice/001.wav")


# ───────────────────────── window_bounds ─────────────────────────
class TestWindowBounds:
    def test_basic_count_and_hop(self) -> None:
        # 10s at 16kHz, 2s window, 1s hop -> windows at 0,1,...,8 = 9 windows.
        b = window_bounds(10 * 16000, 16000, window_s=2.0, hop_s=1.0)
        assert len(b) == 9
        assert b[0] == (0, 32000)
        assert b[1][0] == 16000  # 1s hop

    def test_partial_tail_kept_if_long_enough(self) -> None:
        # 5.5s: last window (4.0-5.5 = 1.5s) >= MIN_DURATION_S -> kept.
        b = window_bounds(int(5.5 * 16000), 16000, 2.0, 1.0)
        assert b[-1][1] == int(5.5 * 16000)

    def test_short_tail_dropped(self) -> None:
        # tail shorter than MIN_DURATION_S must be excluded.
        n = int(4.0 * 16000) + int(0.2 * 16000)  # last hop leaves 0.2s tail
        b = window_bounds(n, 16000, 2.0, 1.0)
        for s, e in b:
            assert (e - s) >= int(MIN_DURATION_S * 16000)

    def test_clip_shorter_than_window(self) -> None:
        # 1s clip, 2s window: one window covering the whole clip (>= min dur).
        b = window_bounds(16000, 16000, 2.0, 1.0)
        assert b == [(0, 16000)]


# ───────────────────────── ema_smooth ─────────────────────────
class TestEmaSmooth:
    def test_alpha_one_is_identity(self) -> None:
        v = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
        assert np.allclose(ema_smooth(v, 1.0), v)

    def test_smoothing_reduces_jumps(self) -> None:
        v = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out = ema_smooth(v, 0.5)
        assert np.allclose(out[1], [0.5, 0.5, 0.5])  # halfway

    def test_nan_row_carries_previous(self) -> None:
        v = np.array([[0.4, 0.4, 0.4], [np.nan, np.nan, np.nan], [0.6, 0.6, 0.6]])
        out = ema_smooth(v, 1.0)
        assert np.allclose(out[1], out[0])  # gap holds last value

    def test_bad_alpha(self) -> None:
        with pytest.raises(ValueError):
            ema_smooth(np.zeros((3, 3)), 0.0)


# ─────────────────── full engine (synthetic models) ───────────────────
def _tiny_regressor() -> DimensionalRegressor:
    rng = np.random.default_rng(0)
    nf = len(feature_names())
    X = rng.normal(size=(80, nf))
    # native 1-7 targets
    Y = rng.uniform(1, 7, size=(80, 3))
    return DimensionalRegressor(model="ridge", calibrate=True).fit(X, Y)


def _tiny_namer() -> CentroidNamer:
    rng = np.random.default_rng(1)
    centres = {"joy": [0.4, 0.5, 0.3], "anger": [-0.5, 0.6, 0.4],
               "sadness": [-0.3, 0.3, -0.1]}
    pts, labels = [], []
    for emo, c in centres.items():
        pts.append(np.array(c) + rng.normal(scale=0.05, size=(30, 3)))
        labels += [emo] * 30
    return CentroidNamer().fit(np.vstack(pts), labels)


class TestTrajectoryEngine:
    def test_construction_guards(self) -> None:
        reg = _tiny_regressor()
        with pytest.raises(ValueError):
            TrajectoryEngine(reg, window_s=0.1)  # below MIN_DURATION_S
        with pytest.raises(ValueError):
            TrajectoryEngine(reg, hop_s=0.0)

    @pytest.mark.skipif(not _SHORT_WAV.exists(), reason="demo wav not present")
    def test_analyze_real_clip(self) -> None:
        eng = TrajectoryEngine(_tiny_regressor(), _tiny_namer(),
                               window_s=2.0, hop_s=1.0)
        pts = eng.analyze(_SHORT_WAV)
        assert len(pts) >= 1
        # timestamps strictly increase; PAD coords in range; named or gap.
        for i, p in enumerate(pts):
            assert p.t_end > p.t_start
            if i > 0:
                assert p.t_start >= pts[i - 1].t_start
            if p.error is None:
                assert -1.0 <= p.valence <= 1.0
                assert 0.0 <= p.arousal <= 1.0
                assert -1.0 <= p.dominance <= 1.0
                assert p.emotion in {"joy", "anger", "sadness"}
                assert abs(sum(p.distribution.values()) - 1.0) < 1e-6

    @pytest.mark.skipif(not _SHORT_WAV.exists(), reason="demo wav not present")
    def test_rows_serializable(self) -> None:
        eng = TrajectoryEngine(_tiny_regressor(), _tiny_namer())
        rows = trajectory_to_rows(eng.analyze(_SHORT_WAV))
        assert isinstance(rows, list) and isinstance(rows[0], dict)
        assert {"t_center", "valence", "emotion", "error"} <= set(rows[0])

    @pytest.mark.skipif(not _SHORT_WAV.exists(), reason="demo wav not present")
    def test_analyze_by_speaker_precomputed_labels(self) -> None:
        # Path B contract: caller supplies per-window labels (e.g. from the
        # neural diarizer). Two labels → two speaker groups covering all windows.
        eng = TrajectoryEngine(_tiny_regressor(), _tiny_namer())
        n = len(eng.analyze(_SHORT_WAV))
        labels = [i % 2 for i in range(n)]  # alternate A/B
        by_spk = eng.analyze_by_speaker(_SHORT_WAV, labels=labels)
        assert set(by_spk) == {"Speaker A", "Speaker B"}
        assert sum(len(v) for v in by_spk.values()) == n

    @pytest.mark.skipif(not _SHORT_WAV.exists(), reason="demo wav not present")
    def test_analyze_by_speaker_label_length_mismatch(self) -> None:
        eng = TrajectoryEngine(_tiny_regressor(), _tiny_namer())
        n = len(eng.analyze(_SHORT_WAV))
        with pytest.raises(ValueError):
            eng.analyze_by_speaker(_SHORT_WAV, labels=[0] * (n + 1))  # off by one

    def test_analyze_by_speaker_requires_diarizer_or_labels(self) -> None:
        eng = TrajectoryEngine(_tiny_regressor(), _tiny_namer())
        with pytest.raises(ValueError):
            eng.analyze_by_speaker("own_voice/001.wav")  # neither provided
