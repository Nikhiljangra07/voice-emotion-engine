"""Tests for the speaker diarization front-end (Path A)."""

from __future__ import annotations

import numpy as np
import pytest

from src.dimensional import (
    SPEAKER_FEATURE_NAMES,
    SpeakerDiarizer,
    speaker_feature_indices,
)
from src.features.feature_vector import feature_names

NAMES = feature_names()


class TestSpeakerFeatures:
    def test_indices_resolve(self) -> None:
        idx = speaker_feature_indices(NAMES)
        assert len(idx) >= 6
        assert all(0 <= i < len(NAMES) for i in idx)
        # every resolved index maps back to a known speaker-feature name
        for i in idx:
            assert NAMES[i] in SPEAKER_FEATURE_NAMES

    def test_schema_change_raises(self) -> None:
        with pytest.raises(ValueError):
            speaker_feature_indices(["unrelated", "columns"])


def _two_speaker_matrix(n_each: int = 30, seed: int = 0) -> np.ndarray:
    """Full (2*n, n_features) matrix with two well-separated speaker clusters
    in the speaker-feature columns, noise elsewhere."""
    rng = np.random.default_rng(seed)
    nf = len(NAMES)
    X = rng.normal(scale=0.1, size=(2 * n_each, nf))
    cols = speaker_feature_indices(NAMES)
    # Speaker A: low formant/MFCC offsets; Speaker B: high — clearly separated.
    X[:n_each][:, cols] += rng.normal(loc=-3.0, scale=0.3, size=(n_each, len(cols)))
    X[n_each:][:, cols] += rng.normal(loc=+3.0, scale=0.3, size=(n_each, len(cols)))
    return X


class TestDiarizer:
    def test_two_distinct_speakers_fixed_k(self) -> None:
        X = _two_speaker_matrix()
        labels = SpeakerDiarizer(n_speakers=2, min_turn_s=0.0).fit_predict(X, NAMES)
        # Two clusters, each block internally consistent (allowing label perm).
        assert len(set(labels)) == 2
        first_half, second_half = labels[:30], labels[30:]
        assert len(set(first_half)) == 1 and len(set(second_half)) == 1
        assert first_half[0] != second_half[0]

    def test_auto_estimate_k(self) -> None:
        X = _two_speaker_matrix()
        d = SpeakerDiarizer(n_speakers=None, max_speakers=5, min_turn_s=0.0)
        d.fit_predict(X, NAMES)
        assert d.estimated_k_ == 2

    def test_nan_rows_labelled_minus_one(self) -> None:
        X = _two_speaker_matrix()
        X[5] = np.nan  # a failed window
        labels = SpeakerDiarizer(n_speakers=2, min_turn_s=0.0).fit_predict(X, NAMES)
        assert labels[5] == -1
        assert (labels >= 0).sum() == len(labels) - 1

    def test_min_turn_smoothing(self) -> None:
        # A single-window blip of speaker 1 inside a speaker-0 run is smoothed.
        d = SpeakerDiarizer(n_speakers=2, min_turn_s=3.0, hop_s=1.0)
        labels = np.array([0, 0, 0, 1, 0, 0, 0])
        assert np.all(d._smooth(labels) == 0)

    def test_long_turn_preserved(self) -> None:
        d = SpeakerDiarizer(n_speakers=2, min_turn_s=3.0, hop_s=1.0)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # both runs >= 3
        assert np.array_equal(d._smooth(labels), labels)
