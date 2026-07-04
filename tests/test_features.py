"""Tests for src.features.opensmile_extractor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.features.opensmile_extractor import (
    extract_features,
    extract_features_from_array,
    feature_names,
)
from src.preprocessing import TARGET_SR


# ── helpers ──────────────────────────────────────────────────────────

def _make_tone_wav(
    freq: float = 220.0,
    duration: float = 2.0,
    sr: int = 16_000,
) -> Path:
    """Create a temporary WAV with a sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return Path(tmp.name)


# ── feature_names ────────────────────────────────────────────────────

class TestFeatureNames:
    def test_count(self) -> None:
        names = feature_names()
        assert len(names) == 88

    def test_known_features_present(self) -> None:
        names = feature_names()
        # Spot-check key features from CLAUDE.md spec
        assert "F0semitoneFrom27.5Hz_sma3nz_amean" in names
        assert "jitterLocal_sma3nz_amean" in names
        assert "shimmerLocaldB_sma3nz_amean" in names
        assert "HNRdBACF_sma3nz_amean" in names
        assert "loudness_sma3_amean" in names
        assert "F1frequency_sma3nz_amean" in names


# ── extract_features ─────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_88_features(self) -> None:
        wav = _make_tone_wav()
        features = extract_features(wav)
        assert len(features) == 88

    def test_all_values_are_floats(self) -> None:
        wav = _make_tone_wav()
        features = extract_features(wav)
        for name, val in features.items():
            assert isinstance(val, float), f"{name} is {type(val)}"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_features("/nonexistent/audio.wav")

    def test_different_frequencies_differ(self) -> None:
        wav_low = _make_tone_wav(freq=100.0)
        wav_high = _make_tone_wav(freq=400.0)
        feat_low = extract_features(wav_low)
        feat_high = extract_features(wav_high)
        # F0 should differ meaningfully
        f0_key = "F0semitoneFrom27.5Hz_sma3nz_amean"
        assert feat_low[f0_key] != feat_high[f0_key]


# ── extract_features_from_array ──────────────────────────────────────

class TestExtractFromArray:
    def test_matches_file_extraction(self) -> None:
        wav = _make_tone_wav()
        from_file = extract_features(wav)

        y, sr = sf.read(str(wav), dtype="float32")
        from_array = extract_features_from_array(y, sr)

        # Should be very close (preprocessing may differ slightly)
        for name in from_file:
            assert name in from_array

    def test_too_short_raises(self) -> None:
        # 10ms of audio — too short for meaningful extraction
        y = np.zeros(160, dtype=np.float32)
        with pytest.raises(RuntimeError):
            extract_features_from_array(y, TARGET_SR)


# ── smoke tests on real datasets ─────────────────────────────────────

class TestRealDatasets:
    def test_ravdess_sample(self) -> None:
        path = Path("data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
        if not path.exists():
            pytest.skip("RAVDESS not downloaded")
        features = extract_features(path)
        assert len(features) == 88
        # F0 should be nonzero for speech
        assert features["F0semitoneFrom27.5Hz_sma3nz_amean"] > 0

    def test_crema_d_sample(self) -> None:
        import glob

        files = glob.glob("data/crema_d/audios/*.wav")
        if not files:
            pytest.skip("CREMA-D not downloaded")
        features = extract_features(files[0])
        assert len(features) == 88
