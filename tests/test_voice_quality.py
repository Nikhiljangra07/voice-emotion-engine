"""Tests for src.features.voice_quality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.features.voice_quality import (
    extract_voice_quality,
    extract_voice_quality_from_array,
)
from src.preprocessing import TARGET_SR


def _make_speech_wav(sr: int = 16_000, duration: float = 2.0) -> Path:
    """Create a WAV with a speech-like signal (vibrato tone)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Vibrato simulates vocal fold vibration
    vibrato = 5.0 * np.sin(2 * np.pi * 5.5 * t)
    y = (0.4 * np.sin(2 * np.pi * (150 + vibrato) * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return Path(tmp.name)


class TestExtractVoiceQuality:
    def test_returns_10_features(self) -> None:
        wav = _make_speech_wav()
        features = extract_voice_quality(wav)
        assert len(features) == 13

    def test_expected_keys(self) -> None:
        wav = _make_speech_wav()
        features = extract_voice_quality(wav)
        expected = {
            "praat_jitter_local",
            "praat_jitter_rap",
            "praat_shimmer_local",
            "praat_shimmer_apq3",
            "praat_hnr_mean",
            "praat_f0_mean_hz",
            "praat_f0_std_hz",
            "praat_f1_mean_hz",
            "praat_f2_mean_hz",
            "praat_f3_mean_hz",
            "praat_f1_bandwidth_hz",
            "praat_f2_bandwidth_hz",
            "praat_f3_bandwidth_hz",
        }
        assert set(features.keys()) == expected

    def test_no_nan_or_inf(self) -> None:
        wav = _make_speech_wav()
        features = extract_voice_quality(wav)
        for name, val in features.items():
            assert not np.isnan(val), f"{name} is NaN"
            assert not np.isinf(val), f"{name} is Inf"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_voice_quality("/nonexistent.wav")

    def test_ravdess_sample(self) -> None:
        path = Path("data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
        if not path.exists():
            pytest.skip("RAVDESS not downloaded")
        features = extract_voice_quality(path)
        assert len(features) == 13
        assert features["praat_f0_mean_hz"] > 0
        assert features["praat_hnr_mean"] > 0
