"""Tests for src.features.prosody."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.features.prosody import extract_prosody, extract_prosody_from_array
from src.preprocessing import TARGET_SR


def _make_wav_with_pauses(sr: int = 16_000) -> Path:
    """Create a WAV with alternating tone and silence (speech-like)."""
    tone_dur = 0.3  # seconds
    pause_dur = 0.2
    segments = []
    for _ in range(5):
        t = np.linspace(0, tone_dur, int(sr * tone_dur), endpoint=False)
        segments.append((0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
        segments.append(np.zeros(int(sr * pause_dur), dtype=np.float32))
    y = np.concatenate(segments)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return Path(tmp.name)


class TestExtractProsody:
    def test_returns_10_features(self) -> None:
        wav = _make_wav_with_pauses()
        features = extract_prosody(wav)
        assert len(features) == 10

    def test_expected_keys(self) -> None:
        wav = _make_wav_with_pauses()
        features = extract_prosody(wav)
        expected = {
            "speech_rate",
            "pause_ratio",
            "pause_count",
            "mean_pause_duration_s",
            "voiced_fraction",
            "tempo_bpm",
            "attack_time_s",
            "decay_time_s",
            "attack_slope",
            "sustain_ratio",
        }
        assert set(features.keys()) == expected

    def test_pause_ratio_in_range(self) -> None:
        wav = _make_wav_with_pauses()
        features = extract_prosody(wav)
        assert 0.0 <= features["pause_ratio"] <= 1.0

    def test_voiced_fraction_complement(self) -> None:
        wav = _make_wav_with_pauses()
        features = extract_prosody(wav)
        assert abs(
            features["voiced_fraction"] + features["pause_ratio"] - 1.0
        ) < 1e-6

    def test_speech_rate_positive(self) -> None:
        wav = _make_wav_with_pauses()
        features = extract_prosody(wav)
        assert features["speech_rate"] > 0

    def test_too_short_raises(self) -> None:
        from src.preprocessing import AudioError
        y = np.zeros(100, dtype=np.float32)
        with pytest.raises(AudioError, match="too short"):
            extract_prosody_from_array(y, TARGET_SR)

    def test_ravdess_sample(self) -> None:
        path = Path("data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
        if not path.exists():
            pytest.skip("RAVDESS not downloaded")
        features = extract_prosody(path)
        assert features["speech_rate"] > 0
        assert 0.0 <= features["pause_ratio"] <= 1.0
