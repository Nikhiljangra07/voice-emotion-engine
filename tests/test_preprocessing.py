"""Tests for src.preprocessing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import (
    TARGET_SR,
    AudioError,
    ErrorCode,
    load_audio,
    normalize,
    preprocess,
    trim_silence,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_wav(
    sr: int = 48_000,
    duration: float = 1.0,
    freq: float = 440.0,
    channels: int = 1,
    amplitude: float = 0.5,
    silence_pad: float = 0.3,
) -> Path:
    """Write a temporary WAV and return its path."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # Pad silence before and after.
    pad = np.zeros(int(sr * silence_pad), dtype=np.float32)
    y = np.concatenate([pad, tone, pad])

    if channels == 2:
        y = np.column_stack([y, y * 0.8])

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return Path(tmp.name)


# ── load_audio ───────────────────────────────────────────────────────

class TestLoadAudio:
    def test_resamples_to_target(self) -> None:
        wav = _make_wav(sr=48_000)
        y, sr = load_audio(wav)
        assert sr == TARGET_SR

    def test_stereo_to_mono(self) -> None:
        wav = _make_wav(channels=2)
        y, sr = load_audio(wav)
        assert y.ndim == 1

    def test_output_is_float32(self) -> None:
        wav = _make_wav()
        y, _ = load_audio(wav)
        assert y.dtype == np.float32

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav")

    def test_already_target_sr(self) -> None:
        wav = _make_wav(sr=TARGET_SR)
        y, sr = load_audio(wav)
        assert sr == TARGET_SR
        assert len(y) > 0


# ── trim_silence ─────────────────────────────────────────────────────

class TestTrimSilence:
    def test_removes_padding(self) -> None:
        sr = 16_000
        pad = np.zeros(sr, dtype=np.float32)  # 1s silence
        tone = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2, endpoint=False)
        ).astype(np.float32)
        y = np.concatenate([pad, tone, pad])

        trimmed = trim_silence(y, sr=sr)
        # Trimmed should be shorter — silence removed.
        assert len(trimmed) < len(y)
        # But the tone should still be there.
        assert len(trimmed) > 0

    def test_all_silence_returns_original(self) -> None:
        y = np.zeros(16_000, dtype=np.float32)
        trimmed = trim_silence(y)
        assert len(trimmed) == len(y)

    def test_no_silence_unchanged(self) -> None:
        sr = 16_000
        t = np.linspace(0, 1, sr, endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        trimmed = trim_silence(y, sr=sr)
        # Should be roughly the same length (within a frame).
        assert abs(len(trimmed) - len(y)) < 512


# ── normalize ────────────────────────────────────────────────────────

class TestNormalize:
    def test_peak_is_one(self) -> None:
        y = np.array([0.0, 0.3, -0.5, 0.2], dtype=np.float32)
        normed = normalize(y)
        assert np.isclose(np.max(np.abs(normed)), 1.0)

    def test_silent_signal_unchanged(self) -> None:
        y = np.zeros(100, dtype=np.float32)
        normed = normalize(y)
        assert np.all(normed == 0.0)

    def test_already_normalized(self) -> None:
        y = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        normed = normalize(y)
        np.testing.assert_array_almost_equal(normed, y)

    def test_output_dtype(self) -> None:
        y = np.array([0.1, -0.2], dtype=np.float64)
        normed = normalize(y)
        assert normed.dtype == np.float32


# ── preprocess (integration) ─────────────────────────────────────────

class TestPreprocess:
    def test_full_pipeline(self) -> None:
        wav = _make_wav(sr=48_000, silence_pad=0.5)
        y, sr = preprocess(wav)

        assert sr == TARGET_SR
        assert y.dtype == np.float32
        assert np.max(np.abs(y)) <= 1.0 + 1e-6
        assert len(y) > 0

    def test_no_trim(self) -> None:
        wav = _make_wav(sr=48_000, silence_pad=0.5)
        y_trim, _ = preprocess(wav, trim=True)
        y_no_trim, _ = preprocess(wav, trim=False)
        # Without trimming, should be longer (silence kept).
        assert len(y_no_trim) >= len(y_trim)

    def test_with_ravdess_file(self) -> None:
        """Smoke test against a real RAVDESS file if available."""
        ravdess = Path("data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
        if not ravdess.exists():
            pytest.skip("RAVDESS not downloaded")
        y, sr = preprocess(ravdess)
        assert sr == TARGET_SR
        assert y.dtype == np.float32
        assert 0.5 < len(y) / sr < 10.0  # Reasonable speech duration

    def test_with_crema_d_file(self) -> None:
        """Smoke test against a real CREMA-D file if available."""
        import glob

        files = glob.glob("data/crema_d/audios/*.wav")
        if not files:
            pytest.skip("CREMA-D not downloaded")
        y, sr = preprocess(files[0])
        assert sr == TARGET_SR
        assert y.dtype == np.float32
        assert len(y) > 0

    def test_too_short_raises_after_trim(self) -> None:
        """Audio that becomes < 0.5s after trimming should raise."""
        # Create a very short WAV (0.05s of audio)
        short_signal = np.random.randn(int(TARGET_SR * 0.05)).astype(np.float32) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, short_signal, TARGET_SR)
            with pytest.raises(AudioError) as exc_info:
                preprocess(f.name)
            assert exc_info.value.code == ErrorCode.TOO_SHORT

    def test_too_short_error_code(self) -> None:
        """AudioError should carry the TOO_SHORT error code."""
        short_signal = np.random.randn(int(TARGET_SR * 0.1)).astype(np.float32) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, short_signal, TARGET_SR)
            with pytest.raises(AudioError) as exc_info:
                preprocess(f.name)
            assert exc_info.value.code == ErrorCode.TOO_SHORT
            assert "too short" in str(exc_info.value).lower()
