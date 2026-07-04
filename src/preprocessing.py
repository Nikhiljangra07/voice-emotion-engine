"""Audio preprocessing: load, resample, normalize, trim silence."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Target format for the entire pipeline.
TARGET_SR: int = 16_000  # 16 kHz mono — standard for speech processing

# Minimum duration in seconds after trimming. Audio shorter than this
# cannot produce reliable features from openSMILE, Praat, or prosody.
MIN_DURATION_S: float = 0.5


class AudioError(RuntimeError):
    """Audio processing error with a machine-readable error code."""

    def __init__(self, message: str, code: "ErrorCode") -> None:
        super().__init__(message)
        self.code = code


class ErrorCode(Enum):
    """Machine-readable error codes for audio processing failures."""

    TOO_SHORT = "too_short"
    DECODE_FAILED = "decode_failed"
    FILE_NOT_FOUND = "file_not_found"


def load_audio(path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and resample.

    Args:
        path: Path to WAV/MP3/FLAC file.
        sr: Target sample rate. Defaults to 16 kHz.

    Returns:
        (samples, sample_rate) where samples is a 1-D float32 array
        normalized to [-1, 1].

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the file cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {path.name}: {exc}") from exc

    # Stereo → mono (average channels).
    if y.ndim == 2:
        y = np.mean(y, axis=1)

    # Resample if needed.
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    return y.astype(np.float32), sr


def trim_silence(
    y: np.ndarray,
    sr: int = TARGET_SR,
    top_db: float = 25.0,
    frame_length: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """Remove leading/trailing silence.

    Uses librosa's energy-based trimming.  A *top_db* of 25 is
    conservative enough to keep soft speech while stripping dead air.

    Args:
        y: Audio samples (1-D float array).
        sr: Sample rate (used only for documentation; trimming is
            frame-based).
        top_db: Threshold in dB below the peak RMS to consider silence.
        frame_length: FFT frame length for energy calculation.
        hop_length: Hop between frames.

    Returns:
        Trimmed audio array.  If the entire signal is below the
        threshold the original array is returned unchanged.
    """
    trimmed, _ = librosa.effects.trim(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    # Guard: never return an empty array.
    if trimmed.size == 0:
        return y
    return trimmed


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1].

    Args:
        y: Audio samples.

    Returns:
        Normalized audio.  If the signal is silent (all zeros) the
        original array is returned to avoid division by zero.
    """
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        return y
    return (y / peak).astype(np.float32)


def preprocess(
    path: str | Path,
    sr: int = TARGET_SR,
    trim: bool = True,
    top_db: float = 25.0,
) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline: load → resample → trim → normalize.

    This is the single entry point that every downstream module should
    call.  It guarantees:
      - 16 kHz mono float32
      - Leading/trailing silence removed (optional)
      - Peak-normalized to [-1, 1]

    Args:
        path: Path to audio file.
        sr: Target sample rate.
        trim: Whether to trim silence.
        top_db: Silence threshold (only used when *trim* is True).

    Returns:
        (samples, sample_rate)
    """
    y, sr = load_audio(path, sr=sr)

    if trim:
        y = trim_silence(y, sr=sr, top_db=top_db)

    # Check minimum duration after trimming.
    duration = len(y) / sr
    if duration < MIN_DURATION_S:
        raise AudioError(
            f"Audio too short after processing: {duration:.3f}s "
            f"(minimum {MIN_DURATION_S}s). File: {path}",
            code=ErrorCode.TOO_SHORT,
        )

    y = normalize(y)

    return y, sr
