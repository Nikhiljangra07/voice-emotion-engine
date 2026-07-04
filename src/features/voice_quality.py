"""Voice quality features via parselmouth (Praat).

Extracts clinical-grade jitter, shimmer, HNR, and formants.
These complement openSMILE's eGeMAPS features with Praat's gold-standard
voice analysis algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call

from src.preprocessing import TARGET_SR, preprocess

# Wide-range pitch floor/ceiling for initial F0 estimation.
# Covers all human voices: elderly males (~50 Hz) to children/high
# female surprise (~800 Hz).
_F0_FLOOR_WIDE: float = 50.0
_F0_CEIL_WIDE: float = 800.0

# Absolute safety limits for the adaptive range.
_F0_FLOOR_MIN: float = 40.0
_F0_CEIL_MAX: float = 900.0


def extract_voice_quality(path: str | Path) -> dict[str, float]:
    """Extract voice quality features from an audio file.

    Args:
        path: Path to audio file.

    Returns:
        Dict with voice quality features.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If extraction fails.
    """
    y, sr = preprocess(path)
    return extract_voice_quality_from_array(y, sr)


def extract_voice_quality_from_array(
    y: np.ndarray,
    sr: int = TARGET_SR,
) -> dict[str, float]:
    """Extract voice quality features from a preprocessed audio array.

    Args:
        y: 1-D float32 audio array.
        sr: Sample rate.

    Returns:
        Dict with keys: praat_jitter_local, praat_jitter_rap,
        praat_shimmer_local, praat_shimmer_apq3, praat_hnr_mean,
        praat_f0_mean, praat_f0_std, praat_f1_mean, praat_f2_mean,
        praat_f3_mean.
    """
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    # ── Adaptive pitch range ───────────────────────────────────
    # Pass 1: wide sweep to find approximate F0.
    # Pass 2: narrow to ±2 octaves around detected F0 for precision.
    f0_floor, f0_ceil = _adaptive_f0_range(snd)

    # ── Pitch ────────────────────────────────────────────────────
    pitch = call(snd, "To Pitch", 0.0, f0_floor, f0_ceil)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    # ── Point process (for jitter/shimmer) ───────────────────────
    point_process = call(
        snd, "To PointProcess (periodic, cc)", f0_floor, f0_ceil
    )

    # ── Jitter ───────────────────────────────────────────────────
    jitter_local = call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_rap = call(
        point_process,
        "Get jitter (rap)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
    )

    # ── Shimmer ──────────────────────────────────────────────────
    shimmer_local = call(
        [snd, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    shimmer_apq3 = call(
        [snd, point_process],
        "Get shimmer (apq3)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    # ── Harmonics-to-noise ratio ─────────────────────────────────
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, f0_floor, 0.1, 1.0)
    hnr_mean = call(harmonicity, "Get mean", 0, 0)

    # ── Formants (frequency + bandwidth) ──────────────────────────
    formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1_mean = call(formants, "Get mean", 1, 0, 0, "Hertz")
    f2_mean = call(formants, "Get mean", 2, 0, 0, "Hertz")
    f3_mean = call(formants, "Get mean", 3, 0, 0, "Hertz")

    # Bandwidth: average across all frames. Narrow = stressed/tense,
    # wide = relaxed/breathy.
    duration = snd.get_total_duration()
    midpoint = duration / 2.0
    f1_bw = call(formants, "Get bandwidth at time", 1, midpoint, "Hertz", "Linear")
    f2_bw = call(formants, "Get bandwidth at time", 2, midpoint, "Hertz", "Linear")
    f3_bw = call(formants, "Get bandwidth at time", 3, midpoint, "Hertz", "Linear")

    return {
        "praat_jitter_local": _safe_float(jitter_local),
        "praat_jitter_rap": _safe_float(jitter_rap),
        "praat_shimmer_local": _safe_float(shimmer_local),
        "praat_shimmer_apq3": _safe_float(shimmer_apq3),
        "praat_hnr_mean": _safe_float(hnr_mean),
        "praat_f0_mean_hz": _safe_float(f0_mean),
        "praat_f0_std_hz": _safe_float(f0_std),
        "praat_f1_mean_hz": _safe_float(f1_mean),
        "praat_f2_mean_hz": _safe_float(f2_mean),
        "praat_f3_mean_hz": _safe_float(f3_mean),
        "praat_f1_bandwidth_hz": _safe_float(f1_bw),
        "praat_f2_bandwidth_hz": _safe_float(f2_bw),
        "praat_f3_bandwidth_hz": _safe_float(f3_bw),
    }


def _adaptive_f0_range(snd: parselmouth.Sound) -> tuple[float, float]:
    """Estimate speaker-adaptive F0 floor and ceiling.

    Pass 1: wide sweep (50-800 Hz) to find approximate F0 mean.
    Pass 2: narrow to ±2 octaves around detected F0.

    Falls back to wide range if no voicing is detected.

    Args:
        snd: parselmouth Sound object.

    Returns:
        (f0_floor, f0_ceiling) in Hz.
    """
    # Pass 1: wide sweep
    pitch_wide = call(snd, "To Pitch", 0.0, _F0_FLOOR_WIDE, _F0_CEIL_WIDE)
    f0_mean = call(pitch_wide, "Get mean", 0, 0, "Hertz")

    if f0_mean is None or np.isnan(f0_mean) or f0_mean <= 0:
        # No voicing detected — fall back to wide range.
        return _F0_FLOOR_WIDE, _F0_CEIL_WIDE

    # Pass 2: ±2 octaves around detected mean.
    # 2 octaves down = f0 / 4, 2 octaves up = f0 * 4
    floor = max(f0_mean / 4.0, _F0_FLOOR_MIN)
    ceil = min(f0_mean * 4.0, _F0_CEIL_MAX)

    return float(floor), float(ceil)


def _safe_float(val: Any) -> float:
    """Convert to float, replacing undefined/NaN with 0.0."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0
