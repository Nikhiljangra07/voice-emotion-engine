"""eGeMAPS feature extraction via openSMILE.

This is the primary feature extractor for the waveform engine.
It produces the standardized 88-feature eGeMAPSv02 vector used in
speech emotion recognition research.

Usage:
    from src.features.opensmile_extractor import extract_features
    features = extract_features("audio.wav")  # dict of 88 floats
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import opensmile

from src.preprocessing import TARGET_SR, preprocess

# Singleton — openSMILE loads C++ configs on init, so reuse the instance.
_smile: opensmile.Smile | None = None


def _get_smile() -> opensmile.Smile:
    """Return a cached openSMILE extractor instance."""
    global _smile
    if _smile is None:
        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _smile


def feature_names() -> list[str]:
    """Return the 88 eGeMAPSv02 feature names in order."""
    return list(_get_smile().feature_names)


def extract_features(path: str | Path) -> dict[str, float]:
    """Extract eGeMAPSv02 features from an audio file.

    The audio is preprocessed first (resample to 16 kHz, trim silence,
    normalize) before feature extraction.

    Args:
        path: Path to an audio file (WAV, MP3, FLAC, etc.).

    Returns:
        Dict mapping each of the 88 feature names to its float value.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If preprocessing or extraction fails.
    """
    y, sr = preprocess(path)
    return extract_features_from_array(y, sr)


def extract_features_from_array(
    y: np.ndarray,
    sr: int = TARGET_SR,
) -> dict[str, float]:
    """Extract eGeMAPSv02 features from a preprocessed audio array.

    Use this when you already have a preprocessed signal and want to
    avoid re-loading from disk.

    Args:
        y: 1-D float32 audio array (should be preprocessed).
        sr: Sample rate of *y*.

    Returns:
        Dict mapping each of the 88 feature names to its float value.

    Raises:
        RuntimeError: If extraction fails or produces no output.
    """
    smile = _get_smile()

    try:
        df = smile.process_signal(y, sr)
    except Exception as exc:
        raise RuntimeError(f"openSMILE extraction failed: {exc}") from exc

    if df.empty:
        raise RuntimeError(
            "openSMILE returned an empty DataFrame — audio may be too "
            "short or entirely silent."
        )

    # DataFrame has one row (Functionals level). Convert to dict.
    row = df.iloc[0]
    result = {name: float(row[name]) for name in smile.feature_names}

    # openSMILE fills with NaN when the segment is too short.
    if all(np.isnan(v) for v in result.values()):
        raise RuntimeError(
            "openSMILE returned all NaN — audio is too short for "
            "meaningful feature extraction."
        )

    return result
