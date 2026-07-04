"""Combined feature vector: openSMILE + parselmouth + prosody.

This is the single entry point for feature extraction. Every downstream
module (classifier, evaluation) should use `build_feature_vector()` to
get a consistent, complete feature dict from an audio file.

Feature groups:
  - eGeMAPS (88 features): F0, jitter, shimmer, HNR, MFCCs, formants,
    loudness, spectral flux, voice segment stats.
  - Praat voice quality (13 features): clinical-grade jitter, shimmer,
    HNR, F0, formants (frequency + bandwidth) via parselmouth.
  - Prosody (10 features): speech rate, pause ratio, pause count,
    mean pause duration, voiced fraction, tempo, energy envelope
    (attack time, decay time, attack slope, sustain ratio).

Total: 111 classifier features + 2 metadata features per audio file.

Metadata features (not used by classifier, but available in the dict):
  - audio_duration_s: post-trim audio length in seconds.
  - snr_estimate_db: estimated signal-to-noise ratio in dB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

from src.features.opensmile_extractor import (
    extract_features_from_array,
)
from src.features.prosody import extract_prosody_from_array
from src.features.voice_quality import extract_voice_quality_from_array
from src.preprocessing import TARGET_SR, preprocess


def _estimate_snr(y: np.ndarray, sr: int) -> float:
    """Estimate signal-to-noise ratio in dB.

    Uses top-10% RMS frames as signal level and bottom-10% as noise floor.

    Args:
        y: Audio signal array.
        sr: Sample rate.

    Returns:
        Estimated SNR in dB. Returns 60.0 if noise floor is near zero.
    """
    frame_length = max(int(0.025 * sr), 1)
    hop_length = max(int(0.010 * sr), 1)
    if len(y) < frame_length:
        return 0.0

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    sorted_rms = np.sort(frame_rms)
    n_frames = len(sorted_rms)

    noise_floor = float(np.mean(sorted_rms[: max(1, n_frames // 10)]))
    signal_level = float(np.mean(sorted_rms[-max(1, n_frames // 10) :]))

    if noise_floor < 1e-10:
        return 60.0
    return float(20.0 * np.log10(signal_level / noise_floor))


# Expected ranges for key features. Values outside these are flagged
# as warnings (not errors) — real speech can occasionally exceed them.
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": (5.0, 60.0),
    "jitterLocal_sma3nz_amean": (0.0, 0.2),
    "shimmerLocaldB_sma3nz_amean": (0.0, 5.0),
    "HNRdBACF_sma3nz_amean": (-5.0, 40.0),
    "loudness_sma3_amean": (0.0, 10.0),
    "praat_f0_mean_hz": (0.0, 800.0),
    "praat_jitter_local": (0.0, 0.2),
    "praat_shimmer_local": (0.0, 1.0),
    "praat_hnr_mean": (-5.0, 40.0),
    "speech_rate": (0.0, 30.0),
    "pause_ratio": (0.0, 1.0),
}


def validate_feature_vector(features: dict[str, float]) -> list[str]:
    """Validate a feature dict for integrity.

    Checks:
      1. All 104 classifier features present.
      2. No NaN or Inf values in classifier features.
      3. Key features within expected ranges (logs warnings, does not raise).

    Args:
        features: Feature dict from build_feature_vector().

    Returns:
        List of warning messages (empty if all checks pass).

    Raises:
        RuntimeError: If classifier features are missing or contain NaN/Inf.
    """
    warnings_list: list[str] = []
    names = feature_names()

    # ── 1. Check all 104 classifier features present ──
    missing = [n for n in names if n not in features]
    if missing:
        raise RuntimeError(
            f"Feature vector missing {len(missing)} classifier features: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # ── 2. Check no NaN/Inf in classifier features ──
    nan_features = [n for n in names if np.isnan(features[n])]
    inf_features = [n for n in names if np.isinf(features[n])]
    if nan_features:
        raise RuntimeError(
            f"Feature vector contains NaN in {len(nan_features)} features: "
            f"{nan_features[:5]}{'...' if len(nan_features) > 5 else ''}"
        )
    if inf_features:
        raise RuntimeError(
            f"Feature vector contains Inf in {len(inf_features)} features: "
            f"{inf_features[:5]}{'...' if len(inf_features) > 5 else ''}"
        )

    # ── 3. Range checks on key features (warn, don't raise) ──
    for feat_name, (lo, hi) in _FEATURE_RANGES.items():
        if feat_name not in features:
            continue
        val = features[feat_name]
        if val < lo or val > hi:
            msg = (
                f"Feature '{feat_name}' = {val:.4f} is outside expected "
                f"range [{lo}, {hi}]"
            )
            warnings_list.append(msg)
            logger.warning(msg)

    return warnings_list


def build_feature_vector(
    path: str | Path,
    trim: bool = True,
    top_db: float = 25.0,
) -> dict[str, float]:
    """Extract all features from an audio file.

    Preprocessing is done once; the same signal is passed to all three
    extractors.

    Args:
        path: Path to audio file.
        trim: Whether to trim leading/trailing silence. Set to False
            for whispered or very soft emotional speech where trimming
            may remove signal.
        top_db: Silence threshold in dB below peak RMS. Lower values
            (e.g. 20) trim more aggressively; higher values (e.g. 30)
            preserve more soft speech. Only used when *trim* is True.

    Returns:
        Dict mapping feature names to float values (104 features).

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If extraction fails.
    """
    y, sr = preprocess(path, trim=trim, top_db=top_db)
    return build_feature_vector_from_array(y, sr)


def build_feature_vector_from_array(
    y: np.ndarray,
    sr: int = TARGET_SR,
) -> dict[str, float]:
    """Extract all features from a preprocessed audio array.

    Args:
        y: 1-D float32 audio array (preprocessed).
        sr: Sample rate.

    Returns:
        Dict mapping feature names to float values.
        Contains 104 classifier features + 2 metadata features
        (audio_duration_s, snr_estimate_db).
    """
    features: dict[str, float] = {}

    # 1. openSMILE eGeMAPS (88 features)
    features.update(extract_features_from_array(y, sr))

    # 2. Praat voice quality (10 features)
    features.update(extract_voice_quality_from_array(y, sr))

    # 3. Prosody (6 features)
    features.update(extract_prosody_from_array(y, sr))

    # 4. Metadata (not used by classifier, but available downstream)
    features["audio_duration_s"] = float(len(y) / sr)
    features["snr_estimate_db"] = _estimate_snr(y, sr)

    # 5. Validate before returning — raises on missing/NaN/Inf,
    #    logs warnings for out-of-range values.
    validate_feature_vector(features)

    return features


def feature_names() -> list[str]:
    """Return the ordered list of all feature names.

    Useful for building numpy arrays with consistent column ordering.
    """
    from src.features.opensmile_extractor import (
        feature_names as opensmile_names,
    )

    praat_names = [
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
    ]

    prosody_names = [
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
    ]

    return opensmile_names() + praat_names + prosody_names


# Metadata keys that are NOT classifier features.
METADATA_KEYS = ("audio_duration_s", "snr_estimate_db")


def to_array(features: dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a 1-D numpy array in canonical order.

    Only includes classifier features (111). Metadata keys
    (audio_duration_s, snr_estimate_db) are excluded.

    Args:
        features: Feature dict from `build_feature_vector()`.

    Returns:
        1-D float64 array of length 111.
    """
    names = feature_names()
    return np.array([float(features[n]) for n in names], dtype=np.float64)
