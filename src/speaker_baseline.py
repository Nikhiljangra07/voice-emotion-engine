"""Per-speaker baseline for expressionStrength normalization.

A speaker's "neutral voice" is captured once (from a calibration
sample or the first few seconds of audio), then all subsequent
expressionStrength values are measured as deviations from THAT
speaker's baseline — not the global RAVDESS average.

This matters because a naturally breathy speaker (low HNR) shouldn't
register as "expressive" just because their HNR differs from the
population mean. ExpressionStrength should measure how different
THIS utterance is from THIS speaker's normal voice.

Usage:
    from src.speaker_baseline import SpeakerBaseline

    # Option 1: calibrate from a "neutral" audio file
    baseline = SpeakerBaseline.from_audio("neutral_sample.wav")

    # Option 2: calibrate from extracted features
    baseline = SpeakerBaseline.from_features(features_dict)

    # Use in signal mapper
    result = map_signal(prediction, speaker_baseline=baseline)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.feature_vector import build_feature_vector

# The 4 features used for expressionStrength.
_ES_FEATURE_KEYS: list[str] = [
    "jitter",
    "shimmer",
    "hnr",
    "pitch_variance",
]

# Mapping from ES keys to full feature names.
_ES_KEY_TO_FEATURE: dict[str, str] = {
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
    "hnr": "HNRdBACF_sma3nz_amean",
    "pitch_variance": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
}


class SpeakerBaseline:
    """Stores a speaker's neutral voice characteristics.

    Used to compute per-speaker expressionStrength in the signal mapper.
    """

    def __init__(self, baselines: dict[str, float]) -> None:
        """Create from a dict of feature baselines.

        Args:
            baselines: Dict with keys matching _ES_FEATURE_KEYS,
                values are the speaker's neutral measurements.
        """
        for key in _ES_FEATURE_KEYS:
            if key not in baselines:
                raise ValueError(f"Missing baseline feature: {key}")
        self._baselines = dict(baselines)

    @property
    def baselines(self) -> dict[str, float]:
        """The speaker's neutral feature values."""
        return dict(self._baselines)

    @classmethod
    def from_audio(cls, path: str | Path) -> "SpeakerBaseline":
        """Create a baseline from a neutral speech audio file.

        The audio should be the speaker talking in their normal,
        unemotional voice. Even a few seconds is enough.

        Args:
            path: Path to audio file.

        Returns:
            SpeakerBaseline calibrated to this speaker.
        """
        features = build_feature_vector(path)
        return cls.from_features(features)

    @classmethod
    def from_features(cls, features: dict[str, Any]) -> "SpeakerBaseline":
        """Create a baseline from already-extracted features.

        Args:
            features: Full 104-feature dict from build_feature_vector(),
                or a prediction's 'features' dict with the short keys.
        """
        baselines: dict[str, float] = {}

        for es_key, full_key in _ES_KEY_TO_FEATURE.items():
            # Try full feature name first, then short key.
            if full_key in features:
                baselines[es_key] = float(features[full_key])
            elif es_key in features:
                baselines[es_key] = float(features[es_key])
            else:
                raise ValueError(
                    f"Cannot find '{es_key}' or '{full_key}' in features."
                )

        return cls(baselines)

    def save(self, path: str | Path) -> None:
        """Save baseline to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._baselines, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerBaseline":
        """Load baseline from a JSON file."""
        with open(path) as f:
            return cls(json.load(f))
