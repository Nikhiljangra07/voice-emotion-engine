"""Single-file prediction from a trained model.

Usage:
    from src.classifier.predict import Predictor
    predictor = Predictor.from_saved("models/svm_ravdess")
    result = predictor.predict("audio.wav")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.classifier.train import TrainResult
from src.features.feature_vector import build_feature_vector, feature_names, to_array


class Predictor:
    """Predict Ekman-6 emotion from a single audio file."""

    def __init__(self, train_result: TrainResult) -> None:
        self._model = train_result.model
        self._normalizer = train_result.normalizer
        self._labels = train_result.labels
        self._model_name = train_result.model_name
        self._names = feature_names()

    @classmethod
    def from_saved(cls, path: str | Path) -> "Predictor":
        """Load a predictor from a saved model directory.

        Args:
            path: Directory containing model.joblib, normalizer/, metadata.json.
        """
        result = TrainResult.load(path)
        return cls(result)

    def predict(
        self,
        path: str | Path,
        trim: bool = True,
        top_db: float = 25.0,
    ) -> dict[str, Any]:
        """Predict emotion from an audio file.

        Args:
            path: Path to audio file (WAV, MP3, etc.).
            trim: Whether to trim leading/trailing silence.
            top_db: Silence threshold in dB (only used when *trim* is True).

        Returns:
            Dict with: emotion, confidence, ekman6_weights, features.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If feature extraction or prediction fails.
        """
        # Extract features.
        features = build_feature_vector(path, trim=trim, top_db=top_db)
        arr = to_array(features)

        # Normalize + select.
        arr_norm = self._normalizer.transform(arr.reshape(1, -1))

        # Predict.
        prediction = self._model.predict(arr_norm)[0]

        # Confidence (probability).
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(arr_norm)[0]
            confidence = float(np.max(proba))
            ekman6_weights = {
                label: round(float(p), 4)
                for label, p in zip(self._model.classes_, proba)
            }
        else:
            confidence = 0.0
            ekman6_weights = {label: 0.0 for label in self._labels}
            ekman6_weights[prediction] = 1.0

        # Key features for inspection.
        key_features = {
            "pitch_mean": features.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0.0),
            "pitch_variance": features.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.0),
            "jitter": features.get("jitterLocal_sma3nz_amean", 0.0),
            "shimmer": features.get("shimmerLocaldB_sma3nz_amean", 0.0),
            "hnr": features.get("HNRdBACF_sma3nz_amean", 0.0),
            "speech_rate": features.get("speech_rate", 0.0),
            "pause_ratio": features.get("pause_ratio", 0.0),
            "audio_duration_s": features.get("audio_duration_s", 0.0),
            "snr_estimate_db": features.get("snr_estimate_db", 0.0),
        }

        return {
            "emotion": prediction,
            "confidence": round(confidence, 4),
            "ekman6_weights": ekman6_weights,
            "features": key_features,
        }
