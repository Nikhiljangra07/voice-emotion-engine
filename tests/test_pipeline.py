"""Tests for src.pipeline and src.signal_mapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import run
from src.signal_mapper import map_signal


class TestSignalMapper:
    def test_anger_mapping(self) -> None:
        prediction = {
            "emotion": "anger",
            "confidence": 0.85,
            "ekman6_weights": {
                "anger": 0.85, "disgust": 0.05, "fear": 0.03,
                "joy": 0.02, "sadness": 0.03, "surprise": 0.02,
            },
            "features": {
                "pitch_mean": 35.0, "pitch_variance": 0.12,
                "jitter": 0.04, "shimmer": 1.4, "hnr": 3.0,
                "speech_rate": 4.5, "pause_ratio": 0.2,
            },
        }
        result = map_signal(prediction)

        assert "valence" in result
        assert "arousal" in result
        assert "dominance" in result
        assert "expressionStrength" in result
        assert result["valence"] < 0  # anger is negative valence
        assert result["arousal"] > 0.5  # anger is high arousal
        assert -1.0 <= result["dominance"] <= 1.0
        assert 0.0 <= result["expressionStrength"] <= 1.0

    def test_joy_mapping(self) -> None:
        prediction = {
            "emotion": "joy",
            "confidence": 0.9,
            "ekman6_weights": {
                "anger": 0.02, "disgust": 0.01, "fear": 0.02,
                "joy": 0.9, "sadness": 0.01, "surprise": 0.04,
            },
            "features": {
                "pitch_mean": 38.0, "pitch_variance": 0.15,
                "jitter": 0.02, "shimmer": 1.6, "hnr": 6.0,
                "speech_rate": 5.0, "pause_ratio": 0.18,
            },
        }
        result = map_signal(prediction)
        assert result["valence"] > 0  # joy is positive valence
        assert result["dominance"] > 0  # joy is dominant

    def test_dominance_separates_anger_from_fear(self) -> None:
        """The point of the Z axis: anger (dominant) vs fear (submissive)
        share negative valence + high arousal but split on dominance."""
        anger = map_signal({
            "emotion": "anger", "confidence": 0.9,
            "ekman6_weights": {"anger": 0.9, "disgust": 0.02, "fear": 0.02,
                               "joy": 0.02, "sadness": 0.02, "surprise": 0.02},
            "features": {"pitch_mean": 35.0, "pitch_variance": 0.12, "jitter": 0.04,
                         "shimmer": 1.4, "hnr": 3.0, "speech_rate": 4.5, "pause_ratio": 0.2},
        })
        fear = map_signal({
            "emotion": "fear", "confidence": 0.9,
            "ekman6_weights": {"anger": 0.02, "disgust": 0.02, "fear": 0.9,
                               "joy": 0.02, "sadness": 0.02, "surprise": 0.02},
            "features": {"pitch_mean": 35.0, "pitch_variance": 0.12, "jitter": 0.04,
                         "shimmer": 1.4, "hnr": 3.0, "speech_rate": 4.5, "pause_ratio": 0.2},
        })
        # Both negative valence, both high arousal...
        assert anger["valence"] < 0 and fear["valence"] < 0
        assert anger["arousal"] > 0.5 and fear["arousal"] > 0.5
        # ...but anger is more dominant than fear (the separation).
        assert anger["dominance"] > fear["dominance"]

    def test_preserves_fields(self) -> None:
        prediction = {
            "emotion": "sadness",
            "confidence": 0.7,
            "ekman6_weights": {"sadness": 0.7, "anger": 0.1, "disgust": 0.1, "fear": 0.05, "joy": 0.03, "surprise": 0.02},
            "features": {"pitch_mean": 25.0, "pitch_variance": 0.05, "jitter": 0.02, "shimmer": 1.5, "hnr": 4.0, "speech_rate": 2.0, "pause_ratio": 0.3},
        }
        result = map_signal(prediction)
        assert result["emotion"] == "sadness"
        assert result["confidence"] == 0.7
        assert result["ekman6_weights"] == prediction["ekman6_weights"]
        assert result["features"] == prediction["features"]


class TestPipeline:
    def test_end_to_end(self) -> None:
        model_dir = Path("models/svm_ravdess")
        audio = Path("data/ravdess/Actor_01/03-01-05-01-01-01-01.wav")
        if not model_dir.exists() or not audio.exists():
            pytest.skip("Model or RAVDESS not available")

        result = run(audio, model_dir)

        # All required fields present.
        assert "valence" in result
        assert "arousal" in result
        assert "dominance" in result
        assert "expressionStrength" in result
        assert "confidence" in result
        assert "emotion" in result
        assert "ekman6_weights" in result
        assert "features" in result

        # Value ranges.
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
        assert -1.0 <= result["dominance"] <= 1.0
        assert 0.0 <= result["expressionStrength"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_layer_a_spec_compliance(self) -> None:
        """Verify output matches the Layer A signal spec exactly."""
        model_dir = Path("models/svm_ravdess")
        audio = Path("data/ravdess/Actor_01/03-01-05-01-01-01-01.wav")
        if not model_dir.exists() or not audio.exists():
            pytest.skip("Model or RAVDESS not available")

        result = run(audio, model_dir)

        # Top-level keys — exact match, no extras, no missing.
        # (Phase 2: dominance added — the (V,A,D) point's third axis.)
        expected_keys = {
            "valence", "arousal", "dominance", "expressionStrength",
            "confidence", "emotion", "ekman6_weights", "features",
        }
        assert set(result.keys()) == expected_keys

        # Types.
        assert isinstance(result["valence"], float)
        assert isinstance(result["arousal"], float)
        assert isinstance(result["dominance"], float)
        assert isinstance(result["expressionStrength"], float)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["emotion"], str)
        assert isinstance(result["ekman6_weights"], dict)
        assert isinstance(result["features"], dict)

        # Ranges.
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
        assert -1.0 <= result["dominance"] <= 1.0
        assert 0.0 <= result["expressionStrength"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

        # Emotion is valid Ekman-6.
        assert result["emotion"] in {
            "anger", "disgust", "fear", "joy", "sadness", "surprise",
        }

        # ekman6_weights has all 6 emotions, sums to ~1.0.
        weights = result["ekman6_weights"]
        for emo in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]:
            assert emo in weights
            assert isinstance(weights[emo], float)
            assert 0.0 <= weights[emo] <= 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.02

        # features has all 7 required keys.
        feats = result["features"]
        for key in [
            "pitch_mean", "pitch_variance", "jitter",
            "shimmer", "hnr", "speech_rate", "pause_ratio",
        ]:
            assert key in feats
            assert isinstance(feats[key], float)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            run("/nonexistent.wav")

    def test_model_not_found(self) -> None:
        audio = Path("data/ravdess/Actor_01/03-01-05-01-01-01-01.wav")
        if not audio.exists():
            pytest.skip("RAVDESS not available")
        with pytest.raises(FileNotFoundError):
            run(audio, "/nonexistent_model/")
