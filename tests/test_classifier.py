"""Tests for src.classifier (train, evaluate, predict)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.classifier.train import TrainResult, train_random_forest, train_svm
from src.classifier.evaluate import (
    full_report,
    misclassification_analysis,
    per_class_metrics,
)
from src.classifier.predict import Predictor


# ── helpers ──────────────────────────────────────────────────────────

def _make_synthetic_data(
    n_per_class: int = 50,
    n_features: int = 104,
    n_classes: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic feature matrix with separable classes."""
    rng = np.random.RandomState(seed)
    labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"][:n_classes]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    X_parts = []
    y_parts = []
    for i, label in enumerate(labels):
        # Each class has a shifted mean so they're somewhat separable.
        center = rng.randn(n_features) * 0.5 + i
        X_class = center + rng.randn(n_per_class, n_features) * 0.8
        X_parts.append(X_class)
        y_parts.extend([label] * n_per_class)

    X = np.vstack(X_parts)
    y = np.array(y_parts)
    return X, y, feature_names


# ── train_svm ────────────────────────────────────────────────────────

class TestTrainSvm:
    def test_returns_train_result(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        assert isinstance(result, TrainResult)
        assert result.model_name == "SVM"

    def test_accuracy_above_random(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        random_baseline = 1.0 / len(set(y))
        assert result.accuracy > random_baseline

    def test_confusion_shape(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        n_classes = len(set(y))
        assert result.confusion.shape == (n_classes, n_classes)

    def test_labels_sorted(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        assert result.labels == sorted(set(y))


# ── train_random_forest ──────────────────────────────────────────────

class TestTrainRF:
    def test_returns_train_result(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_random_forest(X, y, names)
        assert isinstance(result, TrainResult)
        assert result.model_name == "RandomForest"

    def test_has_feature_importances(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_random_forest(X, y, names)
        assert hasattr(result.model, "feature_importances_")
        assert len(result.model.feature_importances_) == result.n_features_out


# ── save / load ──────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)

        with tempfile.TemporaryDirectory() as tmp:
            result.save(tmp)
            loaded = TrainResult.load(tmp)

        assert loaded.model_name == result.model_name
        assert loaded.accuracy == result.accuracy
        assert loaded.labels == result.labels
        assert loaded.n_features_in == result.n_features_in
        assert loaded.n_features_out == result.n_features_out

    def test_loaded_model_predicts(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)

        with tempfile.TemporaryDirectory() as tmp:
            result.save(tmp)
            loaded = TrainResult.load(tmp)

        X_norm = loaded.normalizer.transform(X[:5])
        preds = loaded.model.predict(X_norm)
        assert len(preds) == 5
        assert all(p in loaded.labels for p in preds)


# ── evaluate ─────────────────────────────────────────────────────────

class TestEvaluate:
    def test_per_class_metrics(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        df = per_class_metrics(result)
        assert len(df) == len(result.labels)
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns

    def test_misclassification_analysis(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        df = misclassification_analysis(result)
        assert "true" in df.columns
        assert "predicted" in df.columns
        # No diagonal entries (correct predictions).
        assert all(df["true"] != df["predicted"])

    def test_full_report_returns_string(self) -> None:
        X, y, names = _make_synthetic_data()
        result = train_svm(X, y, names)
        report = full_report(result, dataset_name="test_synthetic", save=False)
        assert "EVALUATION REPORT" in report
        assert "TEST_SYNTHETIC" in report.upper()


# ── predict ──────────────────────────────────────────────────────────

class TestPredictor:
    def test_from_saved(self) -> None:
        model_dir = Path("models/svm_ravdess")
        if not model_dir.exists():
            pytest.skip("No saved model")
        predictor = Predictor.from_saved(model_dir)
        assert predictor is not None

    def test_predict_ravdess_file(self) -> None:
        model_dir = Path("models/svm_ravdess")
        audio = Path("data/ravdess/Actor_01/03-01-05-01-01-01-01.wav")
        if not model_dir.exists() or not audio.exists():
            pytest.skip("Model or RAVDESS not available")

        predictor = Predictor.from_saved(model_dir)
        result = predictor.predict(audio)

        assert "emotion" in result
        assert "confidence" in result
        assert "ekman6_weights" in result
        assert "features" in result
        assert result["emotion"] in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_file_not_found(self) -> None:
        model_dir = Path("models/svm_ravdess")
        if not model_dir.exists():
            pytest.skip("No saved model")
        predictor = Predictor.from_saved(model_dir)
        with pytest.raises(FileNotFoundError):
            predictor.predict("/nonexistent.wav")
