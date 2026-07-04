"""Classifier layer (Layer 2 — swappable, dataset-specific).

Public API:
    from src.classifier import train_svm, TrainResult
    from src.classifier import full_report, plot_confusion_matrix, per_class_metrics
"""

from src.classifier.train import train_svm, train_random_forest, tune_svm, tune_random_forest, TrainResult, TuneResult
from src.classifier.evaluate import (
    full_report,
    misclassification_analysis,
    per_class_metrics,
    plot_confusion_matrix,
)

__all__ = [
    "train_svm",
    "train_random_forest",
    "tune_svm",
    "tune_random_forest",
    "TrainResult",
    "TuneResult",
    "full_report",
    "misclassification_analysis",
    "per_class_metrics",
    "plot_confusion_matrix",
]
