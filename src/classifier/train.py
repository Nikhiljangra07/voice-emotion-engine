"""Train classifiers on feature matrices.

Supports SVM and Random Forest with stratified train/test split.
All thresholds are learned from data — no magic numbers.

Usage:
    from src.classifier.train import train_svm, train_random_forest
    result_svm = train_svm(X, y, feature_names)
    result_rf = train_random_forest(X, y, feature_names)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

from src.features.normalize import FeatureNormalizer


@dataclass
class TrainResult:
    """Results from a training run."""

    model: Any
    model_name: str
    normalizer: FeatureNormalizer
    accuracy: float
    cv_mean: float
    cv_std: float
    confusion: np.ndarray
    report: str
    labels: list[str]
    train_size: int
    test_size: int
    n_features_in: int
    n_features_out: int
    y_test: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)

    def save(self, path: str | Path) -> None:
        """Save trained model, normalizer, and metadata.

        Args:
            path: Directory to save into (will be created).
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Model
        joblib.dump(self.model, save_dir / "model.joblib")

        # Normalizer
        self.normalizer.save(save_dir / "normalizer")

        # Metadata
        meta = {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "labels": self.labels,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "report": self.report,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainResult":
        """Load a saved TrainResult.

        Args:
            path: Directory containing saved files.

        Returns:
            TrainResult with model and normalizer restored.
            Note: y_test and y_pred are empty (not saved).
        """
        load_dir = Path(path)

        model = joblib.load(load_dir / "model.joblib")
        normalizer = FeatureNormalizer.load(load_dir / "normalizer")

        with open(load_dir / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            model=model,
            model_name=meta["model_name"],
            normalizer=normalizer,
            accuracy=meta["accuracy"],
            cv_mean=meta["cv_mean"],
            cv_std=meta["cv_std"],
            confusion=np.array([]),  # not saved
            report=meta["report"],
            labels=meta["labels"],
            train_size=meta["train_size"],
            test_size=meta["test_size"],
            n_features_in=meta["n_features_in"],
            n_features_out=meta["n_features_out"],
            y_test=np.array([]),
            y_pred=np.array([]),
        )


def _train_classifier(
    model: Any,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float,
    random_state: int,
    cv_folds: int,
) -> TrainResult:
    """Shared training pipeline for any sklearn classifier.

    Pipeline:
        1. Stratified train/test split.
        2. Fit normalizer on TRAIN only (no data leakage).
        3. Transform both train and test.
        4. Train model on normalized train.
        5. Cross-validate on train.
        6. Evaluate on held-out test.
    """
    # ── 1. Stratified split ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ── 2. Fit normalizer on TRAIN only ──────────────────────────
    normalizer = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
    X_train_norm = normalizer.fit_transform(X_train, feature_names)
    X_test_norm = normalizer.transform(X_test)

    # ── 3. Train ────────────────────────────────────────────────
    model.fit(X_train_norm, y_train)

    # ── 4. Cross-validate on train ──────────────────────────────
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    cv_scores = cross_val_score(
        model, X_train_norm, y_train,
        cv=cv, scoring="accuracy",
    )

    # ── 5. Evaluate on test ─────────────────────────────────────
    y_pred = model.predict(X_test_norm)
    accuracy = accuracy_score(y_test, y_pred)

    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=labels,
        digits=3,
    )

    return TrainResult(
        model=model,
        model_name=model_name,
        normalizer=normalizer,
        accuracy=accuracy,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        confusion=cm,
        report=report,
        labels=labels,
        train_size=len(y_train),
        test_size=len(y_test),
        n_features_in=X.shape[1],
        n_features_out=X_train_norm.shape[1],
        y_test=y_test,
        y_pred=y_pred,
    )


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
    cv_folds: int = 5,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
) -> TrainResult:
    """Train an SVM classifier with stratified split.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Label array (n_samples,).
        feature_names: Ordered feature names.
        test_size: Fraction for test split.
        random_state: Seed for reproducibility.
        cv_folds: Number of cross-validation folds.
        kernel: SVM kernel type.
        C: Regularization parameter.
        gamma: Kernel coefficient.

    Returns:
        TrainResult with model, metrics, and normalizer.
    """
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state,
        probability=True,
    )
    return _train_classifier(
        svm, "SVM", X, y, feature_names,
        test_size, random_state, cv_folds,
    )


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
    cv_folds: int = 5,
    n_estimators: int = 500,
    max_depth: int | None = None,
) -> TrainResult:
    """Train a Random Forest classifier with stratified split.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Label array (n_samples,).
        feature_names: Ordered feature names.
        test_size: Fraction for test split.
        random_state: Seed for reproducibility.
        cv_folds: Number of cross-validation folds.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (None = unlimited).

    Returns:
        TrainResult with model, metrics, and normalizer.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    return _train_classifier(
        rf, "RandomForest", X, y, feature_names,
        test_size, random_state, cv_folds,
    )


@dataclass
class TuneResult:
    """Results from hyperparameter grid search."""

    best_params: dict[str, Any]
    best_cv_score: float
    all_results: list[dict[str, Any]]
    train_result: TrainResult


def tune_svm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
    cv_folds: int = 5,
) -> TuneResult:
    """Grid search over SVM hyperparameters.

    Searches: kernel (rbf, linear), C (0.1, 1, 10, 100), gamma (scale, auto).
    Trains final model with best params on same split.

    Returns:
        TuneResult with best params and a TrainResult trained with those params.
    """
    # Split first — grid search on train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Normalize train
    normalizer = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
    X_train_norm = normalizer.fit_transform(X_train, feature_names)

    param_grid = {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        SVC(random_state=random_state, probability=True),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_norm, y_train)

    # Collect all results
    all_results = []
    for i in range(len(grid.cv_results_["params"])):
        all_results.append({
            "params": grid.cv_results_["params"][i],
            "mean_score": float(grid.cv_results_["mean_test_score"][i]),
            "std_score": float(grid.cv_results_["std_test_score"][i]),
            "rank": int(grid.cv_results_["rank_test_score"][i]),
        })
    all_results.sort(key=lambda r: r["rank"])

    # Train final model with best params using the standard pipeline
    best = grid.best_params_
    train_result = train_svm(
        X, y, feature_names,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        kernel=best["kernel"],
        C=best["C"],
        gamma=best["gamma"],
    )

    return TuneResult(
        best_params=best,
        best_cv_score=float(grid.best_score_),
        all_results=all_results,
        train_result=train_result,
    )


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
    cv_folds: int = 5,
) -> TuneResult:
    """Grid search over Random Forest hyperparameters.

    Searches: n_estimators (200, 500, 1000), max_depth (None, 20, 40),
    min_samples_split (2, 5).

    Returns:
        TuneResult with best params and a TrainResult trained with those params.
    """
    # Split first — grid search on train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Normalize train
    normalizer = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
    X_train_norm = normalizer.fit_transform(X_train, feature_names)

    param_grid = {
        "n_estimators": [200, 500, 1000],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,  # RF already parallelizes internally
    )
    grid.fit(X_train_norm, y_train)

    # Collect all results
    all_results = []
    for i in range(len(grid.cv_results_["params"])):
        all_results.append({
            "params": grid.cv_results_["params"][i],
            "mean_score": float(grid.cv_results_["mean_test_score"][i]),
            "std_score": float(grid.cv_results_["std_test_score"][i]),
            "rank": int(grid.cv_results_["rank_test_score"][i]),
        })
    all_results.sort(key=lambda r: r["rank"])

    # Train final model with best params
    best = grid.best_params_
    train_result = train_random_forest(
        X, y, feature_names,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
    )

    return TuneResult(
        best_params=best,
        best_cv_score=float(grid.best_score_),
        all_results=all_results,
        train_result=train_result,
    )
