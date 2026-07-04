"""Evaluation utilities for trained classifiers.

Generates confusion matrix plots, per-class metrics, and summary
reports. Saves all artifacts to out/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.classifier.train import TrainResult

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(
    result: TrainResult,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
    normalize: bool = True,
) -> Path:
    """Plot and save a confusion matrix.

    Args:
        result: TrainResult from training.
        title: Plot title.
        filename: Output filename (saved to out/).
        normalize: If True, show percentages per row (recall).

    Returns:
        Path to saved image.
    """
    cm = result.confusion
    labels = result.labels

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt = ".0%"
        title_suffix = " (Recall %)"
    else:
        cm_display = cm.astype(float)
        fmt = ".0f"
        title_suffix = " (Counts)"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{title}{title_suffix}", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells.
    thresh = cm_display.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_display[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight="bold" if i == j else "normal",
            )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()

    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def per_class_metrics(result: TrainResult) -> pd.DataFrame:
    """Build a DataFrame of per-class metrics.

    Returns:
        DataFrame with columns: emotion, precision, recall, f1,
        support, accuracy.
    """
    labels = result.labels
    cm = result.confusion
    rows: list[dict] = []

    for i, label in enumerate(labels):
        tp = cm[i, i]
        support = cm[i].sum()
        recall = tp / support if support > 0 else 0.0

        col_sum = cm[:, i].sum()
        precision = tp / col_sum if col_sum > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append({
            "emotion": label,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": int(support),
            "correct": int(tp),
        })

    return pd.DataFrame(rows)


def misclassification_analysis(result: TrainResult) -> pd.DataFrame:
    """Identify the top confusion pairs.

    Returns:
        DataFrame with columns: true, predicted, count, pct_of_true,
        sorted by count descending. Excludes correct predictions.
    """
    cm = result.confusion
    labels = result.labels
    rows: list[dict] = []

    for i, true_label in enumerate(labels):
        row_total = cm[i].sum()
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = cm[i, j]
            if count > 0:
                rows.append({
                    "true": true_label,
                    "predicted": pred_label,
                    "count": int(count),
                    "pct_of_true": round(count / row_total * 100, 1),
                })

    if not rows:
        return pd.DataFrame(columns=["true", "predicted", "count", "pct_of_true"])
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    return df.reset_index(drop=True)


def full_report(
    result: TrainResult,
    dataset_name: str = "dataset",
    save: bool = True,
) -> str:
    """Generate and optionally save a complete evaluation report.

    Args:
        result: TrainResult from training.
        dataset_name: Name for filenames and titles.
        save: If True, save plots and CSV to out/.

    Returns:
        Formatted report string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"EVALUATION REPORT — {dataset_name.upper()}")
    lines.append("=" * 70)

    # Overall.
    lines.append(f"\nOverall Accuracy:     {result.accuracy * 100:.2f}%")
    lines.append(f"CV Accuracy (5-fold): {result.cv_mean * 100:.2f}% ± {result.cv_std * 100:.2f}%")
    lines.append(f"Random Baseline:      {100 / len(result.labels):.1f}%")
    lines.append(f"Improvement:          +{result.accuracy * 100 - 100 / len(result.labels):.1f}pp")
    lines.append(f"Train/Test:           {result.train_size}/{result.test_size}")
    lines.append(f"Features:             {result.n_features_in} → {result.n_features_out}")

    # Per-class.
    lines.append("\nPer-Class Metrics:")
    metrics_df = per_class_metrics(result)
    lines.append(metrics_df.to_string(index=False))

    # Misclassifications.
    lines.append("\nTop Misclassification Pairs:")
    mis_df = misclassification_analysis(result)
    lines.append(mis_df.head(10).to_string(index=False))

    # sklearn report.
    lines.append("\nFull Classification Report:")
    lines.append(result.report)

    report_str = "\n".join(lines)

    if save:
        # Confusion matrix plots.
        cm_norm_path = plot_confusion_matrix(
            result,
            title=f"SVM — {dataset_name}",
            filename=f"confusion_matrix_{dataset_name}_normalized.png",
            normalize=True,
        )
        cm_count_path = plot_confusion_matrix(
            result,
            title=f"SVM — {dataset_name}",
            filename=f"confusion_matrix_{dataset_name}_counts.png",
            normalize=False,
        )

        # Save metrics CSV.
        metrics_path = OUT_DIR / f"metrics_{dataset_name}.csv"
        metrics_df.to_csv(metrics_path, index=False)

        # Save misclassification CSV.
        mis_path = OUT_DIR / f"misclassifications_{dataset_name}.csv"
        mis_df.to_csv(mis_path, index=False)

        # Save report text.
        report_path = OUT_DIR / f"report_{dataset_name}.txt"
        report_path.write_text(report_str)

        lines.append(f"\nSaved:")
        lines.append(f"  {cm_norm_path}")
        lines.append(f"  {cm_count_path}")
        lines.append(f"  {metrics_path}")
        lines.append(f"  {mis_path}")
        lines.append(f"  {report_path}")
        report_str = "\n".join(lines)

    return report_str
