"""Cross-dataset feature stability analysis.

Computes per-feature mean/std across RAVDESS, CREMA-D, and MELD,
flags features with >30% drift between datasets, and saves a report.

A feature that changes 50% between datasets is a Layer 1 problem —
the classifier can't generalize on it regardless of training strategy.

Usage:
    python -m scripts.feature_stability
    python -m scripts.feature_stability --samples 500
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_vector import build_feature_vector, feature_names, to_array
from src.utils.dataset_loader import load_ravdess, load_crema_d, load_meld


DRIFT_THRESHOLD = 0.30  # 30% relative difference = flagged


def _sample_and_extract(
    samples: list,
    n: int,
    dataset_name: str,
) -> np.ndarray:
    """Extract features from a random sample of files.

    Args:
        samples: List of AudioSample objects.
        n: Number of files to sample.
        dataset_name: Name for progress reporting.

    Returns:
        Feature matrix (n_extracted, n_features).
    """
    random.seed(42)
    selected = random.sample(samples, min(n, len(samples)))

    features_list: list[np.ndarray] = []
    errors = 0
    for i, s in enumerate(selected):
        try:
            fv = build_feature_vector(s.path)
            features_list.append(to_array(fv))
        except Exception:
            errors += 1
        if (i + 1) % 100 == 0:
            print(f"  {dataset_name}: {i + 1}/{len(selected)}...")

    if errors:
        print(f"  {dataset_name}: {errors} extraction errors (skipped)")

    return np.vstack(features_list)


def _relative_diff(a: float, b: float) -> float:
    """Relative difference between two values, using their mean as base."""
    mean = (abs(a) + abs(b)) / 2.0
    if mean < 1e-9:
        return 0.0
    return abs(a - b) / mean


def run_stability_analysis(n_samples: int = 200) -> pd.DataFrame:
    """Run cross-dataset feature stability analysis.

    Args:
        n_samples: Number of files to sample per dataset.

    Returns:
        DataFrame with per-feature stability metrics.
    """
    names = feature_names()

    # Load dataset samples
    print("Loading dataset metadata...")
    ravdess = load_ravdess()
    crema_d = load_crema_d()
    meld = load_meld()

    print(f"Sampling {n_samples} files per dataset...")
    print()

    # Extract features
    t0 = time.time()
    X_rav = _sample_and_extract(ravdess, n_samples, "RAVDESS")
    X_cre = _sample_and_extract(crema_d, n_samples, "CREMA-D")
    X_mel = _sample_and_extract(meld, n_samples, "MELD")
    elapsed = time.time() - t0
    print(f"\nExtracted {X_rav.shape[0] + X_cre.shape[0] + X_mel.shape[0]} "
          f"files in {elapsed:.0f}s")

    # Compute per-feature stats
    rows = []
    for i, name in enumerate(names):
        rav_mean = float(np.mean(X_rav[:, i]))
        rav_std = float(np.std(X_rav[:, i]))
        cre_mean = float(np.mean(X_cre[:, i]))
        cre_std = float(np.std(X_cre[:, i]))
        mel_mean = float(np.mean(X_mel[:, i]))
        mel_std = float(np.std(X_mel[:, i]))

        # Pairwise relative differences
        drift_rav_cre = _relative_diff(rav_mean, cre_mean)
        drift_rav_mel = _relative_diff(rav_mean, mel_mean)
        drift_cre_mel = _relative_diff(cre_mean, mel_mean)
        max_drift = max(drift_rav_cre, drift_rav_mel, drift_cre_mel)

        rows.append({
            "feature": name,
            "ravdess_mean": rav_mean,
            "ravdess_std": rav_std,
            "crema_d_mean": cre_mean,
            "crema_d_std": cre_std,
            "meld_mean": mel_mean,
            "meld_std": mel_std,
            "drift_rav_cre": drift_rav_cre,
            "drift_rav_mel": drift_rav_mel,
            "drift_cre_mel": drift_cre_mel,
            "max_drift": max_drift,
            "stable": max_drift <= DRIFT_THRESHOLD,
        })

    df = pd.DataFrame(rows)
    return df


def print_report(df: pd.DataFrame) -> None:
    """Print a formatted stability report."""
    n_total = len(df)
    n_stable = int(df["stable"].sum())
    n_unstable = n_total - n_stable

    print()
    print("=" * 80)
    print("CROSS-DATASET FEATURE STABILITY REPORT")
    print("=" * 80)
    print(f"\nFeatures: {n_total}")
    print(f"Stable (drift ≤ {DRIFT_THRESHOLD:.0%}): {n_stable} ({n_stable/n_total*100:.0f}%)")
    print(f"Unstable (drift > {DRIFT_THRESHOLD:.0%}): {n_unstable} ({n_unstable/n_total*100:.0f}%)")

    if n_unstable > 0:
        print(f"\n{'─' * 80}")
        print("UNSTABLE FEATURES (drift > 30%)")
        print(f"{'─' * 80}")
        unstable = df[~df["stable"]].sort_values("max_drift", ascending=False)
        print(f"\n{'Feature':<45} {'RAVDESS':>9} {'CREMA-D':>9} {'MELD':>9} {'MaxDrift':>9}")
        print("-" * 85)
        for _, row in unstable.iterrows():
            print(
                f"{row['feature']:<45} "
                f"{row['ravdess_mean']:>9.3f} "
                f"{row['crema_d_mean']:>9.3f} "
                f"{row['meld_mean']:>9.3f} "
                f"{row['max_drift']*100:>8.1f}%"
            )

    print(f"\n{'─' * 80}")
    print("MOST STABLE FEATURES (top 10)")
    print(f"{'─' * 80}")
    stable_top = df[df["stable"]].nsmallest(10, "max_drift")
    print(f"\n{'Feature':<45} {'RAVDESS':>9} {'CREMA-D':>9} {'MELD':>9} {'MaxDrift':>9}")
    print("-" * 85)
    for _, row in stable_top.iterrows():
        print(
            f"{row['feature']:<45} "
            f"{row['ravdess_mean']:>9.3f} "
            f"{row['crema_d_mean']:>9.3f} "
            f"{row['meld_mean']:>9.3f} "
            f"{row['max_drift']*100:>8.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-dataset feature stability analysis",
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Number of files to sample per dataset (default: 200)",
    )
    args = parser.parse_args()

    df = run_stability_analysis(n_samples=args.samples)

    print_report(df)

    # Save
    out_path = Path("out/feature_stability.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
