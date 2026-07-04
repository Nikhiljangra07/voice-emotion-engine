"""Batch feature extraction for RAVDESS and/or CREMA-D.

Extracts 104-feature vectors for all files in a dataset, saves as
CSV (with labels) and NPY (feature matrix only).

Usage:
    python -m scripts.batch_extract --dataset crema_d
    python -m scripts.batch_extract --dataset ravdess
    python -m scripts.batch_extract --dataset all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_vector import (
    build_feature_vector,
    feature_names,
    to_array,
)
from src.utils.dataset_loader import (
    AudioSample,
    load_all,
    load_crema_d,
    load_meld,
    load_ravdess,
    to_dataframe,
)

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


def extract_batch(
    samples: list[AudioSample],
    dataset_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract features for all samples.

    Returns:
        (X, meta_df) where X is (n_samples, 104) and meta_df has
        path, label, dataset, actor, intensity columns.
    """
    names = feature_names()
    rows: list[np.ndarray] = []
    meta_rows: list[dict] = []
    errors: list[dict] = []

    total = len(samples)
    start = time.time()
    last_report = start

    print(f"Extracting {total} files ({dataset_name})...")
    print()

    for i, sample in enumerate(samples):
        try:
            features = build_feature_vector(sample.path)
            arr = to_array(features)

            # Integrity check — no NaN, no Inf.
            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            if nan_count > 0 or inf_count > 0:
                errors.append({
                    "file": sample.path,
                    "error": f"{nan_count} NaN, {inf_count} Inf",
                })
                continue

            rows.append(arr)
            meta_rows.append({
                "path": sample.path,
                "label": sample.label,
                "dataset": sample.dataset,
                "actor": sample.actor,
                "intensity": sample.intensity,
            })

        except Exception as e:
            errors.append({"file": sample.path, "error": str(e)})

        # Progress every 10 seconds or every 500 files.
        now = time.time()
        if now - last_report >= 10 or (i + 1) % 500 == 0 or i + 1 == total:
            elapsed = now - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:>6d}/{total}] "
                f"{elapsed:6.1f}s elapsed, "
                f"{rate:.1f} files/s, "
                f"ETA {eta:.0f}s, "
                f"errors: {len(errors)}"
            )
            last_report = now

    total_time = time.time() - start
    X = np.array(rows, dtype=np.float64)

    print()
    print(f"Done in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Extracted: {len(rows)}/{total}")
    print(f"  Errors: {len(errors)}")
    print(f"  Feature matrix: {X.shape}")

    if errors:
        print(f"\n  First 5 errors:")
        for e in errors[:5]:
            print(f"    {Path(e['file']).name}: {e['error']}")

    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df


def save_results(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Save feature matrix and metadata."""
    names = feature_names()

    # NPY — raw feature matrix.
    npy_path = OUT_DIR / f"features_{dataset_name}.npy"
    np.save(npy_path, X)
    print(f"  Saved: {npy_path} ({X.shape})")

    # CSV — features + metadata.
    feature_df = pd.DataFrame(X, columns=names)
    full_df = pd.concat([meta_df.reset_index(drop=True), feature_df], axis=1)
    csv_path = OUT_DIR / f"features_{dataset_name}.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(full_df)} rows x {len(full_df.columns)} cols)")

    # Labels NPY — for quick classifier loading.
    labels_path = OUT_DIR / f"labels_{dataset_name}.npy"
    np.save(labels_path, meta_df["label"].values)
    print(f"  Saved: {labels_path}")

    # Summary.
    print(f"\n  Label distribution:")
    for label, count in meta_df["label"].value_counts().sort_index().items():
        print(f"    {label:>10s}: {count:>5d}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch feature extraction")
    parser.add_argument(
        "--dataset",
        choices=["ravdess", "crema_d", "meld", "all"],
        required=True,
        help="Which dataset to extract",
    )
    args = parser.parse_args()

    if args.dataset == "ravdess":
        samples = load_ravdess()
    elif args.dataset == "crema_d":
        samples = load_crema_d()
    elif args.dataset == "meld":
        samples = load_meld()
    else:
        samples = load_all()

    if not samples:
        print("No samples found.")
        return 1

    X, meta_df = extract_batch(samples, args.dataset)

    if X.size == 0:
        print("No features extracted.")
        return 1

    save_results(X, meta_df, args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
