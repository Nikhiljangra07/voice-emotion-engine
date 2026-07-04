"""Batch feature extraction for MSP-Podcast (the P2.3 dimensional study).

Unlike scripts/batch_extract.py (categorical datasets), this carries the native
V/A/D targets, primary-emotion label, speaker ID, and split through to disk so
the three V/A/D regressors (Layer 2) and the CentroidNamer (Layer 4) can train
on real dimensional annotations.

Outputs (under out/, all aligned row-for-row):
    features_msp_<split>.npy   (n, 111)  float64 feature matrix
    targets_msp_<split>.npy    (n, 3)    NATIVE 1-7 valence, arousal, dominance
    meta_msp_<split>.csv                 filename, emotion, speaker, split, V/A/D

Checkpoint/resume: a snapshot .npz is written every CHECKPOINT_EVERY files
(atomically). Re-running skips files already in the snapshot, so a crash mid-run
loses at most one checkpoint interval.

Usage:
    python -m scripts.extract_msp --split dev          # probe pass (~34k)
    python -m scripts.extract_msp --split train        # full (~169k, ~hours)
    python -m scripts.extract_msp --split dev --limit 200   # quick smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.dimensional.loader import DimensionalSample, load_msp_podcast
from src.dimensional.metrics import DIMENSIONS
from src.features.feature_vector import build_feature_vector, feature_names, to_array

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
MSP_ROOT = "data/msp_podcast"
CHECKPOINT_EVERY = 2500


def _ckpt_path(split: str) -> Path:
    return OUT_DIR / f"_ckpt_msp_{split}.npz"


def _load_checkpoint(split: str) -> dict:
    """Load a prior snapshot if present. Returns dict with done filenames + rows."""
    p = _ckpt_path(split)
    if not p.exists():
        return {"done": set(), "X": [], "Y": [], "meta": []}
    z = np.load(p, allow_pickle=True)
    meta = list(z["meta"])
    done = {m["filename"] for m in meta}
    print(f"  Resuming from checkpoint: {len(done)} files already extracted.")
    return {"done": done, "X": list(z["X"]), "Y": list(z["Y"]), "meta": meta}


def _save_checkpoint(split: str, X: list, Y: list, meta: list) -> None:
    """Atomically write the snapshot (temp + rename).

    NOTE: the temp name must end in ``.npz`` — ``np.savez`` silently appends
    ``.npz`` to any path that doesn't, which would break the rename target.
    """
    p = _ckpt_path(split)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(
        tmp,
        X=np.asarray(X, dtype=np.float64),
        Y=np.asarray(Y, dtype=np.float64),
        meta=np.asarray(meta, dtype=object),
    )
    tmp.replace(p)


def extract(samples: list[DimensionalSample], split: str) -> tuple:
    """Extract 111 features + carry V/A/D targets. Resumable via checkpoint."""
    ck = _load_checkpoint(split)
    done, X, Y, meta = ck["done"], ck["X"], ck["Y"], ck["meta"]
    errors: list[dict] = []

    todo = [s for s in samples if Path(s.path).name not in done]
    total = len(todo)
    print(f"Extracting {total} files (split={split}); {len(done)} already done.\n")

    start = time.time()
    last_report = start
    for i, s in enumerate(todo):
        fname = Path(s.path).name
        try:
            arr = to_array(build_feature_vector(s.path))
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan or n_inf:
                errors.append({"file": fname, "error": f"{n_nan} NaN, {n_inf} Inf"})
                continue
            X.append(arr)
            Y.append([s.valence, s.arousal, s.dominance])
            meta.append({
                "filename": fname,
                "emotion": s.emotion,
                "speaker": s.speaker,
                "split": s.split,
                "valence": s.valence,
                "arousal": s.arousal,
                "dominance": s.dominance,
            })
        except Exception as e:  # noqa: BLE001 - log & continue, no silent drop
            errors.append({"file": fname, "error": str(e)})

        now = time.time()
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(split, X, Y, meta)
        if now - last_report >= 10 or (i + 1) % 1000 == 0 or i + 1 == total:
            elapsed = now - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:>6d}/{total}] {elapsed:6.1f}s, "
                f"{rate:.1f} files/s, ETA {eta/60:.1f}min, "
                f"kept {len(X)}, errors {len(errors)}"
            )
            last_report = now

    _save_checkpoint(split, X, Y, meta)
    dt = time.time() - start
    print(f"\nDone in {dt/60:.1f} min. Kept {len(X)}, errors {len(errors)}.")
    if errors:
        print("  First 5 errors:")
        for e in errors[:5]:
            print(f"    {e['file']}: {e['error']}")
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        pd.DataFrame(meta),
        errors,
    )


def save(X: np.ndarray, Y: np.ndarray, meta: pd.DataFrame, split: str) -> None:
    np.save(OUT_DIR / f"features_msp_{split}.npy", X)
    np.save(OUT_DIR / f"targets_msp_{split}.npy", Y)
    meta.to_csv(OUT_DIR / f"meta_msp_{split}.csv", index=False)
    print(f"\nSaved features_msp_{split}.npy {X.shape}")
    print(f"Saved targets_msp_{split}.npy  {Y.shape} (native 1-7: {DIMENSIONS})")
    print(f"Saved meta_msp_{split}.csv     ({len(meta)} rows)")

    # Integrity + distribution summary.
    print(f"\n  Integrity: NaN={int(np.isnan(X).sum())}, Inf={int(np.isinf(X).sum())}")
    for j, dim in enumerate(DIMENSIONS):
        col = Y[:, j]
        print(f"  {dim:>9s}: mean {col.mean():.2f}, std {col.std():.2f}, "
              f"range [{col.min():.1f}, {col.max():.1f}]")
    print(f"\n  Emotion distribution:")
    for emo, n in meta["emotion"].value_counts().items():
        print(f"    {emo:>12s}: {n:>6d}")


def main() -> int:
    ap = argparse.ArgumentParser(description="MSP-Podcast feature extraction (P2.3)")
    ap.add_argument("--split", required=True,
                    help="train | dev | test1 | test2 (dev recommended first)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of files (smoke test)")
    ap.add_argument("--ekman6-only", action="store_true",
                    help="keep only Ekman-6 emotions (default: keep all for regression)")
    args = ap.parse_args()

    samples = load_msp_podcast(
        MSP_ROOT, split=args.split, ekman6_only=args.ekman6_only, require_vad=True
    )
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("No samples found.")
        return 1

    X, Y, meta, _ = extract(samples, args.split)
    if X.size == 0:
        print("No features extracted.")
        return 1
    save(X, Y, meta, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
