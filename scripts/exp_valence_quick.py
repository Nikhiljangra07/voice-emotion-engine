"""Quick valence-lever check on MSP Dev — before committing to the 6h Train run.

Two cheap, legitimate levers (no re-extraction, no deep learning):
  1. SVR (the model classical valence baselines use) vs Ridge/RF.
  2. Output calibration: rescale predictions to match the target's mean/std,
     estimated on TRAIN ONLY (no leakage). Given fixed correlation, this is the
     CCC-maximizing affine — it fixes the "predictions hedge to the mean" gap
     (our RF valence: Pearson 0.22 but CCC 0.11).

Same speaker-independent split as study_msp.py (seed 0). SVR is fit on a random
train subsample for speed (RBF SVR is ~O(n^2)).

Usage: python -m scripts.exp_valence_quick --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.dimensional.metrics import DIMENSIONS, ccc, pearson
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import feature_names

OUT_DIR = Path("out")
SVR_SUBSAMPLE = 8000


def _calibrate(p_train, y_train, p_test):
    """CCC-optimal affine (match train mean/std). Fit on train, apply to test."""
    a = y_train.std() / (p_train.std() + 1e-9)
    b = y_train.mean() - a * p_train.mean()
    return a * p_test + b


def _row(kind, Xtr, Ytr, Xte, Yte, names):
    reg = DimensionalRegressor(model=kind).fit(Xtr, Ytr, names)
    ptr, pte = reg.predict(Xtr), reg.predict(Xte)
    out = {}
    for i, dim in enumerate(DIMENSIONS):
        raw = ccc(Yte[:, i], pte[:, i])
        cal = ccc(Yte[:, i], _calibrate(ptr[:, i], Ytr[:, i], pte[:, i]))
        r = pearson(Yte[:, i], pte[:, i])
        out[dim] = (raw, cal, r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    X = np.load(OUT_DIR / f"features_msp_{args.split}.npy")
    Y = np.load(OUT_DIR / f"targets_msp_{args.split}.npy")
    meta = pd.read_csv(OUT_DIR / f"meta_msp_{args.split}.csv")
    speakers = meta["speaker"].astype(str).to_numpy()
    names = feature_names()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(np.zeros(len(speakers)), groups=speakers))

    rng = np.random.RandomState(0)
    sub = rng.choice(tr, size=min(SVR_SUBSAMPLE, len(tr)), replace=False)

    lines = [f"VALENCE QUICK CHECK — split={args.split}, n={len(X)}",
             f"train {len(tr)} / test {len(te)} (speaker-independent); "
             f"SVR on {len(sub)}-row subsample", "",
             "CCC raw = as-is | CCC cal = output-calibrated | r = Pearson", ""]

    configs = [("ridge", tr), ("rf", tr), ("svr", sub)]
    for kind, train_idx in configs:
        res = _row(kind, X[train_idx], Y[train_idx], X[te], Y[te], names)
        lines.append(f"=== {kind} (train n={len(train_idx)}) ===")
        lines.append(f"  {'dim':>10s}  {'CCC raw':>8s}  {'CCC cal':>8s}  {'r':>6s}")
        for dim in DIMENSIONS:
            raw, cal, r = res[dim]
            lines.append(f"  {dim:>10s}  {raw:>8.3f}  {cal:>8.3f}  {r:>6.3f}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    (OUT_DIR / f"exp_valence_quick_{args.split}.txt").write_text(text)
    print(f"Saved -> {OUT_DIR / f'exp_valence_quick_{args.split}.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
