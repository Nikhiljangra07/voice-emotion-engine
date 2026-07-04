"""Experiment: does per-speaker normalization claw back valence CCC?

Evidence (study_msp_dev): valence CCC 0.06–0.11 — below the classical eGeMAPS
baseline (~0.2–0.3). Hypothesis (Law 8): a speaker's identity (timbre, pitch
range) swamps the faint valence signal. Centering each speaker on their OWN
baseline — calibration, not leakage; we touch only features, never targets —
should remove that confound.

This re-runs the EXACT speaker-independent split from study_msp.py, with one
change: features are per-speaker mean-centered first. Each speaker is centered
on their own clips (within their own rows), which is valid at inference time
(collect a speaker's audio, subtract their mean = SpeakerBaseline premise).

No re-extraction, no engine change. Reports baseline vs centered side by side.

Usage:
    python -m scripts.exp_speaker_norm --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.dimensional.metrics import DIMENSIONS
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import feature_names

OUT_DIR = Path("out")


def _speaker_split(speakers, test_size=0.2, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(np.zeros(len(speakers)), groups=speakers))


def _per_speaker_center(X, speakers):
    """Subtract each speaker's own mean feature vector from their rows.

    Speaker means are computed within-speaker only (no cross-speaker mixing),
    so this is calibration, not train/test leakage.
    """
    Xc = X.copy()
    for spk in np.unique(speakers):
        idx = speakers == spk
        Xc[idx] -= X[idx].mean(axis=0)
    return Xc


def _eval(X, Y, tr, te, names, kind):
    reg = DimensionalRegressor(model=kind).fit(X[tr], Y[tr], names)
    return reg.evaluate(X[te], Y[te])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--models", nargs="+", default=["ridge", "rf"])
    args = ap.parse_args()

    X = np.load(OUT_DIR / f"features_msp_{args.split}.npy")
    Y = np.load(OUT_DIR / f"targets_msp_{args.split}.npy")
    meta = pd.read_csv(OUT_DIR / f"meta_msp_{args.split}.csv")
    speakers = meta["speaker"].astype(str).to_numpy()
    names = feature_names()
    tr, te = _speaker_split(speakers)

    Xc = _per_speaker_center(X, speakers)

    lines = [f"PER-SPEAKER NORM EXPERIMENT — split={args.split}, n={len(X)}",
             f"train {len(tr)} / test {len(te)} (speaker-independent)", ""]
    for kind in args.models:
        base = _eval(X, Y, tr, te, names, kind)
        cent = _eval(Xc, Y, tr, te, names, kind)
        lines.append(f"=== model: {kind} ===")
        lines.append(f"  {'dim':>10s}  {'CCC base':>9s}  {'CCC cent':>9s}  {'Δ':>7s}")
        for dim in DIMENSIONS:
            b, c = base[dim]["ccc"], cent[dim]["ccc"]
            lines.append(f"  {dim:>10s}  {b:>9.3f}  {c:>9.3f}  {c-b:>+7.3f}")
        bm, cm = base["mean"]["ccc"], cent["mean"]["ccc"]
        lines.append(f"  {'mean':>10s}  {bm:>9.3f}  {cm:>9.3f}  {cm-bm:>+7.3f}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / f"exp_speaker_norm_{args.split}.txt"
    out.write_text(text)
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
