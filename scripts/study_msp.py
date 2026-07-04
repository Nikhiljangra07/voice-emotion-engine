"""P2.3 dimensional study on MSP-Podcast features.

Runs the first HONEST read of whether the 111-feature engine carries
dimensional signal:

  1. Train the three V/A/D regressors (Layer 2), SPEAKER-INDEPENDENT split
     (no speaker in both train & test), report CCC/RMSE/Pearson per dimension.
  2. Fit the CentroidNamer (Layer 4) on data centroids of the Ekman-6 subset
     (normalized PAD plane), report point->emotion separability (the documented
     overlap "ceiling", Law-7 honesty).

Targets are kept in NATIVE 1-7 for the regression CCC (benchmark-comparable);
normalized to the PAD plane only for the namer/centroids.

Usage (after scripts.extract_msp --split dev):
    python -m scripts.study_msp --split dev
    python -m scripts.study_msp --split dev --models ridge rf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.dimensional.loader import normalize_vad_msp
from src.dimensional.metrics import DIMENSIONS
from src.dimensional.namer import CentroidNamer
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import feature_names

OUT_DIR = Path("out")
MODELS_DIR = Path("models")
# RBF-SVR is ~O(n^2); cap its train rows (standard practice, matches the
# published classical baselines). Ridge/RF use the full train set.
SVR_TRAIN_CAP = 15000

# The namer's class set is the GOLD STANDARD's own emotions (TRAJECTORY_ENGINE
# Law 19): the 6 + contempt + neutral. We drop only Other / no-agreement
# (annotation catch-alls, not emotions). These still feed the regressor (Layer 2
# is taxonomy-free); they are excluded only from the centroid namer (Layer 4).
NAMED_EMOTIONS = {
    "anger", "disgust", "fear", "joy", "sadness", "surprise",
    "contempt", "neutral",
}
_NOT_EMOTIONS = {"other", "no_agreement"}


def _speaker_split(speakers: np.ndarray, test_size: float = 0.2, seed: int = 0):
    """Speaker-independent train/test indices (no speaker in both)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(speakers)), groups=speakers))
    return train_idx, test_idx


def run_regression(X, Y, speakers, models, split, report):
    report.append("=" * 64)
    report.append("LAYER 2 — THREE V/A/D REGRESSORS (native 1-7, CCC)")
    report.append("=" * 64)
    tr, te = _speaker_split(speakers)
    n_tr_spk = len(set(speakers[tr])); n_te_spk = len(set(speakers[te]))
    overlap = set(speakers[tr]) & set(speakers[te])
    report.append(f"Speaker-independent split: train {len(tr)} ({n_tr_spk} spk) / "
                  f"test {len(te)} ({n_te_spk} spk); overlap={len(overlap)}")
    report.append("")

    names = feature_names()
    rng = np.random.RandomState(0)
    best = None
    for kind in models:
        # SVR trains on a capped random subsample; Ridge/RF on the full train.
        if kind == "svr" and len(tr) > SVR_TRAIN_CAP:
            fit_idx = rng.choice(tr, size=SVR_TRAIN_CAP, replace=False)
        else:
            fit_idx = tr
        reg = DimensionalRegressor(model=kind, calibrate=True).fit(
            X[fit_idx], Y[fit_idx], names
        )
        rep = reg.evaluate(X[te], Y[te])
        report.append(f"--- model: {kind} (calibrated, train n={len(fit_idx)}) ---")
        report.append(f"  {'dim':>10s}  {'CCC':>7s}  {'RMSE':>7s}  {'Pearson':>7s}")
        for dim in DIMENSIONS:
            r = rep[dim]
            report.append(f"  {dim:>10s}  {r['ccc']:>7.3f}  {r['rmse']:>7.3f}  "
                          f"{r['pearson']:>7.3f}")
        report.append(f"  {'mean CCC':>10s}  {rep['mean']['ccc']:>7.3f}")
        report.append("")
        if best is None or rep["mean"]["ccc"] > best[2]:
            best = (kind, reg, rep["mean"]["ccc"])

    kind, reg, mean_ccc = best
    save_path = MODELS_DIR / f"dim_{kind}_msp_{split}"
    reg.save(save_path)
    report.append(f"Best model: {kind} (mean CCC {mean_ccc:.3f}) -> saved {save_path}")
    report.append("")
    return tr, te


def run_namer(Y_native, emotions, speakers, split, report):
    report.append("=" * 64)
    report.append("LAYER 4 — POINT -> EMOTION SEPARABILITY (gold set: 6+contempt+neutral)")
    report.append("=" * 64)
    mask = np.array([e in NAMED_EMOTIONS for e in emotions])
    Yn = np.array([normalize_vad_msp(*row) for row in Y_native[mask]])
    emo = np.asarray(emotions)[mask]
    spk = speakers[mask]
    report.append(f"Named-emotion rows: {mask.sum()} / {len(emotions)} "
                  f"(dropped Other/no-agreement)")

    tr, te = _speaker_split(spk)
    namer = CentroidNamer().fit(Yn[tr], list(emo[tr]))

    # Nearest-centroid accuracy on held-out TRUE points (separability of the
    # PAD space itself — independent of the acoustic regressor).
    correct = 0
    confusion: dict[tuple[str, str], int] = {}
    for p, true in zip(Yn[te], emo[te]):
        pred = namer.predict(p)["emotion"]
        if pred == true:
            correct += 1
        confusion[(true, pred)] = confusion.get((true, pred), 0) + 1
    acc = correct / len(te)

    report.append(f"Centroid separability (true V/A/D -> emotion): {acc:.1%} "
                  f"(chance {1/len(namer.labels):.1%})")
    report.append("")
    report.append("Data centroids (normalized PAD plane):")
    report.append(f"  {'emotion':>9s}  {'V':>6s}  {'A':>6s}  {'D':>6s}")
    for e in sorted(namer.labels):
        c = namer._centroids[e]
        report.append(f"  {e:>9s}  {c[0]:>6.2f}  {c[1]:>6.2f}  {c[2]:>6.2f}")
    report.append("")
    report.append("Top confusions (true -> pred: count):")
    conf_sorted = sorted(
        ((k, v) for k, v in confusion.items() if k[0] != k[1]),
        key=lambda kv: kv[1], reverse=True,
    )
    for (t, p), n in conf_sorted[:8]:
        report.append(f"  {t:>9s} -> {p:<9s}: {n}")
    report.append("")
    namer.save(MODELS_DIR / f"namer_msp_{split}")
    report.append(f"Namer saved -> {MODELS_DIR / f'namer_msp_{split}'}")
    report.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description="MSP-Podcast dimensional study (P2.3)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--models", nargs="+", default=["ridge", "rf", "svr"],
                    choices=["ridge", "rf", "svr"])
    args = ap.parse_args()

    fX = OUT_DIR / f"features_msp_{args.split}.npy"
    fY = OUT_DIR / f"targets_msp_{args.split}.npy"
    fM = OUT_DIR / f"meta_msp_{args.split}.csv"
    for f in (fX, fY, fM):
        if not f.exists():
            print(f"Missing {f}. Run: python -m scripts.extract_msp --split {args.split}")
            return 1

    X = np.load(fX)
    Y = np.load(fY)
    meta = pd.read_csv(fM)
    speakers = meta["speaker"].astype(str).to_numpy()
    emotions = meta["emotion"].astype(str).to_numpy()
    print(f"Loaded {X.shape} features, {Y.shape} targets, {len(meta)} meta rows.")
    print(f"Integrity: NaN={int(np.isnan(X).sum())}, Inf={int(np.isinf(X).sum())}\n")

    report: list[str] = []
    report.append(f"MSP-PODCAST DIMENSIONAL STUDY — split={args.split}, n={len(X)}")
    report.append("")
    run_regression(X, Y, speakers, args.models, args.split, report)
    run_namer(Y, emotions, speakers, args.split, report)

    text = "\n".join(report)
    print(text)
    out = OUT_DIR / f"study_msp_{args.split}.txt"
    out.write_text(text)
    print(f"\nReport saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
