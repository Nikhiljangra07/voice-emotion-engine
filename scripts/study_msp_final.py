"""P2.3 DEFINITIVE study — canonical MSP protocol: train on Train, test on Dev.

Unlike study_msp.py (which does an internal speaker-split of one split), this
uses the corpus's own partitions: fit on the full Train split (169k), evaluate
on the held-out Dev split (34k). Speakers do not overlap across MSP partitions,
so this is speaker-independent by construction and benchmark-comparable.

Layer 2: three V/A/D regressors (Ridge, RF, SVR), all output-calibrated.
Layer 4: CentroidNamer fit on Train named-emotion points, separability on Dev.

Usage: python -m scripts.study_msp_final
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.dimensional.loader import normalize_vad_msp
from src.dimensional.metrics import DIMENSIONS
from src.dimensional.namer import CentroidNamer
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import feature_names

OUT_DIR = Path("out")
MODELS_DIR = Path("models")
SVR_TRAIN_CAP = 25000  # more data than the Dev probe (8k); RBF-SVR is ~O(n^2)
NAMED_EMOTIONS = {
    "anger", "disgust", "fear", "joy", "sadness", "surprise",
    "contempt", "neutral",
}


def _load(split: str):
    X = np.load(OUT_DIR / f"features_msp_{split}.npy")
    Y = np.load(OUT_DIR / f"targets_msp_{split}.npy")
    meta = pd.read_csv(OUT_DIR / f"meta_msp_{split}.csv")
    return X, Y, meta


def regression(Xtr, Ytr, Xte, Yte, report):
    report.append("=" * 64)
    report.append("LAYER 2 — V/A/D REGRESSORS  (train=Train 169k → test=Dev 34k, CCC)")
    report.append("=" * 64)
    names = feature_names()
    rng = np.random.RandomState(0)
    best = None
    for kind in ("ridge", "rf", "svr"):
        if kind == "svr" and len(Xtr) > SVR_TRAIN_CAP:
            idx = rng.choice(len(Xtr), size=SVR_TRAIN_CAP, replace=False)
            xt, yt = Xtr[idx], Ytr[idx]
        else:
            xt, yt = Xtr, Ytr
        reg = DimensionalRegressor(model=kind, calibrate=True).fit(xt, yt, names)
        rep = reg.evaluate(Xte, Yte)
        report.append(f"--- {kind} (calibrated, train n={len(xt)}) ---")
        report.append(f"  {'dim':>10s}  {'CCC':>7s}  {'RMSE':>7s}  {'Pearson':>7s}")
        for dim in DIMENSIONS:
            r = rep[dim]
            report.append(f"  {dim:>10s}  {r['ccc']:>7.3f}  {r['rmse']:>7.3f}  "
                          f"{r['pearson']:>7.3f}")
        report.append(f"  {'mean CCC':>10s}  {rep['mean']['ccc']:>7.3f}")
        report.append("")
        # Rank by arousal+dominance+valence CCC sum but keep valence visible.
        score = rep["mean"]["ccc"]
        if best is None or score > best[2]:
            best = (kind, reg, score, rep)
    kind, reg, _, rep = best
    reg.save(MODELS_DIR / f"dim_{kind}_msp_final")
    report.append(f"Best (mean CCC): {kind} → saved models/dim_{kind}_msp_final")
    report.append("")
    return rep


def naming(Ytr, emo_tr, Yte, emo_te, report):
    report.append("=" * 64)
    report.append("LAYER 4 — POINT→EMOTION SEPARABILITY (gold set: 6+contempt+neutral)")
    report.append("=" * 64)
    mtr = np.array([e in NAMED_EMOTIONS for e in emo_tr])
    mte = np.array([e in NAMED_EMOTIONS for e in emo_te])
    Ptr = np.array([normalize_vad_msp(*r) for r in Ytr[mtr]])
    Pte = np.array([normalize_vad_msp(*r) for r in Yte[mte]])
    etr = np.asarray(emo_tr)[mtr]
    ete = np.asarray(emo_te)[mte]
    report.append(f"Train named rows: {mtr.sum()} | Dev named rows: {mte.sum()}")

    namer = CentroidNamer().fit(Ptr, list(etr))
    correct = 0
    confusion: dict[tuple[str, str], int] = {}
    for p, true in zip(Pte, ete):
        pred = namer.predict(p)["emotion"]
        correct += int(pred == true)
        confusion[(true, pred)] = confusion.get((true, pred), 0) + 1
    acc = correct / len(ete)
    report.append(f"Separability (true V/A/D → emotion): {acc:.1%} "
                  f"(chance {1/len(namer.labels):.1%})")
    report.append("")
    report.append("Data centroids (normalized PAD plane, fit on Train):")
    report.append(f"  {'emotion':>9s}  {'V':>6s}  {'A':>6s}  {'D':>6s}")
    for e in sorted(namer.labels):
        c = namer._centroids[e]
        report.append(f"  {e:>9s}  {c[0]:>6.2f}  {c[1]:>6.2f}  {c[2]:>6.2f}")
    report.append("")
    report.append("Top confusions (true → pred: count):")
    for (t, p), n in sorted(
        ((k, v) for k, v in confusion.items() if k[0] != k[1]),
        key=lambda kv: kv[1], reverse=True,
    )[:8]:
        report.append(f"  {t:>9s} → {p:<9s}: {n}")
    report.append("")
    namer.save(MODELS_DIR / "namer_msp_final")
    report.append("Namer saved → models/namer_msp_final")
    report.append("")


def main() -> int:
    for split in ("train", "dev"):
        if not (OUT_DIR / f"features_msp_{split}.npy").exists():
            print(f"Missing features_msp_{split}.npy")
            return 1
    Xtr, Ytr, mtr = _load("train")
    Xte, Yte, mte = _load("dev")
    print(f"Train {Xtr.shape}  Dev {Xte.shape}")
    print(f"Integrity: Train NaN={int(np.isnan(Xtr).sum())} "
          f"Dev NaN={int(np.isnan(Xte).sum())}\n")

    report = [f"MSP-PODCAST DEFINITIVE STUDY — train=Train({len(Xtr)}) test=Dev({len(Xte)})",
              ""]
    regression(Xtr, Ytr, Xte, Yte, report)
    naming(Ytr, mtr["emotion"].astype(str).to_numpy(),
           Yte, mte["emotion"].astype(str).to_numpy(), report)

    text = "\n".join(report)
    print(text)
    (OUT_DIR / "study_msp_final.txt").write_text(text)
    print("\nReport saved → out/study_msp_final.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
