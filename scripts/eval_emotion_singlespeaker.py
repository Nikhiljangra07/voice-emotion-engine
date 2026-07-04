"""End-to-end single-speaker emotion accuracy on labelled MSP clips.

Runs the FULL pipeline that the trajectory uses — features → SVR V/A/D → namer →
emotion — on many single-speaker clips with known labels, and reports accuracy
HONESTLY against the right baselines (MSP is imbalanced, so raw accuracy alone is
misleading):
  - overall accuracy (8-class gold set: 6 + contempt + neutral)
  - majority-class baseline  &  random chance
  - per-class recall + top confusions
  - Ekman-6-only accuracy (comparable to Phase 1)

Usage: python -m scripts.eval_emotion_singlespeaker --split dev
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.dimensional.loader import normalize_vad_msp
from src.dimensional.namer import CentroidNamer
from src.dimensional.regressors import DimensionalRegressor

OUT = Path("out")
NAMED = {"anger", "disgust", "fear", "joy", "sadness", "surprise",
         "contempt", "neutral"}
EKMAN6 = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--regressor", default="models/dim_svr_msp_final")
    ap.add_argument("--namer", default="models/namer_msp_final")
    args = ap.parse_args()

    X = np.load(OUT / f"features_msp_{args.split}.npy")
    meta = pd.read_csv(OUT / f"meta_msp_{args.split}.csv")
    true = meta["emotion"].astype(str).to_numpy()
    reg = DimensionalRegressor.load(args.regressor)
    namer = CentroidNamer.load(args.namer)

    # End-to-end: predicted native V/A/D → clip to scale → PAD → emotion.
    vad = np.clip(reg.predict(X), 1.0, 7.0)
    pad = np.array([normalize_vad_msp(*r) for r in vad])
    pred = np.array([namer.predict(p)["emotion"] for p in pad])

    # Evaluate only on rows whose TRUE label is a nameable emotion.
    mask = np.array([t in NAMED for t in true])
    yt, yp = true[mask], pred[mask]
    n = len(yt)
    acc = float(np.mean(yt == yp))
    maj_label, maj_n = Counter(yt).most_common(1)[0]
    maj_base = maj_n / n

    lines = [f"SINGLE-SPEAKER EMOTION ACCURACY — split={args.split}, n={n}",
             f"(end-to-end: audio → SVR V/A/D → namer → emotion)", ""]
    lines.append(f"8-class accuracy:        {acc*100:5.1f}%")
    lines.append(f"  majority baseline:     {maj_base*100:5.1f}%  (always '{maj_label}')")
    lines.append(f"  random chance:         {100/8:5.1f}%")
    lines.append(f"  lift over majority:    {(acc-maj_base)*100:+5.1f} pts")
    lines.append("")

    # Per-class recall.
    lines.append("Per-class recall:")
    lines.append(f"  {'emotion':>9}  {'recall':>7}  {'support':>7}")
    for emo in sorted(set(yt)):
        idx = yt == emo
        r = float(np.mean(yp[idx] == emo))
        lines.append(f"  {emo:>9}  {r*100:6.1f}%  {int(idx.sum()):>7}")
    lines.append("")

    # Top confusions.
    conf = Counter((t, p) for t, p in zip(yt, yp) if t != p)
    lines.append("Top confusions (true → pred: count):")
    for (t, p), c in conf.most_common(8):
        lines.append(f"  {t:>9} → {p:<9}: {c}")
    lines.append("")

    # Ekman-6 subset (drop neutral/contempt from BOTH the truth filter and preds).
    m6 = np.array([t in EKMAN6 for t in true])
    yt6, yp6 = true[m6], pred[m6]
    # Map any neutral/contempt predictions to "other6" so they count as wrong.
    yp6 = np.array([p if p in EKMAN6 else "(non-ekman)" for p in yp6])
    acc6 = float(np.mean(yt6 == yp6))
    maj6 = Counter(yt6).most_common(1)[0][1] / len(yt6)
    lines.append(f"Ekman-6-only accuracy:   {acc6*100:5.1f}%  "
                 f"(majority {maj6*100:.1f}%, chance {100/6:.1f}%)")

    text = "\n".join(lines)
    print(text)
    (OUT / f"emotion_accuracy_{args.split}.txt").write_text(text)
    print(f"\nSaved → out/emotion_accuracy_{args.split}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
