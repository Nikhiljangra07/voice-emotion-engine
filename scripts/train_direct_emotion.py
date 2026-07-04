"""Direct emotion classifier on MSP — the fair test of 'does voice classify emotion?'

Skips the V/A/D→centroid detour: trains features → emotion directly (RandomForest,
class_weight='balanced' to stop the imbalance starving rare emotions). Trains on
MSP Train, evaluates on a held-out split. Reports accuracy HONESTLY vs the
majority baseline and chance, plus BALANCED accuracy (mean per-class recall),
which is the fair metric on imbalanced data.

Usage: python -m scripts.train_direct_emotion --test dev
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_vector import feature_names
from src.features.normalize import FeatureNormalizer

OUT = Path("out")
MODELS = Path("models")
NAMED = {"anger", "disgust", "fear", "joy", "sadness", "surprise",
         "contempt", "neutral"}


def load(split: str):
    X = np.load(OUT / f"features_msp_{split}.npy")
    meta = pd.read_csv(OUT / f"meta_msp_{split}.csv")
    emo = meta["emotion"].astype(str).to_numpy()
    mask = np.array([e in NAMED for e in emo])
    return X[mask], emo[mask]


def report(yt, yp, title, lines):
    n = len(yt)
    acc = float(np.mean(yt == yp))
    maj_label, maj_n = Counter(yt).most_common(1)[0]
    classes = sorted(set(yt))
    recalls = {c: float(np.mean(yp[yt == c] == c)) for c in classes}
    bal = float(np.mean(list(recalls.values())))
    lines.append(f"=== {title} (n={n}) ===")
    lines.append(f"  accuracy:          {acc*100:5.1f}%")
    lines.append(f"  majority baseline: {maj_n/n*100:5.1f}%  (always '{maj_label}')")
    lines.append(f"  balanced accuracy: {bal*100:5.1f}%  (chance {100/len(classes):.1f}%)")
    lines.append(f"  lift over majority:{(acc - maj_n/n)*100:+5.1f} pts")
    lines.append("  per-class recall: " +
                 ", ".join(f"{c} {recalls[c]*100:.0f}%" for c in classes))
    lines.append("")
    return acc, bal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="dev")
    args = ap.parse_args()

    names = feature_names()
    print("Loading Train...")
    Xtr, ytr = load("train")
    print(f"Train named clips: {len(Xtr)}")

    norm = FeatureNormalizer()
    Xtr_n = norm.fit_transform(Xtr, names)
    print("Training RandomForest (balanced)...")
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=0, n_jobs=-1)
    clf.fit(Xtr_n, ytr)

    lines = ["DIRECT EMOTION CLASSIFIER (features → emotion, balanced RF)", ""]
    # Train (sanity — expect high) then the held-out test split.
    report(ytr, clf.predict(Xtr_n), "train (sanity)", lines)
    Xte, yte = load(args.test)
    acc, bal = report(yte, clf.predict(norm.transform(Xte)),
                      f"HELD-OUT: {args.test}", lines)

    text = "\n".join(lines)
    print(text)
    (OUT / f"direct_emotion_{args.test}.txt").write_text(text)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS / "direct_emotion_rf.joblib")
    norm.save(MODELS / "direct_emotion_norm")
    print(f"Saved → out/direct_emotion_{args.test}.txt + models/direct_emotion_rf.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
