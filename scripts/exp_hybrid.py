"""Hybrid: two backbones back each other up at the DECISION level (not vector level).

Vectors stay separate (different spaces — never mixed). Each backbone produces a
probability over the 6 families; we AVERAGE those probabilities. This is the safe,
correct realization of "use both corpuses together":
  * emotion2vec  -> kNN distribution        (its strong config: neutral, balanced)
  * WavLM-ft     -> trained logistic head    (its strong config: fear/joy/sadness)

Protocol: leave-one-speaker-out (honest). Reports each alone vs the hybrid, plus a
small weight sweep (exploratory — tuned on the same tiny set, so read as directional).

Run:  .venv_diar/bin/python -m scripts.exp_hybrid
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FAM = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
FI = {f: i for i, f in enumerate(FAM)}


def knn_proba_row(S_row, y, valid_idx, k):
    top = valid_idx[np.argsort(-S_row[valid_idx])[:k]]
    p = np.zeros(len(FAM))
    for i in top:
        p[FI[y[i]]] += max(float(S_row[i]), 0.0)
    return p / (p.sum() or 1.0)


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAM]
    y = np.array([e["label"] for e in labeled])
    spk = np.array([e["speaker"] for e in labeled])
    n = len(y)

    Xe = np.load(root / "out/_ownvoice_e2v_emb.npy")
    Xw = np.load(root / "out/_ownvoice_ft_emb.npy")
    Xen = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-8)
    Se = Xen @ Xen.T

    # per-clip probability vectors from each backbone, under leave-one-speaker-out
    Pe = np.zeros((n, len(FAM)))   # emotion2vec kNN
    Pw = np.zeros((n, len(FAM)))   # WavLM head
    for ts in np.unique(spk):
        te = np.where(spk == ts)[0]
        tr_idx = np.where(spk != ts)[0]
        # emotion2vec kNN
        for i in te:
            Pe[i] = knn_proba_row(Se[i], y, tr_idx, k=5)
        # WavLM trained head
        sc = StandardScaler().fit(Xw[tr_idx])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(Xw[tr_idx]), y[tr_idx])
        proba = clf.predict_proba(sc.transform(Xw[te]))
        col = [list(clf.classes_).index(f) for f in FAM]
        Pw[te] = proba[:, col]

    def acc(P):
        return np.mean(np.array([FAM[i] for i in P.argmax(1)]) == y)

    a_e, a_w = acc(Pe), acc(Pw)
    print("=" * 56)
    print(f"emotion2vec (kNN)     : {a_e:.1%}")
    print(f"WavLM-ft (head)       : {a_w:.1%}")

    # equal-weight hybrid
    a_h = acc(0.5 * Pe + 0.5 * Pw)
    print(f"HYBRID (equal weight) : {a_h:.1%}")

    # weight sweep (exploratory)
    print("\nweight sweep (w = emotion2vec share):")
    best_w, best_a = 0.5, a_h
    for w in np.round(np.arange(0.1, 0.95, 0.1), 2):
        a = acc(w * Pe + (1 - w) * Pw)
        mark = "  <-- best" if a > best_a else ""
        if a > best_a:
            best_a, best_w = a, w
        print(f"  w={w:.1f}: {a:.1%}{mark}")

    # per-family for the best hybrid
    Pbest = best_w * Pe + (1 - best_w) * Pw
    pred = np.array([FAM[i] for i in Pbest.argmax(1)])
    print(f"\nBest hybrid (w={best_w:.1f} → {best_a:.1%}) per-family recall:")
    for f in FAM:
        m = y == f
        print(f"  {f:9} {np.mean(pred[m]==f):5.1%}  "
              f"(e2v {np.mean(np.array([FAM[i] for i in Pe[m].argmax(1)])==f):.0%}"
              f" / wav {np.mean(np.array([FAM[i] for i in Pw[m].argmax(1)])==f):.0%})")

    print("\n" + "=" * 56)
    print(f"SUMMARY  emotion2vec {a_e:.1%} | WavLM {a_w:.1%} | "
          f"hybrid(eq) {a_h:.1%} | hybrid(best w={best_w:.1f}) {best_a:.1%}")


if __name__ == "__main__":
    main()
