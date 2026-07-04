"""Does a TRAINED HEAD beat plain kNN on the same frozen vectors? (local proxy)

The #2 question — "is it worth spending GPU to train a head on emotion2vec?" — tested
cheaply and locally: on our 175 own-voice clips, compare
  (a) kNN retrieval        (what the adaptor does now)
  (b) a trained linear head (logistic regression)
on the SAME frozen embeddings, under leave-one-speaker-out (the honest test).

If the head clearly beats kNN even on this tiny data, a head trained on MSP-scale
labels is very likely worth the GPU. If they tie, it's inconclusive at this scale
(MSP's volume could still help) — reported honestly either way.

Run:  .venv_diar/bin/python -m scripts.exp_head_vs_knn
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FAMILIES = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]


def knn_loso(Xn, y, spk, k=5):
    S = Xn @ Xn.T
    pred = []
    for i in range(len(y)):
        valid = np.where(spk != spk[i])[0]
        top = valid[np.argsort(-S[i][valid])[:k]]
        pred.append(Counter(y[top]).most_common(1)[0][0])
    return np.array(pred)


def head_loso(X, y, spk):
    """Train logistic head on one speaker, predict the other; both directions."""
    pred = np.empty(len(y), dtype=object)
    for test_spk in np.unique(spk):
        tr = spk != test_spk
        te = spk == test_spk
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred


def show(tag, y, pred):
    acc = np.mean(pred == y)
    print(f"\n[{tag}]  cross-speaker accuracy = {acc:.1%}")
    for fam in FAMILIES:
        m = y == fam
        print(f"    {fam:9} {np.mean(pred[m]==fam):5.1%}")
    return acc


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAMILIES]
    y = np.array([e["label"] for e in labeled])
    spk = np.array([e["speaker"] for e in labeled])

    for name, cache in [("emotion2vec", "out/_ownvoice_e2v_emb.npy"),
                        ("WavLM-ft", "out/_ownvoice_ft_emb.npy")]:
        X = np.load(root / cache)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        print("\n" + "=" * 56)
        print(f"### {name} embeddings ###")
        a_knn = show("kNN retrieval (k=5)", y, knn_loso(Xn, y, spk))
        a_head = show("trained head (logreg, balanced)", y, head_loso(X, y, spk))
        delta = a_head - a_knn
        verdict = ("HEAD WINS -> a trained head helps; MSP head likely worth GPU"
                   if delta > 0.03 else
                   "TIE/kNN -> head no better at this scale; MSP scale may still help"
                   if delta > -0.03 else
                   "kNN WINS -> retrieval already better here")
        print(f"\n  >>> {name}: head {a_head:.1%} vs kNN {a_knn:.1%} "
              f"(Δ={delta:+.1%}) — {verdict}")


if __name__ == "__main__":
    main()
