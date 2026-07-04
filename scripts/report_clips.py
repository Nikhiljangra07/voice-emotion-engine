"""Clip-by-clip honesty report: true emotion vs what each method names.

Uses the cached embeddings + leave-one-speaker-out (each clip is judged as if its
speaker were unseen — no self-match, no same-speaker cheating). Shows, per clip:
true family | emotion2vec-kNN | WavLM-head | HYBRID | correct?  — plus overall and
per-family accuracy, and a look at the wrong ones.

Run:  .venv_diar/bin/python -m scripts.report_clips [--per-family 4]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FAM = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
FI = {f: i for i, f in enumerate(FAM)}
SHORT = {"anger": "anger", "fear": "fear", "joy": "joy", "neutral": "neutrl",
         "sadness": "sad", "surprise": "surpris"}


def knn_proba(S_row, y, valid_idx, k=5):
    top = valid_idx[np.argsort(-S_row[valid_idx])[:k]]
    p = np.zeros(len(FAM))
    for i in top:
        p[FI[y[i]]] += max(float(S_row[i]), 0.0)
    return p / (p.sum() or 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=4)
    ap.add_argument("--speaker", default=None, help="filter sample table by speaker")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAM]
    y = np.array([e["label"] for e in labeled])
    spk = np.array([e["speaker"] for e in labeled])
    files = [e["file"] for e in labeled]
    n = len(y)

    Xe = np.load(root / "out/_ownvoice_e2v_emb.npy")
    Xw = np.load(root / "out/_ownvoice_ft_emb.npy")
    Xen = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-8)
    Se = Xen @ Xen.T

    Pe = np.zeros((n, len(FAM))); Pw = np.zeros((n, len(FAM)))
    for ts in np.unique(spk):
        te = np.where(spk == ts)[0]; tr = np.where(spk != ts)[0]
        for i in te:
            Pe[i] = knn_proba(Se[i], y, tr)
        sc = StandardScaler().fit(Xw[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(Xw[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(Xw[te]))
        col = [list(clf.classes_).index(f) for f in FAM]
        Pw[te] = proba[:, col]
    Ph = 0.5 * Pe + 0.5 * Pw

    def lab(P, i):
        return FAM[int(P[i].argmax())]

    # ── sample table (balanced across families + speakers) ──
    print("Clip-by-clip (leave-one-speaker-out; ✓=hybrid correct)\n")
    print(f"{'clip':<20}{'speaker':<10}{'TRUE':<9}{'e2v':<8}{'wavlm':<8}"
          f"{'HYBRID':<8}{'conf':<6} ok")
    print("-" * 74)
    idx_by_fam = defaultdict(list)
    for i in range(n):
        if args.speaker and spk[i] != args.speaker:
            continue
        idx_by_fam[y[i]].append(i)
    shown = []
    for fam in FAM:
        picks = idx_by_fam[fam][: args.per_family]
        for i in picks:
            h = lab(Ph, i); ok = "✓" if h == y[i] else "✗"
            conf = Ph[i].max()
            print(f"{files[i][:19]:<20}{spk[i].replace('female_','F-')[:9]:<10}"
                  f"{SHORT[y[i]]:<9}{SHORT[lab(Pe,i)]:<8}{SHORT[lab(Pw,i)]:<8}"
                  f"{SHORT[h]:<8}{conf:<6.0%} {ok}")
            shown.append(i)

    # ── accuracy (FULL set, not just the sample) ──
    def acc(P):
        return np.mean([lab(P, i) == y[i] for i in range(n)])
    print("\n" + "=" * 74)
    print(f"FULL-SET accuracy (all {n} clips, cross-speaker):")
    print(f"  emotion2vec {acc(Pe):.1%} | WavLM {acc(Pw):.1%} | HYBRID {acc(Ph):.1%}")
    print("\nHybrid per-family recall:")
    for fam in FAM:
        m = y == fam
        preds = [lab(Ph, i) for i in np.where(m)[0]]
        r = np.mean(np.array(preds) == fam)
        conf = ", ".join(f"{a}:{b}" for a, b in Counter(preds).most_common(3))
        print(f"  {fam:9} {r:5.1%} (n={m.sum():2d})  -> {conf}")

    # per-speaker hybrid accuracy
    print("\nHybrid accuracy by speaker:")
    for s in np.unique(spk):
        mi = np.where(spk == s)[0]
        a = np.mean([lab(Ph, i) == y[i] for i in mi])
        print(f"  {s:12} {a:.1%} (n={len(mi)})")

    # ── full 175-row CSV dump ──
    import csv
    out = root / "out/family_clip_report.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "speaker", "true", "emotion2vec", "wavlm",
                    "hybrid", "hybrid_conf", "correct"])
        for i in range(n):
            h = lab(Ph, i)
            w.writerow([files[i], spk[i], y[i], lab(Pe, i), lab(Pw, i), h,
                        f"{Ph[i].max():.3f}", int(h == y[i])])
    print(f"\nfull 175-row table -> {out}")


if __name__ == "__main__":
    main()
