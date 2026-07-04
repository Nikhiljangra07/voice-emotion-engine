"""CHILD experiment (isolated .venv_diar): does adding learned audio-vectors help?

Tests the user's hypothesis: convert audio into LEARNED vectors (WavLM SSL
embeddings) and give the pipeline that view alongside our hand-crafted features.
Compares three configs on the SAME sampled clips & split (apples-to-apples):
  (a) classical  — our 111 hand-crafted features
  (b) ssl        — WavLM-base embeddings (mean-pooled, 768-d)
  (c) fused      — both concatenated
…for V/A/D regression (CCC, Ridge linear-probe) and emotion (balanced RF).

Runs in .venv_diar (torch+transformers); never imports the parent src. Embeddings
are cached to .npy so reruns are cheap.

Run: .venv_diar/bin/python -m scripts.exp_ssl_fusion --n-train 5000 --n-test 2000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT = Path("out")
AUDIO = Path("data/msp_podcast/Audios")
NAMED = {"anger", "disgust", "fear", "joy", "sadness", "surprise",
         "contempt", "neutral"}


def ccc(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    mt, mp, vt, vp = yt.mean(), yp.mean(), yt.var(), yp.var()
    cov = float(np.mean((yt - mt) * (yp - mp)))
    d = vt + vp + (mt - mp) ** 2
    return 0.0 if d == 0 else float(2 * cov / d)


def load_split(split):
    X = np.load(OUT / f"features_msp_{split}.npy")
    Y = np.load(OUT / f"targets_msp_{split}.npy")
    meta = pd.read_csv(OUT / f"meta_msp_{split}.csv")
    return X, Y, meta


def embed_clips(filenames, cache: Path):
    if cache.exists():
        print(f"  using cached embeddings {cache.name}")
        return np.load(cache)
    from transformers import AutoFeatureExtractor, AutoModel
    print("  loading WavLM-base...")
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = AutoModel.from_pretrained("microsoft/wavlm-base").eval()
    embs = np.zeros((len(filenames), 768), dtype=np.float32)
    for i, fn in enumerate(filenames):
        y, sr = sf.read(str(AUDIO / fn), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        inp = fe(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            embs[i] = model(**inp).last_hidden_state.mean(dim=1).squeeze().numpy()
        if (i + 1) % 200 == 0:
            print(f"    embedded {i+1}/{len(filenames)}")
    np.save(cache, embs)
    return embs


def vad_ccc(Xtr, Xte, Ytr, Yte, label, lines):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    cccs = []
    for i in range(3):
        m = Ridge(alpha=10.0).fit(Xtr_s, Ytr[:, i])
        cccs.append(ccc(Yte[:, i], m.predict(Xte_s)))
    lines.append(f"  {label:<10} V {cccs[0]:.3f}  A {cccs[1]:.3f}  D {cccs[2]:.3f}"
                 f"   mean {np.mean(cccs):.3f}")
    return cccs


def emo_acc(Xtr, Xte, etr, ete, label, lines):
    sc = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(sc.transform(Xtr), etr)
    pred = clf.predict(sc.transform(Xte))
    acc = float(np.mean(pred == ete))
    classes = sorted(set(ete))
    bal = float(np.mean([np.mean(pred[ete == c] == c) for c in classes]))
    lines.append(f"  {label:<10} acc {acc*100:5.1f}%   balanced {bal*100:5.1f}%")
    return acc, bal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    Xtr_all, Ytr_all, mtr = load_split("train")
    Xte_all, Yte_all, mte = load_split("test1")
    itr = rng.choice(len(Xtr_all), min(args.n_train, len(Xtr_all)), replace=False)
    ite = rng.choice(len(Xte_all), min(args.n_test, len(Xte_all)), replace=False)

    print("Extracting WavLM embeddings (train)...")
    Etr = embed_clips(mtr.iloc[itr]["filename"].tolist(),
                      OUT / f"_ssl_train_{len(itr)}.npy")
    print("Extracting WavLM embeddings (test1)...")
    Ete = embed_clips(mte.iloc[ite]["filename"].tolist(),
                      OUT / f"_ssl_test1_{len(ite)}.npy")

    Ctr, Cte = Xtr_all[itr], Xte_all[ite]          # classical features
    Ytr, Yte = Ytr_all[itr], Yte_all[ite]          # native V/A/D
    Ftr, Fte = np.hstack([Ctr, Etr]), np.hstack([Cte, Ete])  # fused

    lines = [f"SSL FUSION EXPERIMENT — train {len(itr)} / test1 {len(ite)} (held-out)",
             f"WavLM-base (768-d) vs 111 hand-crafted; Ridge linear-probe / balanced RF", ""]
    lines.append("V/A/D regression — CCC (higher better):")
    vad_ccc(Ctr, Cte, Ytr, Yte, "classical", lines)
    vad_ccc(Etr, Ete, Ytr, Yte, "ssl", lines)
    vad_ccc(Ftr, Fte, Ytr, Yte, "fused", lines)
    lines.append("")

    # Emotion (named subset only).
    etr = mtr.iloc[itr]["emotion"].astype(str).to_numpy()
    ete = mte.iloc[ite]["emotion"].astype(str).to_numpy()
    mtr_named = np.array([e in NAMED for e in etr])
    mte_named = np.array([e in NAMED for e in ete])
    maj = Counter(ete[mte_named]).most_common(1)[0]
    lines.append(f"Emotion classification (named subset; majority "
                 f"{maj[1]/mte_named.sum()*100:.0f}% '{maj[0]}'):")
    emo_acc(Ctr[mtr_named], Cte[mte_named], etr[mtr_named], ete[mte_named],
            "classical", lines)
    emo_acc(Etr[mtr_named], Ete[mte_named], etr[mtr_named], ete[mte_named],
            "ssl", lines)
    emo_acc(Ftr[mtr_named], Fte[mte_named], etr[mtr_named], ete[mte_named],
            "fused", lines)

    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "ssl_fusion_result.txt").write_text(text)
    print(f"\nSaved → out/ssl_fusion_result.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
