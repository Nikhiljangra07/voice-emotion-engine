"""Head-to-head: emotion2vec vs fine-tuned-WavLM vectors for family kNN.

Same clips, same protocols (leave-one-clip-out + leave-one-speaker-out), same
metric. The only variable is the VECTORIZER:
  * WavLM (fine-tuned) — general model we taught emotion to  [cached]
  * emotion2vec (+large) — model pre-trained specifically for emotion

Answers one question honestly: does an emotion-specialized vectorizer give a
better retrieval database than our fine-tuned general one, especially across
strangers (the weak spot)?

Run:  .venv_diar/bin/python -m scripts.exp_emotion2vec
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

FAMILIES = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]


def embed_emotion2vec(root, labeled):
    cache = root / "out/_ownvoice_e2v_emb.npy"
    meta = root / "out/_ownvoice_e2v_emb.meta.json"
    files = [e["file"] for e in labeled]
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == files:
        print("emotion2vec: using cached embeddings")
        return np.load(cache)
    import logging
    logging.disable(logging.WARNING)
    import setuptools  # noqa: F401  (activates its bundled distutils on py3.13)
    from funasr import AutoModel
    m = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True,
                  disable_pbar=True)
    print(f"emotion2vec: embedding {len(labeled)} clips (CPU) ...")
    X = np.zeros((len(labeled), 1024), dtype=np.float32)
    for i, e in enumerate(labeled):
        rec = m.generate(str(root / "own_voice" / e["file"]),
                         granularity="utterance", extract_embedding=True)
        X[i] = np.asarray(rec[0]["feats"], dtype=np.float32).ravel()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(labeled)}")
    np.save(cache, X); meta.write_text(json.dumps(files))
    return X


def knn(sims, labels, valid, k):
    idx = np.where(valid)[0]
    top = idx[np.argsort(-sims[idx])[:k]]
    return Counter(labels[top]).most_common(1)[0][0]


def evaluate(X, y, spk, tag):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    n = len(y)
    out = {}
    # leave-one-clip-out
    for k in (1, 3, 5):
        pred = np.array([knn(S[i], y, _mask_not(i, n), k) for i in range(n)])
        out[f"loo_k{k}"] = float(np.mean(pred == y))
    # leave-one-speaker-out
    for k in (1, 3, 5):
        pred = np.array([knn(S[i], y, spk != spk[i], k) for i in range(n)])
        out[f"loso_k{k}"] = float(np.mean(pred == y))
        if k == 5:
            out["loso_pred_k5"] = pred
    print(f"\n[{tag}]")
    print("  leave-one-clip-out :  " + "  ".join(
        f"k{k}={out[f'loo_k{k}']:.1%}" for k in (1, 3, 5)))
    print("  leave-one-spkr-out :  " + "  ".join(
        f"k{k}={out[f'loso_k{k}']:.1%}" for k in (1, 3, 5)))
    pred = out["loso_pred_k5"]
    print("  per-family recall (cross-speaker, k=5):")
    for fam in FAMILIES:
        m = y == fam
        r = np.mean(pred[m] == fam)
        conf = ", ".join(f"{a}:{b}" for a, b in Counter(pred[m]).most_common(2))
        print(f"    {fam:9} {r:5.1%}  -> {conf}")
    return out


def _mask_not(i, n):
    v = np.ones(n, bool); v[i] = False; return v


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAMILIES]
    y = np.array([e["label"] for e in labeled])
    spk = np.array([e["speaker"] for e in labeled])

    # WavLM cached vectors (must match labeled order)
    wav_meta = json.loads((root / "out/_ownvoice_ft_emb.meta.json").read_text())
    assert wav_meta == [e["file"] for e in labeled], "WavLM cache order mismatch"
    Xw = np.load(root / "out/_ownvoice_ft_emb.npy")
    Xe = embed_emotion2vec(root, labeled)

    print("\n" + "=" * 64)
    print("HEAD-TO-HEAD  (family kNN, 175 clips, 2 speakers)")
    rw = evaluate(Xw, y, spk, "WavLM (fine-tuned)")
    re = evaluate(Xe, y, spk, "emotion2vec (+large, frozen)")

    print("\n" + "=" * 64)
    print("SUMMARY (higher = better)")
    print(f"{'protocol':26}{'WavLM-ft':>12}{'emotion2vec':>14}")
    for k, lab in [("loo_k5", "in-domain (k5)"),
                   ("loso_k5", "cross-speaker (k5)")]:
        print(f"{lab:26}{rw[k]:>11.1%}{re[k]:>14.1%}")


if __name__ == "__main__":
    main()
