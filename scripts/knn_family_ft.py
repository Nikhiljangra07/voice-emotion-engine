"""Shazam-style emotion family recognition: match audio VECTORS, don't squeeze to V/A/D.

Embeds each labeled clip with the fine-tuned WavLM (the 1024-dim pooled vector,
BEFORE the V/A/D head), then classifies by nearest-neighbor in that vector space
against a database of labeled vectors (kNN / prototype retrieval). This keeps the
full vector instead of squeezing to 3 numbers, and — because kNN is multi-modal —
can hold both "loud anger" and "cold anger" as separate exemplars.

Two honest protocols:
  * leave-one-clip-out  (in-domain; same speakers in DB — optimistic)
  * leave-one-speaker-out (DB = other speaker only — the real generalization test)

Compared against the centroid-namer baseline (47.4% on these clips).

Run:  .venv_diar/bin/python -m scripts.knn_family_ft
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.predict_wavlm_ft import WavLMRegressor, load_audio

FAMILIES = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]


def embed_all(root, labeled, device):
    cache = root / "out/_ownvoice_ft_emb.npy"
    meta = root / "out/_ownvoice_ft_emb.meta.json"
    files = [e["file"] for e in labeled]
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == files:
        print("using cached embeddings")
        return np.load(cache)
    model = WavLMRegressor(str(root / "models/wavlm_vad_ft")).to(device).eval()
    X = np.zeros((len(labeled), model.backbone.config.hidden_size), dtype=np.float32)
    print(f"embedding {len(labeled)} clips on {device} ...")
    for i, e in enumerate(labeled):
        y = load_audio(str(root / "own_voice" / e["file"]))
        wav = torch.from_numpy(y).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            h = model.backbone(wav, attention_mask=mask).last_hidden_state.mean(1)
        X[i] = h.float().cpu().numpy().ravel()
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(labeled)}")
    (root / "out").mkdir(exist_ok=True)
    np.save(cache, X); meta.write_text(json.dumps(files))
    return X


def knn_predict(sims_row, labels, mask_valid, k):
    idx = np.where(mask_valid)[0]
    order = idx[np.argsort(-sims_row[idx])[:k]]
    return Counter(labels[order]).most_common(1)[0][0]


def report(name, y_true, y_pred):
    n = len(y_true)
    acc = np.mean(y_true == y_pred)
    print(f"\n{name}: {acc:.1%}  (n={n})")
    for fam in FAMILIES:
        m = y_true == fam
        if m.sum():
            r = np.mean(y_pred[m] == fam)
            conf = Counter(y_pred[m]).most_common(2)
            print(f"  {fam:9} {r:5.1%} (n={m.sum():2d})  -> "
                  + ", ".join(f"{a}:{b}" for a, b in conf))
    return acc


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAMILIES]
    y = np.array([e["label"] for e in labeled])
    spk = np.array([e["speaker"] for e in labeled])
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    X = embed_all(root, labeled, device)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T  # cosine similarity matrix
    n = len(y)

    print("\n" + "=" * 62)
    print("Baseline (V/A/D -> centroid namer, measured earlier): 47.4%")

    # ── leave-one-clip-out (in-domain) ──
    for k in (1, 3, 5):
        pred = np.empty(n, dtype=object)
        for i in range(n):
            valid = np.ones(n, bool); valid[i] = False
            pred[i] = knn_predict(S[i], y, valid, k)
        acc = np.mean(pred == y)
        print(f"leave-one-clip-out  k={k}: {acc:.1%}")
        if k == 3:
            loo_pred = pred
    report("LEAVE-ONE-CLIP-OUT (k=3, in-domain, optimistic)", y, loo_pred)

    # ── leave-one-speaker-out (honest generalization) ──
    for k in (1, 3, 5):
        pred = np.empty(n, dtype=object)
        for i in range(n):
            valid = spk != spk[i]          # DB = the OTHER speaker only
            pred[i] = knn_predict(S[i], y, valid, k)
        acc = np.mean(pred == y)
        print(f"\nleave-one-speaker-out k={k}: {acc:.1%}")
        if k == 5:
            loso_pred = pred
    report("LEAVE-ONE-SPEAKER-OUT (k=5, cross-speaker, HONEST)", y, loso_pred)

    # confusion for the honest run
    print("\nConfusion — cross-speaker (rows=true, cols=pred):")
    print("true\\pred   " + "".join(f"{f[:4]:>6}" for f in FAMILIES))
    for t in FAMILIES:
        row = Counter(loso_pred[y == t])
        print(f"{t:9}  " + "".join(f"{row.get(p,0):>6}" for p in FAMILIES))


if __name__ == "__main__":
    main()
