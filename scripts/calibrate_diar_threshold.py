"""Calibrate the cosine-distance threshold for neural auto speaker-counting.

Runs in the isolated .venv_diar. Embeds the controlled conversations once, then
sweeps the cosine threshold and reports which value best recovers the TRUE
speaker counts (data-driven — not a guessed magic number). Use the printed best
threshold as --threshold for scripts.diarize_neural.

Run: .venv_diar/bin/python -m scripts.calibrate_diar_threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from scripts.diarize_neural import load_audio, window_bounds

CONVO_DIR = Path("own_voice/test_convos")
WIN, HOP = 2.0, 1.0


def main() -> int:
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=".venv_diar/ecapa_model", run_opts={"device": "cpu"})

    convos = sorted(CONVO_DIR.glob("*.json"))
    data = []
    for jf in convos:
        truth = json.loads(jf.read_text())
        y, sr = load_audio(str(jf.with_suffix(".wav")))
        bounds = window_bounds(len(y), sr, WIN, HOP)
        embs = np.zeros((len(bounds), 192), dtype=np.float32)
        for i, (s, e) in enumerate(bounds):
            with torch.no_grad():
                embs[i] = enc.encode_batch(
                    torch.tensor(y[s:e]).unsqueeze(0)).squeeze().cpu().numpy()
        data.append((truth["name"], truth["n_speakers"], embs))
        print(f"embedded {truth['name']} (true_k={truth['n_speakers']})")

    print(f"\n{'threshold':>10}  correct_k/{len(data)}  detected_per_convo")
    best_T, best_correct = None, -1
    for T in np.arange(0.40, 1.25, 0.05):
        detected, correct = [], 0
        for _, true_k, embs in data:
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=float(T),
                metric="cosine", linkage="average").fit_predict(embs)
            k = len(set(labels))
            detected.append(k)
            correct += int(k == true_k)
        marker = ""
        if correct > best_correct:
            best_correct, best_T, marker = correct, float(T), "  <= best"
        print(f"{T:>10.2f}  {correct:>11}  {detected}{marker}")

    print(f"\nBest threshold: {best_T:.2f} ({best_correct}/{len(data)} correct)")
    print(f"true_k per convo: {[d[1] for d in data]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
