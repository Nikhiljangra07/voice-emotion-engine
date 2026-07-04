"""WavLM embeddings for arbitrary audio files — isolated .venv_diar.

Used by the main-pipeline fused predictor for inference on new clips. Outputs one
mean-pooled 768-d embedding per input file, in input order.

Run: .venv_diar/bin/python -m scripts.embed_files --inputs a.wav b.wav --out e.npy
"""

from __future__ import annotations

import argparse
import sys

import librosa
import numpy as np
import soundfile as sf
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = AutoModel.from_pretrained("microsoft/wavlm-base").eval()

    embs = np.zeros((len(args.inputs), 768), dtype=np.float32)
    for i, fn in enumerate(args.inputs):
        y, sr = sf.read(fn, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        inp = fe(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            embs[i] = model(**inp).last_hidden_state.mean(dim=1).squeeze().numpy()
    np.save(args.out, embs)
    print(f"Saved {embs.shape} → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
