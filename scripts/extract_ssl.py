"""SSL (WavLM) embedding extractor — runs in the isolated .venv_diar.

Batch-extracts mean-pooled WavLM-base embeddings (768-d) for a sample of an MSP
split, aligned by row index to that split's classical feature matrix so the main
pipeline can fuse them. Standalone (no parent src imports). Caches embeddings +
the sampled indices so the main-venv trainer can select matching classical rows.

Run (isolated interpreter):
    .venv_diar/bin/python -m scripts.extract_ssl --split train --n 15000
    .venv_diar/bin/python -m scripts.extract_ssl --split test1 --n 0   # 0 = all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

OUT = Path("out")
AUDIO = Path("data/msp_podcast/Audios")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=0, help="sample size; 0 = all rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(OUT / f"meta_msp_{args.split}.csv")
    n_total = len(meta)
    if args.n and args.n < n_total:
        idx = np.random.RandomState(args.seed).choice(n_total, args.n, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_total)
    files = meta.iloc[idx]["filename"].tolist()
    print(f"{args.split}: extracting {len(files)} of {n_total} clips")

    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = AutoModel.from_pretrained("microsoft/wavlm-base").eval()

    embs = np.zeros((len(files), 768), dtype=np.float32)
    for i, fn in enumerate(files):
        y, sr = sf.read(str(AUDIO / fn), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        inp = fe(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            embs[i] = model(**inp).last_hidden_state.mean(dim=1).squeeze().numpy()
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)}")

    np.save(OUT / f"ssl_{args.split}.npy", embs)
    np.save(OUT / f"ssl_{args.split}_idx.npy", idx)
    print(f"Saved ssl_{args.split}.npy {embs.shape} + ssl_{args.split}_idx.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
