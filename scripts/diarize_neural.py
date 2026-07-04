"""Neural speaker diarizer (Path B) — runs in the ISOLATED .venv_diar.

Standalone by design: imports ONLY torch / speechbrain / sklearn / soundfile /
librosa — never our src engine (which lives in the main venv). It reads one audio
file, computes an ECAPA-TDNN speaker embedding per window (same 2s/1s grid as the
emotion engine), clusters the embeddings into speakers, and writes a
"who-spoke-when" labels CSV that the main pipeline consumes.

These embeddings are trained specifically to encode speaker identity invariant to
emotion/words — the thing hand-crafted acoustic features can't do. This is the
"distinguisher"; the emotion "measurer" stays in the main venv, untouched.

Run (with the isolated interpreter):
    .venv_diar/bin/python -m scripts.diarize_neural --input clip.wav --out out/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

TARGET_SR = 16_000
MIN_DURATION_S = 0.5


def window_bounds(n: int, sr: int, win_s: float, hop_s: float):
    """Identical rule to src.dimensional.trajectory.window_bounds (alignment)."""
    win = max(1, int(round(win_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    min_len = int(MIN_DURATION_S * sr)
    bounds, start = [], 0
    while start < n:
        end = min(start + win, n)
        if end - start >= min_len:
            bounds.append((start, end))
        if end >= n:
            break
        start += hop
    return bounds


def load_audio(path: str):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32), TARGET_SR


def estimate_k(Xs: np.ndarray, max_k: int) -> int:
    upper = min(max_k, Xs.shape[0] - 1)
    if upper < 2:
        return 1
    best_k, best = 2, -1.0
    for k in range(2, upper + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
        try:
            s = silhouette_score(Xs, labels)
        except ValueError:
            continue
        if s > best:
            best_k, best = k, s
    return best_k if best >= 0.10 else 1


def smooth(labels: np.ndarray, min_win: int) -> np.ndarray:
    out = labels.copy()
    i, n = 0, len(out)
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_win and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=6)
    ap.add_argument("--min-turn", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine-distance threshold for auto speaker count "
                         "(more reliable than silhouette); overrides silhouette")
    args = ap.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier

    y, sr = load_audio(args.input)
    bounds = window_bounds(len(y), sr, args.window, args.hop)
    print(f"{len(bounds)} windows; loading ECAPA-TDNN encoder...")
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=".venv_diar/ecapa_model",
        run_opts={"device": "cpu"},
    )

    embs = np.zeros((len(bounds), 192), dtype=np.float32)
    for i, (s, e) in enumerate(bounds):
        seg = torch.tensor(y[s:e]).unsqueeze(0)
        with torch.no_grad():
            embs[i] = enc.encode_batch(seg).squeeze().cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"  embedded {i+1}/{len(bounds)}")

    # Speaker embeddings are compared by COSINE distance → L2-normalize so that
    # Euclidean/ward clustering on the unit sphere is equivalent to cosine.
    Xs = normalize(embs)
    np.save(str(out_csv) + ".emb.npy", embs)  # cache for threshold calibration

    if args.speakers is not None:
        k = args.speakers
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
    elif args.threshold is not None:
        # Cosine-distance threshold: cut the dendrogram where between-cluster
        # cosine distance exceeds the threshold — the standard diarization way to
        # decide the speaker count (far more reliable than silhouette).
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=args.threshold,
            metric="cosine", linkage="average").fit_predict(embs)
        k = len(set(labels))
    else:
        k = estimate_k(Xs, args.max_speakers)
        labels = (np.zeros(len(bounds), dtype=int) if k <= 1
                  else AgglomerativeClustering(n_clusters=k).fit_predict(Xs))
    print(f"Detected {k} speaker(s).")
    labels = smooth(labels, max(1, int(round(args.min_turn / args.hop))))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "t_start", "t_end", "speaker"])
        for i, (s, e) in enumerate(bounds):
            w.writerow([i, round(s / sr, 3), round(e / sr, 3), int(labels[i])])
    print(f"Wrote {len(bounds)} labels (k={k}) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
