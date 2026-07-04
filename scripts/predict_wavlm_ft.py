"""End-to-end emotion prediction with the fine-tuned WavLM V/A/D model.

This is the inference counterpart to scripts/finetune_wavlm.py. It loads the
WavLM-large backbone fine-tuned on MSP-Podcast (held-out Test1 CCC: V 0.705,
A 0.714, D 0.626) plus its regression head, predicts valence/arousal/dominance
directly from audio, and names the emotion via the data-grounded PAD centroids.

Runs in the isolated deep-learning env (.venv_diar) which carries torch +
transformers; the namer math is inlined (a few lines of numpy) so the heavy
src package and its deps are not needed here.

Usage:
    .venv_diar/bin/python -m scripts.predict_wavlm_ft \
        --inputs own_voice/001.wav own_voice/008.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoModel

SR = 16_000
DIMS = ("valence", "arousal", "dominance")


# ── model (mirrors finetune_wavlm.WavLMRegressor) ─────────────────
class WavLMRegressor(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 3))

    def forward(self, wav, mask):
        out = self.backbone(wav, attention_mask=mask).last_hidden_state
        return self.head(out.mean(dim=1))


# ── audio (mirrors training preprocessing) ────────────────────────
def load_audio(path: str, max_s: float = 8.0) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) > int(max_s * SR):
        y = y[: int(max_s * SR)]
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak).astype("float32")


def normalize_vad(v: float, a: float, d: float) -> tuple[float, float, float]:
    """MSP SAM 1-7 -> PAD (matches src.dimensional.loader.normalize_vad_msp)."""
    return ((v - 4) / 3, (a - 1) / 6, (d - 4) / 3)


# ── namer (inlined CentroidNamer.predict) ─────────────────────────
class Namer:
    def __init__(self, path: str):
        meta = json.loads((Path(path) / "namer.json").read_text())
        self.labels = meta["labels"]
        self.margin = meta["ambiguity_margin"]
        self.cent = {e: np.array(v) for e, v in meta["centroids"].items()}
        self.inv = {e: np.array(v) for e, v in meta["inv_cov"].items()}

    def predict(self, point) -> dict:
        p = np.asarray(point, dtype=float).ravel()
        dist = {}
        for e in self.labels:
            delta = p - self.cent[e]
            dist[e] = float(np.sqrt(max(float(delta @ self.inv[e] @ delta), 0.0)))
        d = np.array([dist[e] for e in self.labels])
        logits = -d
        logits -= logits.max()
        w = np.exp(logits)
        probs = w / w.sum()
        distribution = {e: float(pr) for e, pr in zip(self.labels, probs)}
        order = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return {
            "emotion": order[0][0],
            "distribution": distribution,
            "ambiguous": bool(margin < self.margin),
            "intensity": float(np.linalg.norm(p)),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--model", default="models/wavlm_vad_ft")
    ap.add_argument("--namer", default="models/namer_msp_final")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    model = WavLMRegressor(args.model)
    head_sd = torch.load(Path(args.model) / "head.pt", map_location="cpu")
    model.head.load_state_dict(head_sd)
    model.to(device).eval()
    namer = Namer(args.namer)

    rows = []
    for f in args.inputs:
        y = load_audio(f)
        wav = torch.from_numpy(y).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = model(wav, mask).float().cpu().numpy().ravel()  # [0,1] each
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0                  # -> 1-7
        pad = normalize_vad(*raw)
        r = namer.predict(pad)
        rows.append({
            "clip": Path(f).name,
            "valence_raw": float(raw[0]), "arousal_raw": float(raw[1]),
            "dominance_raw": float(raw[2]),
            "valence": float(pad[0]), "arousal": float(pad[1]),
            "dominance": float(pad[2]),
            "emotion": r["emotion"], "ambiguous": r["ambiguous"],
            "intensity": r["intensity"], "distribution": r["distribution"],
        })

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\n{'clip':<26}{'V':>6}{'A':>6}{'D':>6}   emotion")
        print("-" * 56)
        for r in rows:
            amb = " ~" if r["ambiguous"] else ""
            print(f"{r['clip']:<26}{r['valence']:>6.2f}{r['arousal']:>6.2f}"
                  f"{r['dominance']:>6.2f}   {r['emotion']}{amb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
