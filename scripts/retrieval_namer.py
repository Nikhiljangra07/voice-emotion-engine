"""Retrieval namer — "Shazam for emotion families": frozen model + growable DB.

The architecture the design converged on:
  * CORPUS 1 = the fine-tuned WavLM (frozen). It only ever *embeds* — turns a clip
    into a 1024-dim vector. It is never retrained, so it can never forget.
  * CORPUS 2 = a vector DATABASE of (vector, family, speaker) exemplars. Knowledge
    is added by EMBED-AND-STORE (append), never by gradient training. Enrolling a
    speaker = dropping their labeled vectors in. kNN is multi-modal, so it can hold
    both "loud anger" and "cold anger" as separate exemplars.

A new clip is named by cosine k-NN against the database — matching the full vector
instead of squeezing to V/A/D (which on this data lifted acted fear 0%->73%).

Self-contained; runs in the isolated env:
    .venv_diar/bin/python -m scripts.retrieval_namer build      # seed DB from cache
    .venv_diar/bin/python -m scripts.retrieval_namer enroll --label joy --speaker me f1.wav f2.wav
    .venv_diar/bin/python -m scripts.retrieval_namer predict clip.wav [clip2.wav ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── the database-backed namer (pure numpy) ────────────────────────
class RetrievalNamer:
    def __init__(self, ambiguity_margin: float = 0.15):
        self.ambiguity_margin = ambiguity_margin
        self.vectors = np.zeros((0, 0), dtype=np.float32)  # L2-normalized rows
        self.labels: list[str] = []
        self.speakers: list[str] = []

    @property
    def size(self) -> int:
        return len(self.labels)

    def add(self, vectors, labels, speakers=None):
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        speakers = speakers or ["?"] * len(labels)
        if self.vectors.size == 0:
            self.vectors = v
        else:
            self.vectors = np.vstack([self.vectors, v])
        self.labels.extend(labels)
        self.speakers.extend(speakers)

    def predict(self, vector, k: int = 5, exclude_speaker: str | None = None) -> dict:
        if self.size == 0:
            raise RuntimeError("database is empty — build/enroll first.")
        q = np.asarray(vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self.vectors @ q
        valid = np.ones(self.size, bool)
        if exclude_speaker is not None:
            valid = np.array([s != exclude_speaker for s in self.speakers])
        idx = np.where(valid)[0]
        top = idx[np.argsort(-sims[idx])[:k]]
        # similarity-weighted votes per family
        votes: dict[str, float] = defaultdict(float)
        for i in top:
            votes[self.labels[i]] += float(sims[i])
        total = sum(votes.values()) or 1.0
        dist = {fam: v / total for fam, v in votes.items()}
        order = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return {
            "emotion": order[0][0],
            "confidence": order[0][1],
            "distribution": dict(order),
            "ambiguous": bool(margin < self.ambiguity_margin),
            "neighbors": [(self.labels[i], self.speakers[i], round(float(sims[i]), 3))
                          for i in top],
        }

    def save(self, path):
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps({
            "ambiguity_margin": self.ambiguity_margin,
            "labels": self.labels, "speakers": self.speakers}))

    @classmethod
    def load(cls, path):
        d = Path(path); m = json.loads((d / "meta.json").read_text())
        obj = cls(ambiguity_margin=m["ambiguity_margin"])
        obj.vectors = np.load(d / "vectors.npy")
        obj.labels = m["labels"]; obj.speakers = m["speakers"]
        return obj


# ── embedding (needs the frozen fine-tuned WavLM) ─────────────────
def _embedder(root):
    import torch
    from scripts.predict_wavlm_ft import WavLMRegressor, load_audio
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = WavLMRegressor(str(root / "models/wavlm_vad_ft")).to(device).eval()

    def embed(path):
        y = load_audio(str(path))
        wav = torch.from_numpy(y).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            h = model.backbone(wav, attention_mask=mask).last_hidden_state.mean(1)
        return h.float().cpu().numpy().ravel()
    return embed


FAMILIES = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]
DB_DIR = "models/retrieval_db"


def cmd_build(root, args):
    """Seed the DB from the cached 175-clip embeddings (no re-embedding)."""
    cache = root / "out/_ownvoice_ft_emb.npy"
    meta = root / "out/_ownvoice_ft_emb.meta.json"
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAMILIES]
    if cache.exists() and json.loads(meta.read_text()) == [e["file"] for e in labeled]:
        X = np.load(cache); print(f"seeding from cache ({len(X)} vectors)")
    else:
        embed = _embedder(root)
        X = np.vstack([embed(root / "own_voice" / e["file"]) for e in labeled])
    namer = RetrievalNamer()
    namer.add(X, [e["label"] for e in labeled], [e["speaker"] for e in labeled])
    namer.save(root / DB_DIR)
    print(f"built DB: {namer.size} exemplars, families="
          f"{sorted(set(namer.labels))}, speakers={sorted(set(namer.speakers))}")
    print(f"saved -> {DB_DIR}")


def cmd_enroll(root, args):
    namer = RetrievalNamer.load(root / DB_DIR)
    embed = _embedder(root)
    X = np.vstack([embed(f) for f in args.files])
    namer.add(X, [args.label] * len(args.files), [args.speaker] * len(args.files))
    namer.save(root / DB_DIR)
    print(f"enrolled {len(args.files)} '{args.label}' clip(s) for '{args.speaker}'. "
          f"DB now {namer.size} exemplars.")


def cmd_predict(root, args):
    namer = RetrievalNamer.load(root / DB_DIR)
    embed = _embedder(root)
    print(f"DB: {namer.size} exemplars\n")
    for f in args.files:
        r = namer.predict(embed(f), k=args.k)
        amb = " (ambiguous)" if r["ambiguous"] else ""
        print(f"{Path(f).name}: {r['emotion']}  conf={r['confidence']:.0%}{amb}")
        nb = ", ".join(f"{lab}[{spk}]={s}" for lab, spk, s in r["neighbors"])
        print(f"   nearest: {nb}")


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("enroll")
    e.add_argument("--label", required=True, choices=FAMILIES)
    e.add_argument("--speaker", required=True)
    e.add_argument("files", nargs="+")
    p = sub.add_parser("predict")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"build": cmd_build, "enroll": cmd_enroll, "predict": cmd_predict}[args.cmd](root, args)


if __name__ == "__main__":
    main()
