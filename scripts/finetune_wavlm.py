"""Fine-tune WavLM end-to-end on MSP-Podcast for V/A/D regression (GPU).

The real SOTA step: instead of freezing WavLM and probing it, we update its
weights to predict valence/arousal/dominance, optimizing the CCC objective (the
AVEC standard). Designed to run on a single 48 GB GPU (A40 / RTX 6000 Ada).

Self-contained: needs torch, transformers, soundfile, librosa, pandas, numpy.
Reads MSP's labels_consensus.csv (same schema as our loader) + the Audios/ dir.

Example:
    python finetune_wavlm.py \
        --labels data/Labels/labels_consensus.csv --audio-dir data/Audios \
        --model microsoft/wavlm-large --epochs 4 --batch 16 --out wavlm_vad_ft
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# Large eval sets + many workers exhaust file descriptors with the default
# file_descriptor sharing strategy; file_system avoids the "Too many open files" crash.
try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

SR = 16_000
DIMS = ("valence", "arousal", "dominance")


# ── data ──────────────────────────────────────────────────────────
def _norm(v):  # MSP SAM 1-7 → [0,1] (CCC is scale-invariant; helps optimization)
    return (np.asarray(v, dtype=np.float32) - 1.0) / 6.0


class MSPDataset(Dataset):
    def __init__(self, df, audio_dir, max_s=8.0):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.max_len = int(max_s * SR)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            y, sr = sf.read(str(self.audio_dir / r["FileName"]), dtype="float32",
                            always_2d=False)
        except Exception:
            return None  # unreadable/truncated file — skip, don't crash the run
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        if len(y) < int(0.1 * SR):  # empty/too-short (e.g. quota-truncated)
            return None
        if len(y) > self.max_len:
            y = y[: self.max_len]
        peak = np.max(np.abs(y)) or 1.0
        y = y / peak
        tgt = _norm([r["EmoVal"], r["EmoAct"], r["EmoDom"]])
        return torch.from_numpy(y), torch.from_numpy(tgt)


def collate(batch):
    batch = [b for b in batch if b is not None]  # drop skipped (bad) files
    if not batch:
        return None
    ys, ts = zip(*batch)
    n = max(y.shape[0] for y in ys)
    wav = torch.zeros(len(ys), n)
    mask = torch.zeros(len(ys), n, dtype=torch.long)
    for i, y in enumerate(ys):
        wav[i, : y.shape[0]] = y
        mask[i, : y.shape[0]] = 1
    return wav, mask, torch.stack(ts)


# ── model ─────────────────────────────────────────────────────────
class WavLMRegressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.backbone.feature_extractor._freeze_parameters()  # standard: freeze CNN
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 3))

    def forward(self, wav, mask):
        out = self.backbone(wav, attention_mask=mask).last_hidden_state
        # mask is at sample resolution; derive frame-level lengths via mean-pool
        pooled = out.mean(dim=1)  # WavLM downsamples ~320x; mean-pool over frames
        return self.head(pooled)


# ── CCC loss / metric ─────────────────────────────────────────────
def ccc(yt, yp):
    mt, mp = yt.mean(0), yp.mean(0)
    vt, vp = yt.var(0, unbiased=False), yp.var(0, unbiased=False)
    cov = ((yt - mt) * (yp - mp)).mean(0)
    return 2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-8)  # per-dim


def ccc_loss(yt, yp):
    return (1.0 - ccc(yt, yp)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    T, P = [], []
    for batch in loader:
        if batch is None:
            continue
        wav, mask, tgt = batch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(wav.to(device), mask.to(device))
        T.append(tgt); P.append(pred.float().cpu())
    T, P = torch.cat(T), torch.cat(P)
    c = ccc(T, P)
    return {d: float(c[i]) for i, d in enumerate(DIMS)}


# ── train ─────────────────────────────────────────────────────────
def load_split(labels, split):
    df = pd.read_csv(labels)
    canon = {"train": "Train", "dev": "Development", "development": "Development",
             "test1": "Test1", "test2": "Test2"}
    df = df[df["Split_Set"] == canon.get(split.lower(), split)]
    for c in ("EmoVal", "EmoAct", "EmoDom"):
        df = df[pd.to_numeric(df[c], errors="coerce").notna()]
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--model", default="microsoft/wavlm-large")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-s", type=float, default=8.0)
    ap.add_argument("--subset", type=int, default=0, help="cap train rows (debug)")
    ap.add_argument("--out", default="wavlm_vad_ft")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}")

    tr = load_split(args.labels, "train")
    dev = load_split(args.labels, "dev")
    if args.subset:
        tr = tr.sample(n=min(args.subset, len(tr)), random_state=0)
    print(f"train {len(tr)}  dev {len(dev)}")

    dl_tr = DataLoader(MSPDataset(tr, args.audio_dir, args.max_s), batch_size=args.batch,
                       shuffle=True, num_workers=8, collate_fn=collate, drop_last=True)
    dl_dev = DataLoader(MSPDataset(dev, args.audio_dir, args.max_s), batch_size=args.batch,
                        shuffle=False, num_workers=8, collate_fn=collate)

    model = WavLMRegressor(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for it, batch in enumerate(dl_tr):
            if batch is None:
                continue
            wav, mask, tgt = batch
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(wav.to(device), mask.to(device))
                loss = ccc_loss(tgt.to(device), pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if it % 100 == 0:
                print(f"  ep{ep} it{it}/{len(dl_tr)} loss {loss.item():.4f}")
        c = evaluate(model, dl_dev, device)
        mean = sum(c.values()) / 3
        print(f"[ep{ep}] dev CCC  V {c['valence']:.3f}  A {c['arousal']:.3f}  "
              f"D {c['dominance']:.3f}  mean {mean:.3f}")
        if mean > best:
            best = mean
            Path(args.out).mkdir(exist_ok=True, parents=True)
            model.backbone.save_pretrained(args.out)
            torch.save(model.head.state_dict(), Path(args.out) / "head.pt")
            print(f"   saved best → {args.out} (mean {mean:.3f})")

    # Final held-out Test1.
    test = load_split(args.labels, "test1")
    dl_te = DataLoader(MSPDataset(test, args.audio_dir, args.max_s),
                       batch_size=args.batch, shuffle=False, num_workers=8,
                       collate_fn=collate)
    c = evaluate(model, dl_te, device)
    print(f"\n[FINAL Test1] CCC  V {c['valence']:.3f}  A {c['arousal']:.3f}  "
          f"D {c['dominance']:.3f}  mean {sum(c.values())/3:.3f}")


if __name__ == "__main__":
    main()
