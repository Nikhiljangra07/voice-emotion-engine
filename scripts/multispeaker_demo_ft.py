"""Multi-speaker demo: who is speaking + what emotion FAMILY they're in, over time.

Builds a 2-speaker conversation by stitching labeled own-voice clips (so we have
ground truth for BOTH speaker and family at every turn), then:
  1. DISTINGUISHER — ECAPA-TDNN neural embeddings per window -> cluster into
     speakers (k known = 2, the reliable regime), scored by attribution accuracy.
  2. MEASURER — fine-tuned WavLM per turn -> V/A/D -> emotion family.
Prints a turn-by-turn timeline and scores both axes.

Runs in the isolated env:
    .venv_diar/bin/python -m scripts.multispeaker_demo_ft
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from scripts.diarize_neural import window_bounds
from scripts.eval_family_ft import TO_FAMILY
from scripts.predict_wavlm_ft import Namer, WavLMRegressor, normalize_vad

SR = 16_000
GAP_S = 0.4  # silence between turns

# Conversation script: (speaker, family). Alternating turns, mix of easy + hard.
PLAN = [
    ("female_A", "anger"), ("female_B", "joy"),
    ("female_A", "sadness"), ("female_B", "fear"),
    ("female_A", "joy"),   ("female_B", "sadness"),
    ("female_A", "neutral"), ("female_B", "surprise"),
    ("female_A", "fear"),  ("female_B", "anger"),
]


def load_clip(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak).astype("float32")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    by_key = defaultdict(list)
    for e in manifest:
        by_key[(e["speaker"], e["label"])].append(e["file"])

    # ── 1. build the conversation + ground-truth turns ──
    gap = np.zeros(int(GAP_S * SR), dtype="float32")
    audio_parts, turns, used = [], [], defaultdict(int)
    t = 0.0
    for spk, fam in PLAN:
        files = by_key.get((spk, fam), [])
        if not files:
            continue
        f = files[used[(spk, fam)] % len(files)]
        used[(spk, fam)] += 1
        y = load_clip(root / "own_voice" / f)
        start = t
        audio_parts.append(y)
        audio_parts.append(gap)
        dur = len(y) / SR
        turns.append({"start": start, "end": start + dur, "speaker": spk,
                      "family": fam, "file": f, "audio": y})
        t += dur + GAP_S
    convo = np.concatenate(audio_parts)
    (root / "own_voice/test_convos").mkdir(parents=True, exist_ok=True)
    sf.write(str(root / "own_voice/test_convos/convo_ft_demo.wav"), convo, SR)
    print(f"Built {len(turns)}-turn convo, {len(convo)/SR:.1f}s, 2 speakers.\n")

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    # ── 2. DISTINGUISHER: ECAPA embeddings per window -> cluster (k=2) ──
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=".venv_diar/ecapa_model", run_opts={"device": "cpu"})
    bounds = window_bounds(len(convo), SR, 2.0, 1.0)
    embs = np.zeros((len(bounds), 192), dtype=np.float32)
    for i, (s, e) in enumerate(bounds):
        with torch.no_grad():
            embs[i] = enc.encode_batch(torch.tensor(convo[s:e]).unsqueeze(0)
                                       ).squeeze().cpu().numpy()
    clabels = AgglomerativeClustering(n_clusters=2).fit_predict(normalize(embs))

    # true speaker per window = turn covering the window midpoint
    def turn_at(mid):
        for tn in turns:
            if tn["start"] <= mid < tn["end"]:
                return tn
        return None
    win_true, win_pred = [], []
    for (s, e), cl in zip(bounds, clabels):
        tn = turn_at((s + e) / 2 / SR)
        if tn is None:
            continue
        win_true.append(tn["speaker"]); win_pred.append(int(cl))
    # best cluster->speaker mapping (2! permutations)
    spk_ids = ["female_A", "female_B"]
    best_map, best_acc = None, -1
    for perm in permutations(spk_ids):
        m = {0: perm[0], 1: perm[1]}
        acc = np.mean([m[p] == t for p, t in zip(win_pred, win_true)])
        if acc > best_acc:
            best_acc, best_map = acc, m
    diar_acc = best_acc

    # ── 3. MEASURER: fine-tuned WavLM per turn -> family ──
    model = WavLMRegressor(str(root / "models/wavlm_vad_ft"))
    model.head.load_state_dict(torch.load(root / "models/wavlm_vad_ft/head.pt",
                                          map_location="cpu"))
    model.to(device).eval()
    namer = Namer(str(root / "models/namer_msp_final"))

    for tn in turns:
        y = tn["audio"][: int(8 * SR)]
        wav = torch.from_numpy(y).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = model(wav, mask).float().cpu().numpy().ravel()
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0
        pad = normalize_vad(*raw)
        tn["pred_family"] = TO_FAMILY[namer.predict(pad)["emotion"]]
        # predicted speaker for the turn = majority cluster over its windows
        cls = [best_map[int(cl)] for (s, e), cl in zip(bounds, clabels)
               if tn["start"] <= (s + e) / 2 / SR < tn["end"]]
        tn["pred_speaker"] = max(set(cls), key=cls.count) if cls else "?"

    # ── 4. report ──
    print("TIMELINE  (✓/✗ = speaker | family correct)")
    print(f"{'time':>11}  {'pred spk':10}{'true spk':10} spk  "
          f"{'pred fam':10}{'true fam':10} fam")
    print("-" * 74)
    fam_ok = spk_ok = 0
    for tn in turns:
        sp_ok = tn["pred_speaker"] == tn["speaker"]
        fa_ok = tn["pred_family"] == tn["family"]
        spk_ok += sp_ok; fam_ok += fa_ok
        print(f"{tn['start']:5.1f}-{tn['end']:4.1f}s  "
              f"{tn['pred_speaker']:10}{tn['speaker']:10}{'✓' if sp_ok else '✗':>3}  "
              f"{tn['pred_family']:10}{tn['family']:10}{'✓' if fa_ok else '✗':>3}")

    n = len(turns)
    print("\n" + "=" * 60)
    print(f"DISTINGUISHER (per-window speaker attribution, k=2): {diar_acc:.1%}")
    print(f"DISTINGUISHER (per-turn speaker):                    {spk_ok}/{n} = {spk_ok/n:.1%}")
    print(f"MEASURER (per-turn emotion family):                 {fam_ok}/{n} = {fam_ok/n:.1%}")
    # per-speaker family accuracy
    print("\nFamily accuracy by (true) speaker:")
    for spk in spk_ids:
        rs = [tn for tn in turns if tn["speaker"] == spk]
        acc = sum(tn["pred_family"] == tn["family"] for tn in rs) / len(rs)
        print(f"  {spk:10} {acc:.1%} (n={len(rs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
