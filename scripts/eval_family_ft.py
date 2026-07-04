"""Family-level emotion accuracy of the fine-tuned WavLM model on labeled clips.

Predicts V/A/D with the fine-tuned WavLM, names the 8-class MSP emotion via the
PAD centroids, then COLLAPSES to coarse families (Happy/Sad/Angry/Fear/Surprise/
Neutral) and scores against the ground-truth families in own_voice/manifest.json.

Family, not sub-emotion, on purpose: joy is *within* the happy family; naming the
family is the honest, achievable target for wordless audio. Reports overall +
per-family + per-speaker accuracy, a confusion matrix, and mean V/A/D per true
family (to see where the model's dimensions land).

Run:  .venv_diar/bin/python -m scripts.eval_family_ft
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.predict_wavlm_ft import Namer, WavLMRegressor, load_audio, normalize_vad

# 8-class MSP emotion  ->  coarse family (matches manifest label vocabulary)
TO_FAMILY = {
    "joy": "joy", "sadness": "sadness", "anger": "anger",
    "fear": "fear", "surprise": "surprise", "neutral": "neutral",
    "contempt": "anger", "disgust": "anger",   # no contempt/disgust in this test set
}
FAMILIES = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "own_voice/manifest.json").read_text())
    labeled = [e for e in manifest if e["label"] in FAMILIES]

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = WavLMRegressor(str(root / "models/wavlm_vad_ft"))
    model.head.load_state_dict(torch.load(root / "models/wavlm_vad_ft/head.pt",
                                          map_location="cpu"))
    model.to(device).eval()
    namer = Namer(str(root / "models/namer_msp_final"))

    n = len(labeled)
    print(f"Scoring {n} labeled clips on {device} ...")
    rows = []
    for i, e in enumerate(labeled):
        y = load_audio(str(root / "own_voice" / e["file"]))
        wav = torch.from_numpy(y).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = model(wav, mask).float().cpu().numpy().ravel()
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0
        pad = normalize_vad(*raw)
        pred8 = namer.predict(pad)["emotion"]
        fam = TO_FAMILY[pred8]
        rows.append({"true": e["label"], "pred": fam, "spk": e["speaker"],
                     "V": float(pad[0]), "A": float(pad[1]), "D": float(pad[2])})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{n}")

    # ── overall + baselines ──
    correct = sum(r["true"] == r["pred"] for r in rows)
    maj = Counter(r["true"] for r in rows).most_common(1)[0]
    print("\n" + "=" * 60)
    print(f"OVERALL family accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"  chance (6-way): 16.7%   majority-class ('{maj[0]}'): {maj[1]/n:.1%}")

    # ── per-family recall ──
    print("\nPer-family recall:")
    by_true = defaultdict(list)
    for r in rows:
        by_true[r["true"]].append(r)
    for fam in FAMILIES:
        rs = by_true[fam]
        if not rs:
            continue
        acc = sum(r["pred"] == fam for r in rs) / len(rs)
        conf = Counter(r["pred"] for r in rs).most_common(2)
        confstr = ", ".join(f"{k}:{v}" for k, v in conf)
        print(f"  {fam:9} {acc:5.1%}  (n={len(rs):2d})   -> {confstr}")

    # ── per-speaker ──
    print("\nPer-speaker accuracy:")
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["spk"]].append(r)
    for spk, rs in sorted(by_spk.items()):
        acc = sum(r["true"] == r["pred"] for r in rs) / len(rs)
        print(f"  {spk:14} {acc:5.1%}  (n={len(rs)})")

    # ── confusion matrix ──
    print("\nConfusion (rows=true, cols=pred):")
    hdr = "true\\pred   " + "".join(f"{f[:4]:>6}" for f in FAMILIES)
    print(hdr)
    for t in FAMILIES:
        rs = by_true[t]
        cnt = Counter(r["pred"] for r in rs)
        line = f"{t:9}  " + "".join(f"{cnt.get(p,0):>6}" for p in FAMILIES)
        print(line)

    # ── mean V/A/D per true family (diagnose where dims land) ──
    print("\nMean predicted V/A/D per TRUE family:")
    print(f"{'family':9}   {'V':>6}{'A':>6}{'D':>6}")
    for t in FAMILIES:
        rs = by_true[t]
        v = np.mean([r["V"] for r in rs]); a = np.mean([r["A"] for r in rs])
        d = np.mean([r["D"] for r in rs])
        print(f"{t:9}   {v:6.2f}{a:6.2f}{d:6.2f}")

    (root / "out").mkdir(exist_ok=True)
    (root / "out/family_eval_ft.json").write_text(json.dumps(rows, indent=1))
    print("\nsaved -> out/family_eval_ft.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
