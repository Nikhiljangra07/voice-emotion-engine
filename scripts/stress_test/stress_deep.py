"""Stress harness — DEEP path: fine-tuned WavLM V/A/D + hybrid family namer.

Models load ONCE; every stress clip runs through both, exceptions caught per file.
Checks: NaN/Inf in outputs, out-of-range V/A/D, crash vs explicit error.
"""
import json, math, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
STRESS = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
from scripts.predict_wavlm_ft import WavLMRegressor, load_audio, normalize_vad
from scripts.adaptors import HybridAdaptor

root = Path.cwd()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
reg = WavLMRegressor("models/wavlm_vad_ft")
head_sd = torch.load(Path("models/wavlm_vad_ft") / "head.pt", map_location="cpu")
reg.head.load_state_dict(head_sd)          # <-- the step my first harness missed
reg = reg.to(dev).eval()
hyb = HybridAdaptor.load(root)
print(f"models loaded (device={dev})", flush=True)

def bad(x):
    return math.isnan(x) or math.isinf(x)

rows = []
for f in sorted(STRESS.glob("*.wav")):
    rec = {"clip": f.name}
    # ── V/A/D regression ──
    try:
        y = load_audio(str(f))
        wav = torch.from_numpy(y).unsqueeze(0).to(dev)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = reg(wav, mask).float().cpu().numpy().ravel()   # [0,1] each
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0                  # -> SAM 1-7
        v, a, d = normalize_vad(*raw)                              # -> PAD
        v, a, d = (v + 1) / 2, a, (d + 1) / 2                      # -> [0,1] for range check
        issues = []
        if any(bad(float(x)) for x in (v, a, d)):
            issues.append("NaN/Inf")
        if not all(-0.01 <= float(x) <= 1.01 for x in (v, a, d)):
            issues.append(f"out-of-range v={v:.2f} a={a:.2f} d={d:.2f}")
        rec["vad_verdict"] = "PREDICTED" + ("!" + ";".join(issues) if issues else "")
        rec["V"], rec["A"], rec["D"] = round(float(v),3), round(float(a),3), round(float(d),3)
    except Exception as e:
        msg = str(e)[:120]
        rec["vad_verdict"] = "EXPLICIT_ERR" if msg else "CRASH"
        rec["vad_error"] = f"{type(e).__name__}: {msg}"
    # ── hybrid family namer ──
    try:
        r = hyb.predict(str(f))
        conf = float(r["confidence"])
        rec["hyb_verdict"] = "PREDICTED" + ("!NaN" if bad(conf) else "")
        rec["hyb_emotion"] = r["emotion"]
        rec["hyb_conf"] = round(conf, 3)
        rec["hyb_ambiguous"] = bool(r["ambiguous"])
    except Exception as e:
        msg = str(e)[:120]
        rec["hyb_verdict"] = "EXPLICIT_ERR" if msg else "CRASH"
        rec["hyb_error"] = f"{type(e).__name__}: {msg}"
    rows.append(rec)
    vs = rec.get("vad_verdict"); hs = rec.get("hyb_verdict")
    det = (f"V={rec.get('V')} A={rec.get('A')} D={rec.get('D')}"
           if "V" in rec else rec.get("vad_error", "")[:60])
    hd = (f"{rec.get('hyb_emotion')}@{rec.get('hyb_conf')}"
          f"{'(amb)' if rec.get('hyb_ambiguous') else ''}"
          if "hyb_emotion" in rec else rec.get("hyb_error", "")[:60])
    print(f"{f.name:32s} vad:{vs:14s} {det:38s} hyb:{hs:14s} {hd}", flush=True)

OUT.write_text(json.dumps(rows, indent=1))
print(f"\nsaved -> {OUT}")
