"""P4.2 sweep — PHASE B: judge every sweep clip, append ledger rows (runs in .venv_tts).

Run:  .venv_tts/bin/python tts_steering/sweep_p42_judge.py
"""

import csv
import json
import math
from pathlib import Path

from bridge import judge  # tts_steering/bridge.py — subprocess-only, no engine imports

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tts_steering/out/sweep_p42"
LEDGER = ROOT / "tts_steering/out/loop_ledger.csv"

# Data-grounded PAD centroids (parent WRITEUP §5.4 — fit on 137k MSP clips).
CENTROIDS = {
    "sadness": (-0.28, 0.39, -0.07),
    "joy":     (+0.30, 0.58, +0.19),
    "anger":   (-0.42, 0.70, +0.41),
    "neutral": (-0.04, 0.45, +0.03),
}

manifest = json.loads((OUT / "manifest.json").read_text())
clips = [m["file"] for m in manifest]
results = judge(clips)   # one batched call: WavLM V/A/D + e2v family per clip

rows = []
start_iter = sum(1 for _ in open(LEDGER)) if LEDGER.exists() else 1  # header = row 0
for m, r in zip(manifest, results):
    tgt = CENTROIDS[m["target"]]
    got = (r["V"], r["A"], r["D"])
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(tgt, got)))
    fam_hit = (r["judge_family"] == m["target"]) or \
              (m["target"] == "neutral" and r["judge_family"] == "neutral")
    rows.append({
        "iteration": start_iter + len(rows),
        "system": "indextts2",
        "target_emotion": m["target"],
        "control_params": f"emo_vector={m['vector']};spk={m['prompt']}",
        "judged_V": round(r["V"], 3), "judged_A": round(r["A"], 3),
        "judged_D": round(r["D"], 3),
        "judge_family": r["judge_family"],
        "judge_confidence": round(r["judge_confidence"], 2),
        "distance_to_target": round(dist, 3),
        "accepted": int(fam_hit),
        # not in ledger schema, printed only:
        "_name": m["name"], "_wavlm": r["wavlm_emotion"],
        "_amb": r["judge_ambiguous"],
    })

with open(LEDGER, "a", newline="") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow([r[k] for k in ("iteration", "system", "target_emotion",
                                   "control_params", "judged_V", "judged_A",
                                   "judged_D", "judge_family", "judge_confidence",
                                   "distance_to_target", "accepted")])

print(f"\n{'clip':<24}{'target':<9}{'V':>6}{'A':>6}{'D':>6}  "
      f"{'dist':>5}  {'wavlm':<10}{'judge(e2v)':<14} hit")
print("-" * 88)
for r in rows:
    amb = "~" if r["_amb"] else ""
    print(f"{r['_name']:<24}{r['target_emotion']:<9}"
          f"{r['judged_V']:>6.2f}{r['judged_A']:>6.2f}{r['judged_D']:>6.2f}  "
          f"{r['distance_to_target']:>5.3f}  {r['_wavlm']:<10}"
          f"{r['judge_family']+'@'+format(r['judge_confidence'],'.0%')+amb:<14} "
          f"{'HIT' if r['accepted'] else 'miss'}")
print(f"\nledger now has {start_iter + len(rows) - 1} data rows -> {LEDGER}")
