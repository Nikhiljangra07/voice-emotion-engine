"""Judge the Chatterbox probe clips through the frozen bridge; append ledger rows.

Run:  .venv_tts/bin/python tts_steering/judge_cbx.py
"""

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402

LEDGER = HERE / "out/loop_ledger.csv"
CENTROIDS = {  # parent WRITEUP §5.4, fit on 137k MSP clips
    "neutral": (-0.04, 0.45, +0.03),
    "sadness": (-0.28, 0.39, -0.07),
    "joy":     (+0.30, 0.58, +0.19),
    "anger":   (-0.42, 0.70, +0.41),
}

manifest = json.loads((HERE / "out/cbx/manifest.json").read_text())
results = judge([m["file"] for m in manifest])

n_prev = sum(1 for _ in open(LEDGER)) - 1
rows = []
print(f"{'clip':18s} {'target':8s} {'V':>6s} {'A':>6s} {'D':>6s} {'d':>6s}  judge")
for m, r in zip(manifest, results):
    c = CENTROIDS[m["target"]]
    d = math.sqrt((r["V"] - c[0]) ** 2 + (r["A"] - c[1]) ** 2 + (r["D"] - c[2]) ** 2)
    hit = r["judge_family"] == m["target"]
    n_prev += 1
    rows.append([n_prev, "chatterbox", m["target"], m["control"],
                 round(r["V"], 3), round(r["A"], 3), round(r["D"], 3),
                 r["judge_family"], round(r["judge_confidence"], 2),
                 round(d, 3), int(hit)])
    mark = "HIT" if hit else ""
    print(f"{m['name']:18s} {m['target']:8s} {r['V']:+.2f} {r['A']:+.2f} "
          f"{r['D']:+.2f} {d:6.3f}  {r['judge_family']}@{r['judge_confidence']:.0%} {mark}")

with open(LEDGER, "a", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\nledger rows added: {len(rows)} (total {n_prev}) -> {LEDGER}")
print("JUDGE_CBX_DONE")
