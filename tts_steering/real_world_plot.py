"""Graph the real-world trajectory: 'Sorry, Wrong Number' (1943) read by the ear.

Three panels over the full 30 minutes:
  1. Valence (raw dots + rolling median) — the drift into darkness
  2. Arousal (raw dots + rolling median) — the mounting activation
  3. Family verdicts as a colored timeline + confidence

No ground-truth file exists — the evaluation is against the documented
dramatic arc (irritation -> overheard murder plot -> mounting panic ->
terror climax at the end). The graph must tell that story on its own.

Run:  venv/bin/python tts_steering/real_world_plot.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out/real_world"
rows = json.loads((OUT / "traj.json").read_text())

t = np.array([r["t"] for r in rows]) / 60.0
V = np.array([r["V"] for r in rows])
A = np.array([r["A"] for r in rows])
fam = [r["family"] for r in rows]
conf = np.array([r["conf"] for r in rows])
rms = np.array([r["rms"] for r in rows], dtype=float)

quiet = rms < np.percentile(rms, 15)


def med(x, k=15):
    out = np.copy(x)
    h = k // 2
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - h):i + h + 1])
    return out


COLORS = {"anger": "#d62728", "fear": "#9467bd", "joy": "#ffcf40",
          "sadness": "#1f77b4", "surprise": "#ff7f0e", "neutral": "#a0a0a0"}

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 2]})
fig.suptitle("The ear on found audio — 'Sorry, Wrong Number' "
             "(Suspense, 1943, public domain) · 3s windows, 50% overlap",
             fontsize=13)

ax = axes[0]
ax.scatter(t, V, s=4, alpha=0.25, c="#1f77b4")
ax.plot(t, med(V), lw=2.2, color="#0b3d91", label="valence (rolling median)")
ax.axhline(0, color="k", lw=0.5, alpha=0.4)
ax.set_ylabel("Valence")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-1, 1)

ax = axes[1]
ax.scatter(t, A, s=4, alpha=0.25, c="#d62728")
ax.plot(t, med(A), lw=2.2, color="#8b0000", label="arousal (rolling median)")
ax.set_ylabel("Arousal")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 1)

ax = axes[2]
for x, f, c, q in zip(t, fam, conf, quiet):
    ax.bar(x, 1.0, width=0.028, bottom=0,
           color=COLORS.get(f, "#cccccc"), alpha=(0.25 if q else min(1.0, 0.35 + 0.65 * c)))
ax.set_ylabel("Family")
ax.set_yticks([])
ax.set_xlabel("Minutes")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
ax.legend(handles, COLORS.keys(), loc="upper right", fontsize=7, ncol=6)

for a in axes:
    for x0, x1 in [(0, 0)]:
        pass
plt.tight_layout()
plt.savefig(OUT / "real_world_trajectory.png", dpi=140)
print(f"saved {OUT/'real_world_trajectory.png'}")

thirds = np.array_split(np.arange(len(rows)), 6)
print("\nsegment summary (5-min slices):")
for i, idx in enumerate(thirds):
    fams = [fam[j] for j in idx]
    top = max(set(fams), key=fams.count)
    print(f"  {t[idx[0]]:4.1f}-{t[idx[-1]]:4.1f} min: "
          f"V={np.median(V[idx]):+.2f} A={np.median(A[idx]):.2f} "
          f"top-family={top} ({fams.count(top)}/{len(idx)})")
