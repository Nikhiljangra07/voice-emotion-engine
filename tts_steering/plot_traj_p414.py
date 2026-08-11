"""P4.14 — the graph: emotional fluctuation over the journey.

Reads out/p414/trajectory.json, renders a 3-panel figure:
  1. Valence over time (all window sizes faint, chosen size bold) + commanded
  2. Arousal over time (same)
  3. Judge family per window (chosen size) as a colored strip
Act boundaries as vertical lines with labels.

Run:  venv/bin/python tts_steering/plot_traj_p414.py
Out:  tts_steering/out/p414/trajectory_graph.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out/p414"
d = json.loads((OUT / "trajectory.json").read_text())
bounds, results, chosen = d["bounds"], d["results"], str(d["chosen"])

CENTROIDS = {"neutral": (-0.04, 0.45), "joy": (+0.30, 0.58),
             "sadness": (-0.28, 0.39), "anger": (-0.42, 0.70)}
ACT_COLOR = {"neutral": "#9e9e9e", "joy": "#f2a900", "sadness": "#4a6fa5",
             "anger": "#c0392b", "surprise": "#7d3c98", "fear": "#1a7a6d",
             "disgust": "#6d6d2a", "contempt": "#555555"}

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 1]})
fig.suptitle("P4.14 — The mouth writes a four-act emotional journey; "
             "the ear reads it back", fontsize=13)

for ax, dim, label in [(axes[0], "V", "valence"), (axes[1], "A", "arousal")]:
    for w, r in results.items():
        centers = [s + float(w) / 2 for s in r["starts"]]
        if w == chosen:
            ax.plot(centers, r[dim], "-o", ms=3, lw=2, color="#222222",
                    label=f"measured ({w}s window)")
        else:
            ax.plot(centers, r[dim], "-", lw=0.8, alpha=0.35,
                    label=f"{w}s")
    # commanded step trajectory
    xs, ys = [], []
    for b in bounds:
        c = CENTROIDS[b["act"]][0 if dim == "V" else 1]
        xs += [b["start"], b["end"]]
        ys += [c, c]
    ax.plot(xs, ys, lw=2.5, color="#e74c3c", alpha=0.8, ls="--",
            label="commanded (centroid)")
    for b in bounds:
        ax.axvline(b["start"], color="#bbbbbb", lw=0.8)
        ax.text((b["start"] + b["end"]) / 2,
                ax.get_ylim()[1] if dim == "V" else 0.98,
                b["act"].upper(), ha="center", va="top", fontsize=9,
                color=ACT_COLOR[b["act"]], fontweight="bold")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
axes[0].legend(loc="lower left", fontsize=8, ncol=3)
axes[0].set_ylim(-1, 1)
axes[1].set_ylim(0, 1)

r = results[chosen]
w = float(chosen)
for s, fam in zip(r["starts"], r.get("judge", [])):
    axes[2].barh(0, w / 2, left=s + w / 4, height=1,
                 color=ACT_COLOR.get(fam, "#dddddd"))
for b in bounds:
    axes[2].axvline(b["start"], color="white", lw=1)
axes[2].set_yticks([])
axes[2].set_ylabel("judge", rotation=0, ha="right", va="center")
axes[2].set_xlabel("time (s)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c)
           for f, c in ACT_COLOR.items() if f in set(r.get("judge", []))]
labels = [f for f in ACT_COLOR if f in set(r.get("judge", []))]
axes[2].legend(handles, labels, loc="upper right", fontsize=7,
               ncol=len(labels), frameon=False)

plt.tight_layout()
dest = OUT / "trajectory_graph.png"
plt.savefig(dest, dpi=140)
print(f"GRAPH_SAVED {dest}")
