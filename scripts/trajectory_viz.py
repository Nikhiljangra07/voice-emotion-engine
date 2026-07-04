"""P2.5 — the "emotion web": visualize a trajectory through PAD space.

Reads a trajectory CSV (scripts.trajectory_demo) + the namer's data centroids and
renders a 4-panel figure: the 3-D PAD path, two 2-D projections (V-A and V-D,
the latter showing the dominance separation), and V/A/D over time.

Usage:
    python -m scripts.trajectory_viz --trajectory out/trajectory_long_1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dimensional.namer import CentroidNamer  # noqa: E402

OUT_DIR = Path("out")


def _draw_centroids_2d(ax, namer, ix, iy):
    for emo in namer.labels:
        c = namer._centroids[emo]
        ax.scatter(c[ix], c[iy], marker="*", s=220, c="black", zorder=5)
        ax.annotate(emo, (c[ix], c[iy]), fontsize=8, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points", zorder=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="out/trajectory_long_1.csv")
    ap.add_argument("--namer", default="models/namer_msp_final")
    args = ap.parse_args()

    df = pd.read_csv(args.trajectory).dropna(subset=["valence", "arousal", "dominance"])
    namer = CentroidNamer.load(args.namer)
    t = df["t_center"].to_numpy()
    V, A, D = df["valence"].to_numpy(), df["arousal"].to_numpy(), df["dominance"].to_numpy()
    stem = Path(args.trajectory).stem

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Emotion Trajectory — {stem}  ({len(df)} windows, "
                 f"{t.max():.0f}s)", fontsize=15, fontweight="bold")

    # ── (1) 3-D PAD path, colored by time ──
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(V, A, D, c="0.6", lw=0.7, alpha=0.6, zorder=1)
    ax.scatter(V, A, D, c=t, cmap="viridis", s=14, zorder=2)
    for emo in namer.labels:
        c = namer._centroids[emo]
        ax.scatter(c[0], c[1], c[2], marker="*", s=180, c="red", zorder=5)
        ax.text(c[0], c[1], c[2], emo, fontsize=7, fontweight="bold")
    ax.set_xlabel("Valence"); ax.set_ylabel("Arousal"); ax.set_zlabel("Dominance")
    ax.set_title("3-D PAD path (color = time)")

    # ── (2) V-A projection ──
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(V, A, c="0.7", lw=0.6, alpha=0.6)
    sc = ax.scatter(V, A, c=t, cmap="viridis", s=16)
    _draw_centroids_2d(ax, namer, 0, 1)
    ax.axhline(0.5, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel("Valence"); ax.set_ylabel("Arousal")
    ax.set_title("Valence–Arousal projection")
    plt.colorbar(sc, ax=ax, label="time (s)")

    # ── (3) V-D projection (dominance separation) ──
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(V, D, c="0.7", lw=0.6, alpha=0.6)
    ax.scatter(V, D, c=t, cmap="viridis", s=16)
    _draw_centroids_2d(ax, namer, 0, 2)
    ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel("Valence"); ax.set_ylabel("Dominance")
    ax.set_title("Valence–Dominance projection")

    # ── (4) V/A/D over time ──
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, V, label="valence", lw=1.2)
    ax.plot(t, A, label="arousal", lw=1.2)
    ax.plot(t, D, label="dominance", lw=1.2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("PAD value")
    ax.set_title("V / A / D over time"); ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / f"emotion_web_{stem}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved emotion web → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
