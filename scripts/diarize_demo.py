"""Speaker-aware emotion demo: per-speaker V/A/D trajectories from one audio file.

Diarizes a conversation (Path A — classical, our own vocal-tract features) and
runs the emotion engine per speaker, producing one trajectory per Speaker A/B/C…
plus a multi-speaker visualization.

HONEST SCOPE: diarization quality is moderate (good for acoustically distinct
voices, weaker for similar ones); per-speaker valence is still weak and labels
still ambiguous (arousal/dominance are the reliable axes). This is the first
real step, not a finished diarizer.

Usage:
    python -m scripts.diarize_demo --input own_voice/long_1.wav          # auto-k
    python -m scripts.diarize_demo --input clip.wav --speakers 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.dimensional.diarization import SpeakerDiarizer  # noqa: E402
from src.dimensional.namer import CentroidNamer  # noqa: E402
from src.dimensional.trajectory import TrajectoryEngine, trajectory_to_rows  # noqa: E402

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _summary(speaker: str, pts) -> None:
    ok = [p for p in pts if p.error is None]
    if not ok:
        print(f"  {speaker}: no analyzable windows"); return
    span = f"{pts[0].t_start:.0f}-{pts[-1].t_end:.0f}s"
    v = sum(p.valence for p in ok) / len(ok)
    a = sum(p.arousal for p in ok) / len(ok)
    d = sum(p.dominance for p in ok) / len(ok)
    amb = sum(p.ambiguous for p in ok)
    from collections import Counter
    top = Counter(p.emotion for p in ok).most_common(3)
    print(f"  {speaker}: {len(ok)} windows ({span})  "
          f"V={v:+.2f} A={a:.2f} D={d:+.2f}  ambiguous {amb}/{len(ok)}")
    print(f"      top emotions: " + ", ".join(f"{e} {n}" for e, n in top))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="own_voice/long_1.wav")
    ap.add_argument("--regressor", default="models/dim_svr_msp_final")
    ap.add_argument("--namer", default="models/namer_msp_final")
    ap.add_argument("--speakers", type=int, default=None, help="fixed count; default auto")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    args = ap.parse_args()

    eng = TrajectoryEngine.from_saved(args.regressor, args.namer,
                                      window_s=args.window, hop_s=args.hop)
    diar = SpeakerDiarizer(n_speakers=args.speakers, hop_s=args.hop)
    print(f"Analyzing {args.input} (diarize: "
          f"{'auto' if args.speakers is None else args.speakers} speakers)...")
    by_spk = eng.analyze_by_speaker(args.input, diar)
    print(f"\nDetected {diar.estimated_k_} speaker(s).\n")

    rows = []
    for spk, pts in by_spk.items():
        _summary(spk, pts)
        for r in trajectory_to_rows(pts):
            r["speaker"] = spk
            rows.append(r)

    stem = Path(args.input).stem
    df = pd.DataFrame(rows).sort_values("t_center")
    csv = OUT_DIR / f"diarized_{stem}.csv"
    df.to_csv(csv, index=False)

    # ── multi-speaker visualization ──
    namer = CentroidNamer.load(args.namer)
    ok = df.dropna(subset=["valence", "arousal", "dominance"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Per-Speaker Emotion — {stem}  "
                 f"({diar.estimated_k_} speakers, {ok['t_end'].max():.0f}s)",
                 fontsize=14, fontweight="bold")
    spk_color = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(by_spk)}

    def _centroids(ax, ix, iy):
        for emo in namer.labels:
            c = namer._centroids[emo]
            ax.scatter(c[ix], c[iy], marker="*", s=160, c="black", zorder=5)
            ax.annotate(emo, (c[ix], c[iy]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", zorder=6)

    # V-A and V-D planes, colored by speaker
    for ax, (ix, iy, xl, yl) in zip(
        axes[0], [(0, 1, "Valence", "Arousal"), (0, 2, "Valence", "Dominance")]
    ):
        for spk in by_spk:
            sub = ok[ok["speaker"] == spk]
            ax.scatter(sub["valence"], sub[yl.lower()], s=18,
                       c=spk_color[spk], label=spk, alpha=0.7)
        _centroids(ax, ix, iy)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl}–{yl}")
        ax.legend(fontsize=8)

    # arousal & valence over time, per speaker
    for ax, dim in zip(axes[1], ["arousal", "valence"]):
        for spk in by_spk:
            sub = ok[ok["speaker"] == spk]
            ax.scatter(sub["t_center"], sub[dim], s=14, c=spk_color[spk],
                       label=spk, alpha=0.7)
        ax.set_xlabel("time (s)"); ax.set_ylabel(dim); ax.set_title(f"{dim} over time")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = OUT_DIR / f"diarized_emotion_{stem}.png"
    fig.savefig(png, dpi=130)
    print(f"\nSaved → {csv}\nSaved → {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
