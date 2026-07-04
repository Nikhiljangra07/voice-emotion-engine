"""One-command speaker-aware emotion analysis using the NEURAL distinguisher.

This is the unified "one system, two angles" entry point:
  angle 1 (distinguisher): neural ECAPA diarizer in the isolated .venv_diar →
           a who-spoke-when labels file;
  angle 2 (measurer): our emotion engine (main venv) reads each labelled window →
           per-speaker V/A/D trajectory + emotion.

The heavy neural dependency stays quarantined in .venv_diar (invoked as a
subprocess); the emotion engine is never touched.

Usage:
    python -m scripts.diarize_neural_demo --input own_voice/kitchen_debate.wav
    python -m scripts.diarize_neural_demo --input clip.wav --speakers 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.dimensional.namer import CentroidNamer  # noqa: E402
from src.dimensional.trajectory import TrajectoryEngine, trajectory_to_rows  # noqa: E402

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
DIAR_PY = ".venv_diar/bin/python"
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def run_neural_diarizer(inp: str, out_csv: Path, window: float, hop: float,
                        speakers: int | None) -> None:
    if not Path(DIAR_PY).exists():
        sys.exit(f"Isolated env not found at {DIAR_PY}. Create it first "
                 "(see TRAJECTORY_ENGINE / diarize_neural).")
    cmd = [DIAR_PY, "-m", "scripts.diarize_neural", "--input", inp,
           "--out", str(out_csv), "--window", str(window), "--hop", str(hop)]
    if speakers is not None:
        cmd += ["--speakers", str(speakers)]
    print("→ running neural distinguisher (isolated env)...")
    subprocess.run(cmd, check=True)


def summarize(by_spk: dict) -> None:
    for spk, pts in by_spk.items():
        ok = [p for p in pts if p.error is None]
        if not ok:
            print(f"  {spk}: no analyzable windows"); continue
        v = sum(p.valence for p in ok) / len(ok)
        a = sum(p.arousal for p in ok) / len(ok)
        d = sum(p.dominance for p in ok) / len(ok)
        amb = sum(p.ambiguous for p in ok)
        top = Counter(p.emotion for p in ok).most_common(3)
        print(f"  {spk}: {len(ok)} windows  V={v:+.2f} A={a:.2f} D={d:+.2f}  "
              f"ambiguous {amb}/{len(ok)}")
        print("      top: " + ", ".join(f"{e} {n}" for e, n in top))


def render(by_spk: dict, namer: CentroidNamer, stem: str, k: int) -> None:
    rows = []
    for spk, pts in by_spk.items():
        for r in trajectory_to_rows(pts):
            r["speaker"] = spk
            rows.append(r)
    df = pd.DataFrame(rows).sort_values("t_center")
    df.to_csv(OUT_DIR / f"neural_diarized_{stem}.csv", index=False)
    ok = df.dropna(subset=["valence", "arousal", "dominance"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Per-Speaker Emotion (neural diarizer) — {stem}  "
                 f"({k} speakers, {ok['t_end'].max():.0f}s)",
                 fontsize=14, fontweight="bold")
    color = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(by_spk)}

    def centroids(ax, ix, iy):
        for emo in namer.labels:
            c = namer._centroids[emo]
            ax.scatter(c[ix], c[iy], marker="*", s=160, c="black", zorder=5)
            ax.annotate(emo, (c[ix], c[iy]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", zorder=6)

    for ax, (iy, yl) in zip(axes[0], [(1, "arousal"), (2, "dominance")]):
        for spk in by_spk:
            sub = ok[ok["speaker"] == spk]
            ax.scatter(sub["valence"], sub[yl], s=18, c=color[spk],
                       label=spk, alpha=0.7)
        centroids(ax, 0, iy)
        ax.set_xlabel("valence"); ax.set_ylabel(yl)
        ax.set_title(f"valence–{yl}"); ax.legend(fontsize=8)

    for ax, dim in zip(axes[1], ["arousal", "valence"]):
        for spk in by_spk:
            sub = ok[ok["speaker"] == spk]
            ax.scatter(sub["t_center"], sub[dim], s=14, c=color[spk],
                       label=spk, alpha=0.7)
        ax.set_xlabel("time (s)"); ax.set_ylabel(dim)
        ax.set_title(f"{dim} over time"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / f"neural_diarized_emotion_{stem}.png", dpi=130)
    print(f"\nSaved → out/neural_diarized_{stem}.csv")
    print(f"Saved → out/neural_diarized_emotion_{stem}.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--regressor", default="models/dim_svr_msp_final")
    ap.add_argument("--namer", default="models/namer_msp_final")
    ap.add_argument("--speakers", type=int, default=None)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    args = ap.parse_args()

    stem = Path(args.input).stem
    labels_csv = OUT_DIR / f"neural_labels_{stem}.csv"
    run_neural_diarizer(args.input, labels_csv, args.window, args.hop, args.speakers)

    ldf = pd.read_csv(labels_csv).sort_values("window_index")
    labels = ldf["speaker"].to_numpy()
    k = len(set(labels))
    print(f"\n← distinguisher returned {k} speaker(s); reading emotion per speaker...\n")

    eng = TrajectoryEngine.from_saved(args.regressor, args.namer,
                                      window_s=args.window, hop_s=args.hop)
    by_spk = eng.analyze_by_speaker(args.input, labels=labels)
    summarize(by_spk)
    render(by_spk, CentroidNamer.load(args.namer), stem, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
