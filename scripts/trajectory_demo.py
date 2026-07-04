"""P2.4 demo: emotion trajectory over a full recording.

Runs the saved MSP regressor + namer over a long audio file window-by-window
and writes a timestamped trajectory (CSV) + a console summary.

NOTE (Law 18): this is a DEMO/visualization. MSP labels are per-segment, so
per-window accuracy is NOT validated here — that needs a continuous corpus.

Usage:
    python -m scripts.trajectory_demo --input own_voice/long_1.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.dimensional.trajectory import TrajectoryEngine, trajectory_to_rows

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="own_voice/long_1.wav")
    ap.add_argument("--regressor", default="models/dim_svr_msp_final")
    ap.add_argument("--namer", default="models/namer_msp_final")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args()

    eng = TrajectoryEngine.from_saved(
        args.regressor, args.namer,
        window_s=args.window, hop_s=args.hop, smoothing_alpha=args.alpha,
    )
    print(f"Analyzing {args.input} (window {args.window}s / hop {args.hop}s)...")
    points = eng.analyze(args.input)
    rows = trajectory_to_rows(points)
    df = pd.DataFrame(rows)

    stem = Path(args.input).stem
    csv = OUT_DIR / f"trajectory_{stem}.csv"
    df.to_csv(csv, index=False)

    gaps = int(df["error"].notna().sum())
    ok = df[df["error"].isna()]
    print(f"\nWindows: {len(df)}  (gaps: {gaps})")
    print(f"Duration covered: {df['t_end'].max():.1f}s")
    if not ok.empty:
        print(f"\nPAD means (normalized): "
              f"V={ok['valence'].mean():+.2f}  A={ok['arousal'].mean():.2f}  "
              f"D={ok['dominance'].mean():+.2f}")
        print(f"Mean intensity (radius): {ok['intensity'].mean():.2f}")
        print(f"Ambiguous windows: {int(ok['ambiguous'].sum())}/{len(ok)}")
        print("\nEmotion time-share:")
        for emo, n in ok["emotion"].value_counts().items():
            print(f"  {emo:>9s}: {n:>4d} windows ({100*n/len(ok):.0f}%)")
        # Compact emotion timeline (collapse consecutive repeats).
        seq = ok.sort_values("t_center")[["t_start", "emotion"]].values.tolist()
        timeline, last = [], None
        for t, e in seq:
            if e != last:
                timeline.append(f"{t:.0f}s:{e}")
                last = e
        print("\nEmotion timeline (transitions):")
        print("  " + " → ".join(timeline[:40]))
        if len(timeline) > 40:
            print(f"  ... (+{len(timeline)-40} more transitions)")

    print(f"\nSaved trajectory → {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
