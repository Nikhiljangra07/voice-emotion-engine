"""Generate spectrograms for 20 RAVDESS sample files.

Produces a grid of mel spectrograms organized by emotion,
plus a pandas summary of key features.

Usage:
    python scripts/spectrogram_20.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.feature_vector import build_feature_vector
from src.preprocessing import preprocess

# ── RAVDESS emotion mapping ──────────────────────────────────────────
RAVDESS_EMOTION: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


def select_20_files() -> list[dict[str, str]]:
    """Pick ~2-3 files per emotion from RAVDESS."""
    all_files = sorted(glob.glob("data/ravdess/Actor_*/03-01-*.wav"))

    by_emo: dict[str, list[str]] = {}
    for f in all_files:
        code = Path(f).stem.split("-")[2]
        emo = RAVDESS_EMOTION.get(code, "unknown")
        by_emo.setdefault(emo, []).append(f)

    selected: list[dict[str, str]] = []
    for emo in sorted(by_emo):
        for f in by_emo[emo][:3]:
            if len(selected) >= 20:
                break
            parts = Path(f).stem.split("-")
            selected.append({
                "path": f,
                "emotion": emo,
                "actor": f"Actor_{parts[6]}",
                "intensity": "normal" if parts[3] == "01" else "strong",
            })
        if len(selected) >= 20:
            break

    return selected


def plot_spectrograms(samples: list[dict[str, str]]) -> Path:
    """Generate a 4x5 grid of mel spectrograms, one per file."""
    fig, axes = plt.subplots(4, 5, figsize=(22, 14))
    fig.suptitle(
        "Mel Spectrograms — 20 RAVDESS Samples by Emotion",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Color map per emotion for visual grouping
    emo_colors: dict[str, str] = {
        "anger": "#e74c3c",
        "calm": "#3498db",
        "disgust": "#8e44ad",
        "fear": "#e67e22",
        "joy": "#2ecc71",
        "neutral": "#95a5a6",
        "sadness": "#2c3e50",
        "surprise": "#f1c40f",
    }

    for idx, sample in enumerate(samples):
        row, col = divmod(idx, 5)
        ax = axes[row][col]

        y, sr = preprocess(sample["path"])

        # Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000, hop_length=256,
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(
            S_dB, sr=sr, hop_length=256, x_axis="time", y_axis="mel",
            ax=ax, cmap="magma",
        )

        emo = sample["emotion"]
        color = emo_colors.get(emo, "#333333")
        ax.set_title(
            f"{emo.upper()}\n{sample['actor']} ({sample['intensity']})",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
        ax.set_xlabel("")
        ax.set_ylabel("" if col > 0 else "Hz")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "spectrograms_20_ravdess.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_summary_df(samples: list[dict[str, str]]) -> pd.DataFrame:
    """Build a pandas DataFrame with key features for all 20 files."""
    rows: list[dict[str, object]] = []

    for sample in samples:
        features = build_feature_vector(sample["path"])
        rows.append({
            "file": Path(sample["path"]).name,
            "emotion": sample["emotion"],
            "actor": sample["actor"],
            "intensity": sample["intensity"],
            "F0_semitones": round(features["F0semitoneFrom27.5Hz_sma3nz_amean"], 2),
            "jitter_opensmile": round(features["jitterLocal_sma3nz_amean"], 4),
            "shimmer_dB": round(features["shimmerLocaldB_sma3nz_amean"], 4),
            "HNR_dB": round(features["HNRdBACF_sma3nz_amean"], 2),
            "loudness": round(features["loudness_sma3_amean"], 3),
            "praat_jitter": round(features["praat_jitter_local"], 4),
            "praat_shimmer": round(features["praat_shimmer_local"], 4),
            "praat_HNR": round(features["praat_hnr_mean"], 2),
            "praat_F0_Hz": round(features["praat_f0_mean_hz"], 1),
            "speech_rate": round(features["speech_rate"], 2),
            "pause_ratio": round(features["pause_ratio"], 3),
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("Selecting 20 RAVDESS files...")
    samples = select_20_files()
    print(f"Selected {len(samples)} files\n")

    print("Generating spectrograms...")
    spec_path = plot_spectrograms(samples)
    print(f"Saved: {spec_path}\n")

    print("Extracting features into pandas DataFrame...")
    df = build_summary_df(samples)

    csv_path = OUT_DIR / "features_20_ravdess.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}\n")

    # Print full table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print("=" * 120)
    print("FEATURE SUMMARY (20 files)")
    print("=" * 120)
    print(df.to_string(index=False))

    # Per-emotion aggregation
    print("\n")
    print("=" * 120)
    print("PER-EMOTION MEANS")
    print("=" * 120)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    means = df.groupby("emotion")[numeric_cols].mean().round(3)
    print(means.to_string())

    print(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
