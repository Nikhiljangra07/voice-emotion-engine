"""Dataset loaders for RAVDESS, CREMA-D, and MELD.

Parses filenames into Ekman-6 labels and returns structured metadata.
This module lives in utils/ (dataset-specific) — the engine layer
(src/features/) is dataset-agnostic.

Ekman-6 families: anger, fear, sadness, joy, disgust, surprise.
"""

from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ── Ekman-6 label constants ──────────────────────────────────────────

EKMAN6_LABELS: list[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

# ── RAVDESS ──────────────────────────────────────────────────────────
# Filename: {Modality}-{VocalChannel}-{Emotion}-{Intensity}-{Statement}-{Repetition}-{Actor}.wav
# Emotion codes:
#   01=neutral, 02=calm, 03=happy, 04=sad,
#   05=angry, 06=fearful, 07=disgust, 08=surprised

_RAVDESS_EMOTION_MAP: dict[str, str | None] = {
    "01": None,         # neutral — not in Ekman-6
    "02": None,         # calm — not in Ekman-6
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


@dataclass
class AudioSample:
    """Metadata for a single audio file."""

    path: str
    label: str          # Ekman-6 emotion
    dataset: str        # "ravdess" or "crema_d"
    actor: str          # Speaker identifier
    intensity: str      # "normal", "strong", "low", "medium", "high", "unspecified"


def load_ravdess(data_dir: str | Path = "data/ravdess") -> list[AudioSample]:
    """Load RAVDESS dataset with Ekman-6 labels.

    Skips neutral (01) and calm (02) — they are not Ekman-6 emotions.
    Only loads speech files (modality 03).

    Args:
        data_dir: Root directory containing Actor_XX folders.

    Returns:
        List of AudioSample with Ekman-6 labels.

    Raises:
        FileNotFoundError: If data_dir does not exist or has no WAVs.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"RAVDESS directory not found: {data_dir}")

    wav_files = sorted(glob.glob(str(data_dir / "Actor_*" / "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {data_dir}")

    samples: list[AudioSample] = []
    for path in wav_files:
        parts = Path(path).stem.split("-")
        if len(parts) < 7:
            continue

        # Only speech (modality 03).
        if parts[0] != "03":
            continue

        emotion_code = parts[2]
        label = _RAVDESS_EMOTION_MAP.get(emotion_code)
        if label is None:
            # Skip neutral and calm.
            continue

        intensity = "normal" if parts[3] == "01" else "strong"
        actor = f"ravdess_{parts[6]}"

        samples.append(AudioSample(
            path=path,
            label=label,
            dataset="ravdess",
            actor=actor,
            intensity=intensity,
        ))

    return samples


# ── CREMA-D ──────────────────────────────────────────────────────────
# Filename: {Actor}_{Sentence}_{Emotion}_{Level}.wav
# Emotion codes: ANG, DIS, FEA, HAP, NEU, SAD
# Level: HI, MD, LO, XX (unspecified)

_CREMAD_EMOTION_MAP: dict[str, str | None] = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": None,        # neutral — not in Ekman-6
    "SAD": "sadness",
}

_CREMAD_INTENSITY_MAP: dict[str, str] = {
    "HI": "high",
    "MD": "medium",
    "LO": "low",
    "XX": "unspecified",
}


def load_crema_d(data_dir: str | Path = "data/crema_d/audios") -> list[AudioSample]:
    """Load CREMA-D dataset with Ekman-6 labels.

    Skips neutral (NEU) — not in Ekman-6.
    Note: CREMA-D has no "surprise" emotion.

    Args:
        data_dir: Directory containing WAV files.

    Returns:
        List of AudioSample with Ekman-6 labels.

    Raises:
        FileNotFoundError: If data_dir does not exist or has no WAVs.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"CREMA-D directory not found: {data_dir}")

    wav_files = sorted(glob.glob(str(data_dir / "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {data_dir}")

    samples: list[AudioSample] = []
    for path in wav_files:
        parts = Path(path).stem.split("_")
        if len(parts) < 4:
            continue

        emotion_code = parts[2]
        label = _CREMAD_EMOTION_MAP.get(emotion_code)
        if label is None:
            # Skip neutral.
            continue

        intensity = _CREMAD_INTENSITY_MAP.get(parts[3], "unspecified")
        actor = f"cremad_{parts[0]}"

        samples.append(AudioSample(
            path=path,
            label=label,
            dataset="crema_d",
            actor=actor,
            intensity=intensity,
        ))

    return samples


# ── MELD ──────────────────────────────────────────────────────────────
# CSV columns: Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,
#              Utterance_ID,Season,Episode,StartTime,EndTime
# Emotion labels: neutral, joy, sadness, anger, fear, disgust, surprise
# Audio: FLAC at 16kHz, filename = dia{Dialogue_ID}_utt{Utterance_ID}.flac

_MELD_EKMAN6: set[str] = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}


def load_meld(data_dir: str | Path = "data/meld") -> list[AudioSample]:
    """Load MELD dataset with Ekman-6 labels.

    Combines train, dev, and test splits. Skips neutral.

    Args:
        data_dir: Root MELD directory containing CSVs and audio/.

    Returns:
        List of AudioSample with Ekman-6 labels.

    Raises:
        FileNotFoundError: If data_dir does not exist or has no CSVs.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"MELD directory not found: {data_dir}")

    audio_base = data_dir / "audio"
    if not audio_base.exists():
        raise FileNotFoundError(
            f"MELD audio directory not found: {audio_base}. "
            f"Extract the tar.gz archives first."
        )

    samples: list[AudioSample] = []

    for split in ("train", "dev", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emotion = row["Emotion"].strip().lower()
                if emotion not in _MELD_EKMAN6:
                    continue

                dialogue_id = row["Dialogue_ID"].strip()
                utterance_id = row["Utterance_ID"].strip()
                filename = f"dia{dialogue_id}_utt{utterance_id}.flac"
                audio_path = audio_base / split / filename

                if not audio_path.exists():
                    continue

                speaker = row.get("Speaker", "unknown").strip()
                actor = f"meld_{speaker}"

                samples.append(AudioSample(
                    path=str(audio_path),
                    label=emotion,
                    dataset="meld",
                    actor=actor,
                    intensity="unspecified",
                ))

    return samples


# ── Combined loader ──────────────────────────────────────────────────

def load_all(
    ravdess_dir: str | Path = "data/ravdess",
    crema_d_dir: str | Path = "data/crema_d/audios",
    meld_dir: str | Path = "data/meld",
    include_meld: bool = True,
) -> list[AudioSample]:
    """Load RAVDESS, CREMA-D, and optionally MELD with Ekman-6 labels.

    Args:
        ravdess_dir: RAVDESS root directory.
        crema_d_dir: CREMA-D audio directory.
        meld_dir: MELD root directory.
        include_meld: Whether to include MELD.

    Returns:
        Combined list of AudioSample from all datasets.
    """
    samples: list[AudioSample] = []
    samples.extend(load_ravdess(ravdess_dir))
    samples.extend(load_crema_d(crema_d_dir))
    if include_meld and Path(meld_dir).exists():
        samples.extend(load_meld(meld_dir))
    return samples


def to_dataframe(samples: list[AudioSample]) -> pd.DataFrame:
    """Convert samples to a pandas DataFrame.

    Columns: path, label, dataset, actor, intensity.
    """
    return pd.DataFrame([
        {
            "path": s.path,
            "label": s.label,
            "dataset": s.dataset,
            "actor": s.actor,
            "intensity": s.intensity,
        }
        for s in samples
    ])
