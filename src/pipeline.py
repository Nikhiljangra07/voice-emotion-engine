"""End-to-end pipeline: audio file → emotion prediction → JSON output.

Usage:
    python -m src.pipeline --input audio.wav
    python -m src.pipeline --input audio.wav --model models/svm_ravdess
    python -m src.pipeline --input audio.wav --baseline neutral_sample.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.classifier.predict import Predictor
from src.signal_mapper import map_signal
from src.speaker_baseline import SpeakerBaseline

DEFAULT_MODEL_DIR = Path("models/svm_ravdess")


def run(
    audio_path: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    speaker_baseline: SpeakerBaseline | None = None,
    trim: bool = True,
    top_db: float = 25.0,
) -> dict:
    """Run the full pipeline on a single audio file.

    Audio → preprocessing → features → classifier → signal mapper → JSON.

    Args:
        audio_path: Path to audio file.
        model_dir: Path to saved model directory.
        speaker_baseline: Optional per-speaker baseline for
            expressionStrength. If None, uses global RAVDESS baseline.
        trim: Whether to trim leading/trailing silence. Set to False
            for whispered or very soft emotional speech.
        top_db: Silence threshold in dB below peak RMS. Lower values
            trim more aggressively. Only used when *trim* is True.

    Returns:
        Dict with: valence, arousal, expressionStrength, confidence,
        emotion, ekman6_weights, features.

    Raises:
        FileNotFoundError: If audio file or model directory doesn't exist.
    """
    audio_path = Path(audio_path)
    model_dir = Path(model_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found: {model_dir}. "
            f"Train a model first with src.classifier.train."
        )

    predictor = Predictor.from_saved(model_dir)
    prediction = predictor.predict(audio_path, trim=trim, top_db=top_db)
    return map_signal(prediction, speaker_baseline=speaker_baseline)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Voice Emotion Engine — audio → emotion prediction",
    )
    parser.add_argument(
        "--input", required=True, help="Path to audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_DIR),
        help=f"Path to saved model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to speaker's neutral audio file for per-speaker "
             "expressionStrength calibration",
    )
    parser.add_argument(
        "--no-trim", action="store_true",
        help="Disable silence trimming (preserve soft/whispered speech)",
    )
    parser.add_argument(
        "--top-db", type=float, default=25.0,
        help="Silence threshold in dB (default: 25.0, lower = more aggressive)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    try:
        baseline = None
        if args.baseline:
            baseline_path = Path(args.baseline)
            if not baseline_path.exists():
                print(f"Error: Baseline file not found: {baseline_path}", file=sys.stderr)
                return 1
            baseline = SpeakerBaseline.from_audio(baseline_path)

        result = run(
            args.input, args.model,
            speaker_baseline=baseline,
            trim=not args.no_trim,
            top_db=args.top_db,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
