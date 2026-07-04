"""Prosody features: speech rate, pause ratio, pitch dynamics.

These are derived features that complement the openSMILE and parselmouth
extractors with higher-level temporal/rhythmic information.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src.preprocessing import TARGET_SR, AudioError, ErrorCode, preprocess


def extract_prosody(path: str | Path) -> dict[str, float]:
    """Extract prosody features from an audio file.

    Args:
        path: Path to audio file.

    Returns:
        Dict with prosody features.
    """
    y, sr = preprocess(path)
    return extract_prosody_from_array(y, sr)


def extract_prosody_from_array(
    y: np.ndarray,
    sr: int = TARGET_SR,
) -> dict[str, float]:
    """Extract prosody features from a preprocessed audio array.

    Args:
        y: 1-D float32 audio array.
        sr: Sample rate.

    Returns:
        Dict with keys: speech_rate, pause_ratio, pause_count,
        mean_pause_duration_s, voiced_fraction, tempo_bpm.
    """
    duration = len(y) / sr
    if duration < 0.1:
        raise AudioError(
            f"Audio too short for prosody extraction: {duration:.3f}s "
            f"(minimum 0.1s)",
            code=ErrorCode.TOO_SHORT,
        )

    # ── Speech rate (onset-based syllable proxy) ─────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=256,
        backtrack=True,
        delta=0.3,
    )
    speech_rate = len(onsets) / duration

    # ── Pause detection (RMS-based) ──────────────────────────────
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    frame_dur = hop / sr  # seconds per frame

    # Threshold: 20th percentile of RMS (adaptive to signal level).
    threshold = float(np.percentile(rms, 20))
    is_silent = rms < threshold

    # Count contiguous silent regions as pauses.
    pause_lengths: list[int] = []
    current_pause = 0
    for silent in is_silent:
        if silent:
            current_pause += 1
        else:
            if current_pause > 0:
                pause_lengths.append(current_pause)
            current_pause = 0
    if current_pause > 0:
        pause_lengths.append(current_pause)

    total_silent_frames = int(np.sum(is_silent))
    pause_ratio = total_silent_frames / len(rms) if len(rms) > 0 else 0.0
    pause_count = len(pause_lengths)
    mean_pause_dur = (
        float(np.mean(pause_lengths)) * frame_dur
        if pause_lengths
        else 0.0
    )

    # ── Voiced fraction ──────────────────────────────────────────
    # Ratio of frames above the silence threshold.
    voiced_fraction = 1.0 - pause_ratio

    # ── Tempo ────────────────────────────────────────────────────
    tempo_arr = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)[0]
    if isinstance(tempo_arr, np.ndarray):
        tempo_bpm = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0
    else:
        tempo_bpm = float(tempo_arr)

    # ── Energy envelope (attack/decay) ─────────────────────────
    attack_time, decay_time, attack_slope, sustain_ratio = (
        _energy_envelope(rms, frame_dur)
    )

    return {
        "speech_rate": float(speech_rate),
        "pause_ratio": float(pause_ratio),
        "pause_count": int(pause_count),
        "mean_pause_duration_s": float(mean_pause_dur),
        "voiced_fraction": float(voiced_fraction),
        "tempo_bpm": float(tempo_bpm),
        "attack_time_s": float(attack_time),
        "decay_time_s": float(decay_time),
        "attack_slope": float(attack_slope),
        "sustain_ratio": float(sustain_ratio),
    }


def _energy_envelope(
    rms: np.ndarray,
    frame_dur: float,
) -> tuple[float, float, float, float]:
    """Compute energy envelope features from the RMS contour.

    Finds the peak RMS frame, then measures:
      - attack_time_s: time from onset (10% of peak) to peak.
      - decay_time_s: time from peak to offset (10% of peak after peak).
      - attack_slope: peak RMS / attack_time (how sharply energy rises).
      - sustain_ratio: fraction of frames above 50% of peak RMS
        (how much energy is sustained vs transient).

    Args:
        rms: 1-D RMS energy contour (from librosa).
        frame_dur: Duration of each frame in seconds.

    Returns:
        (attack_time_s, decay_time_s, attack_slope, sustain_ratio)
    """
    if len(rms) < 3:
        return 0.0, 0.0, 0.0, 0.0

    peak_idx = int(np.argmax(rms))
    peak_val = float(rms[peak_idx])

    if peak_val < 1e-9:
        return 0.0, 0.0, 0.0, 0.0

    onset_threshold = 0.1 * peak_val

    # Attack: find first frame before peak that crosses 10% of peak.
    onset_idx = 0
    for i in range(peak_idx, -1, -1):
        if rms[i] < onset_threshold:
            onset_idx = i
            break

    attack_frames = peak_idx - onset_idx
    attack_time = attack_frames * frame_dur

    # Decay: find first frame after peak that drops below 10% of peak.
    offset_idx = len(rms) - 1
    for i in range(peak_idx, len(rms)):
        if rms[i] < onset_threshold:
            offset_idx = i
            break

    decay_frames = offset_idx - peak_idx
    decay_time = decay_frames * frame_dur

    # Attack slope: how sharply energy rises (RMS units per second).
    attack_slope = peak_val / attack_time if attack_time > 0 else peak_val / frame_dur

    # Sustain ratio: fraction of all frames above 50% of peak.
    sustain_threshold = 0.5 * peak_val
    sustained_frames = int(np.sum(rms >= sustain_threshold))
    sustain_ratio = sustained_frames / len(rms)

    return attack_time, decay_time, attack_slope, sustain_ratio


def _empty_prosody() -> dict[str, float]:
    """Return zeroed prosody features for too-short audio."""
    return {
        "speech_rate": 0.0,
        "pause_ratio": 0.0,
        "pause_count": 0,
        "mean_pause_duration_s": 0.0,
        "voiced_fraction": 0.0,
        "tempo_bpm": 0.0,
        "attack_time_s": 0.0,
        "decay_time_s": 0.0,
        "attack_slope": 0.0,
        "sustain_ratio": 0.0,
    }
