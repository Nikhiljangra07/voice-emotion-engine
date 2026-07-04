"""Construct controlled multi-speaker test conversations with EXACT ground truth.

Diarization can only be measured if you know who spoke when. So instead of random
internet clips, we stitch known single-speaker RAVDESS actor clips into
"conversations" and record the true speaker timeline. RAVDESS actor parity: odd
actor number = male, even = female — lets us build easy (mixed-gender) and hard
(same-gender) cases, plus a one-speaker/many-emotion case that directly tests
whether a method falsely splits a person when their emotion changes.

Outputs (own_voice/test_convos/): one WAV + one ground-truth JSON per conversation.

Usage: python -m scripts.build_test_conversations
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RAVDESS = Path("data/ravdess")
OUT = Path("own_voice/test_convos")
SR = 16_000
GAP_S = 0.3  # silence between turns

# Each conversation: ordered list of actor numbers (one RAVDESS clip per turn).
# Odd = male, even = female.
CONVERSATIONS: dict[str, list[int]] = {
    "convo_2spk_diffgender": [1, 2, 1, 2, 1, 2, 1, 2],          # M/F — easy
    "convo_2spk_samegender": [1, 3, 1, 3, 1, 3, 1, 3],          # M/M — hard
    "convo_3spk_mixed":      [1, 2, 5, 1, 2, 5, 1, 2, 5],       # 3 speakers
    "convo_1spk_emotions":   [7, 7, 7, 7, 7, 7],                # ONE speaker, many emotions
    "convo_2spk_diffgender_b": [9, 12, 9, 12, 9, 12, 9, 12],    # different M/F pair
}


def actor_clips(actor: int, rng: random.Random, n: int) -> list[Path]:
    folder = RAVDESS / f"Actor_{actor:02d}"
    clips = sorted(folder.glob("*.wav"))
    if len(clips) < n:
        raise FileNotFoundError(f"{folder}: need {n} clips, found {len(clips)}.")
    return rng.sample(clips, n)


def build(name: str, actors: list[int], rng: random.Random) -> None:
    # Pre-pick distinct clips per actor so repeated turns use different emotions.
    need = {a: actors.count(a) for a in set(actors)}
    pool = {a: actor_clips(a, rng, need[a]) for a in need}
    cursor = {a: 0 for a in need}

    audio: list[np.ndarray] = []
    segments = []
    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    t = 0.0
    speaker_ids = {a: i for i, a in enumerate(sorted(set(actors)))}

    for actor in actors:
        clip = pool[actor][cursor[actor]]; cursor[actor] += 1
        y, sr = sf.read(str(clip), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        peak = np.max(np.abs(y)) or 1.0
        y = (y / peak).astype(np.float32)
        dur = len(y) / SR
        segments.append({"start": round(t, 3), "end": round(t + dur, 3),
                         "speaker": speaker_ids[actor], "actor": actor})
        audio.append(y); audio.append(gap)
        t += dur + GAP_S

    full = np.concatenate(audio)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT / f"{name}.wav"), full, SR)
    truth = {"name": name, "n_speakers": len(set(actors)),
             "duration_s": round(len(full) / SR, 2), "segments": segments}
    (OUT / f"{name}.json").write_text(json.dumps(truth, indent=2))
    print(f"  {name}: {len(actors)} turns, {truth['n_speakers']} speakers, "
          f"{truth['duration_s']}s")


def main() -> int:
    if not RAVDESS.exists():
        print(f"RAVDESS not found at {RAVDESS}"); return 1
    rng = random.Random(0)
    print("Building controlled test conversations...")
    for name, actors in CONVERSATIONS.items():
        build(name, actors, rng)
    print(f"\nSaved to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
