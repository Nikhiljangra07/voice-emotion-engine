"""FULL SPEECH IN THE USER'S CLONED VOICE — the congruence law, performed.

Speaker prompt: own_voice/clone_ref_1.wav (11.3s, user's own recording,
provided for this purpose — local only, never published).

Seven acts, one continuous story. Each act = proven vector + CONGRUENT text
(the codified law: resonance emotions get matching words; carriers barely
need them). Acts synthesized separately (the mouth is per-utterance),
concatenated with 0.4s gaps, each act judged and ledgered
(indextts2-user-voice-speech). Judge frozen.

Run:  .venv_tts/bin/python tts_steering/user_voice_speech.py
Play: afplay tts_steering/out/user_speech/full_speech.wav
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "out/user_speech"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
PROMPT = str(ROOT / "own_voice/clone_ref_1.wav")

V = {
    "neutral":  [0.0] * 8,
    "joy":      [0.409, -0.266, 0.022, 0.131, 0.058, -0.19, -0.23, 0.003],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "fear":     [0.115, -0.286, 0.319, 0.296, 0.069, 0.165, 0.071, -0.278],
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "sadness":  [0.007, 0.063, 0.005, 0.052, 0.079, 0.784, -0.095, 0.206],
}

ACTS = [
    ("neutral", "Let me tell you about the strangest day of my life. "
                "It started like any other morning, with coffee and the "
                "usual quiet streets."),
    ("joy", "Then the phone rang with the news we had waited months to "
            "hear. She said yes, and I could not stop smiling the whole "
            "way home."),
    ("surprise", "But when I opened my front door, the lights were "
                 "already on? Out of everyone I knew, they had all been "
                 "hiding in my living room?"),
    ("fear", "Then the lights went out all at once. I heard footsteps "
             "moving slowly in the dark, and my hands would not stop "
             "shaking."),
    ("anger", "And then I saw who it was. You broke your promise again "
              "after everything we talked about. I have had enough of "
              "these excuses, this ends right now."),
    ("sadness", "After everyone left, the house felt empty again. I kept "
                "finding small things she left behind in the drawers, "
                "and nothing has felt the same since."),
    ("joy", "Still, when I think about that whole day now, I have to "
            "smile. We are going to see them all again this weekend."),
]


def main():
    jobs, meta = [], []
    for i, (emo, text) in enumerate(ACTS, 1):
        wav = OUT / f"act{i}_{emo}.wav"
        if not wav.exists():
            jobs.append({"prompt": PROMPT, "text": text, "vector": V[emo],
                         "out": str(wav)})
        meta.append((i, emo, str(wav)))
    if jobs:
        jf = OUT / "_jobs.json"
        jf.write_text(json.dumps(jobs))
        print(f"synthesizing {len(jobs)} acts in the user's voice ...",
              flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
             str(jf)],
            cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                                  "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=7200)
        if "WORKER_DONE" not in proc.stdout:
            raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # concatenate with 0.4s gaps
    concat = OUT / "_concat.txt"
    sil = OUT / "_sil.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=22050:cl=mono", "-t", "0.4", str(sil)],
                   check=True)
    lines = []
    for _, _, w in meta:
        lines.append(f"file '{w}'")
        lines.append(f"file '{sil}'")
    concat.write_text("\n".join(lines[:-1]))
    full = OUT / "full_speech.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "concat", "-safe",
                    "0", "-i", str(concat), "-ar", "22050", str(full)],
                   check=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(full)], capture_output=True,
        text=True).stdout)
    print(f"full speech: {full}  ({dur:.1f}s)", flush=True)

    print("\njudging each act (the loop closes on your voice):", flush=True)
    res = judge([w for _, _, w in meta])
    rows = []
    n = sum(1 for _ in open(LEDGER)) - 1
    hits = 0
    for (i, emo, w), m in zip(meta, res):
        hit = m["judge_family"] == emo
        hits += hit
        n += 1
        rows.append([n, "indextts2-user-voice-speech", emo,
                     f"raw_emo_vector={V[emo]};act{i};congruent;spk=user",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     "", int(hit)])
        print(f"  act {i} [{emo:8s}] -> {m['judge_family']}"
              f"@{m['judge_confidence']:.0%}  V={m['V']:+.2f} "
              f"A={m['A']:.2f}  {'✓' if hit else ''}")
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nacts confirmed: {hits}/{len(meta)}  "
          f"(ledger +{len(rows)}, total {n})")
    print(f"PLAY IT:  afplay {full}")
    print("USER_SPEECH_DONE")


if __name__ == "__main__":
    main()
