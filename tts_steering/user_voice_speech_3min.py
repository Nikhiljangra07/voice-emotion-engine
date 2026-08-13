"""3-MINUTE CONTINUOUS SPEECH in the user's cloned voice — one file.

Ninth-bank lesson applied (see 2026-08-13 log): the user's prompt reads
melancholic, and subtract-heavy vectors amplify prompt residue at (1−Σw).
So in THIS speech:
  * joy acts    -> emo_audio REFERENCE channel (RAVDESS happy, alpha 1.0)
                   — bypasses the Σw arithmetic entirely (P4.6 joyref route)
  * other acts  -> proven vectors, all with bank-mass Σw ≥ 0.47 (fear) and
                   most ≥ 0.8 — bank-dominant, prompt-immune
  * neutral open -> zero vector: inherits the prompt's calm read (documented;
                   acceptable for a storyteller's opening)

Seven acts × ~60-70 words ≈ 25s each, 0.25s joins -> single ~3 min wav.
Every act judged + ledgered (indextts2-user-voice-3min). Judge frozen.

Run:  .venv_tts/bin/python tts_steering/user_voice_speech_3min.py
Play: afplay tts_steering/out/user_speech/full_speech_3min.wav
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
REF_JOY = str(ROOT / "data/ravdess/Actor_01/03-01-03-02-01-01-01.wav")

V = {
    "neutral":  [0.0] * 8,
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "fear":     [0.115, -0.286, 0.319, 0.296, 0.069, 0.165, 0.071, -0.278],
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "sadness":  [0.007, 0.063, 0.005, 0.052, 0.079, 0.784, -0.095, 0.206],
}

ACTS = [
    ("neutral", None,
     "Let me tell you about the strangest day of my life. It started like "
     "any other morning. I made my coffee, read the paper for a while, and "
     "watched the quiet street outside my window. The mail came at nine, "
     "the neighbor walked his old dog past the gate, and nothing in the "
     "world suggested that this day would be any different from the "
     "hundred ordinary days before it."),
    ("joy", "ref",
     "Then the phone rang, and it was the news we had been waiting months "
     "to hear. She said yes. She actually said yes! I ran down the stairs "
     "two at a time, laughing at nothing, grinning at strangers on the "
     "street. The whole way home the sun felt warmer, the city felt "
     "kinder, and I could not stop smiling even when my face began to "
     "ache from it."),
    ("surprise", None,
     "But when I reached my front door, something was wrong. The lights "
     "were already on inside? I never leave the lights on. I pushed the "
     "door open slowly, and suddenly forty people jumped out from behind "
     "my furniture? My brother flew in from overseas without telling me? "
     "Even my old teacher from school was standing in my kitchen? I could "
     "not believe what I was seeing."),
    ("fear", None,
     "And then, all at once, the lights went out. The whole house went "
     "black and completely silent. I heard footsteps moving slowly across "
     "the floor, coming closer and closer to where I stood. My hands "
     "would not stop shaking. Someone whispered my name from the dark "
     "hallway, and I pressed my back against the cold wall, too afraid "
     "to even breathe."),
    ("anger", None,
     "Then I saw who was holding the flashlight, and everything in me "
     "boiled over. You. You broke your promise again, after everything we "
     "talked about, after everything I forgave. I have had enough of the "
     "excuses, enough of the lies, enough of being made a fool of in my "
     "own house. This ends today. This ends right now, and I mean every "
     "single word of it."),
    ("sadness", None,
     "After everyone finally left, the house felt emptier than it had "
     "ever felt before. I walked from room to room turning off the "
     "lights, and I kept finding small things she had left behind in the "
     "drawers. A hair clip. A folded note. Half a photograph. Nothing in "
     "this house has felt the same since that night, and I do not think "
     "it ever will."),
    ("joy", "ref",
     "But you know what? When I think about that whole impossible day "
     "now, I have to smile all over again. Every one of those people came "
     "for me. My brother crossed an ocean for me! And this weekend, we "
     "are all going to be together again, around one loud, crowded, "
     "happy table. I truly cannot wait to see them."),
]


def main():
    jobs, meta = [], []
    for i, (emo, mode, text) in enumerate(ACTS, 1):
        wav = OUT / f"a3_{i}_{emo}.wav"
        if not wav.exists():
            j = {"prompt": PROMPT, "text": text, "out": str(wav)}
            if mode == "ref":
                j["emo_audio"] = REF_JOY
                j["emo_alpha"] = 1.0
            else:
                j["vector"] = V[emo]
            jobs.append(j)
        meta.append((i, emo, mode, str(wav)))
    if jobs:
        jf = OUT / "_jobs3.json"
        jf.write_text(json.dumps(jobs))
        print(f"synthesizing {len(jobs)} long acts in the user's voice ...",
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

    sil = OUT / "_sil25.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=22050:cl=mono", "-t", "0.25", str(sil)],
                   check=True)
    lines = []
    for _, _, _, w in meta:
        lines.append(f"file '{w}'")
        lines.append(f"file '{sil}'")
    concat = OUT / "_concat3.txt"
    concat.write_text("\n".join(lines[:-1]))
    full = OUT / "full_speech_3min.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "concat", "-safe",
                    "0", "-i", str(concat), "-ar", "22050", str(full)],
                   check=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(full)], capture_output=True,
        text=True).stdout)
    print(f"single file: {full}  ({dur/60:.1f} min)", flush=True)

    print("\njudging each act:", flush=True)
    res = judge([w for _, _, _, w in meta])
    rows = []
    n = sum(1 for _ in open(LEDGER)) - 1
    hits = 0
    for (i, emo, mode, w), m in zip(meta, res):
        hit = m["judge_family"] == emo
        hits += hit
        n += 1
        ctrl = ("emo_audio=rav_happy;alpha=1.0" if mode == "ref"
                else f"raw_emo_vector={V[emo]}")
        rows.append([n, "indextts2-user-voice-3min", emo,
                     f"{ctrl};act{i};3min;spk=user",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     "", int(hit)])
        print(f"  act {i} [{emo:8s}{'*' if mode=='ref' else ' '}] -> "
              f"{m['judge_family']}@{m['judge_confidence']:.0%}  "
              f"V={m['V']:+.2f} A={m['A']:.2f}  {'✓' if hit else ''}")
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nacts confirmed: {hits}/{len(meta)} "
          f"(* = reference-channel joy)  ledger total {n}")
    print(f"PLAY IT:  afplay {full}")
    print("USER_SPEECH_3MIN_DONE")


if __name__ == "__main__":
    main()
