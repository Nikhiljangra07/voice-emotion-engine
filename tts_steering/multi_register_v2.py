"""MULTI-REGISTER SPEECH + AUTO-RETRY CERTIFICATION — the cloning upgrade.

Two product mechanisms, first combined run:
1. REGISTER-MATCHED PROMPTS (fixes register-bound identity): each act is
   cloned from the user recording whose register matches the act's energy —
   calm acts from the neutral memo, high-energy acts from the excited memo,
   dark acts from the sad memo. All prompts WPE'd + normalized (room law).
   NINTH BANK AS ALLY: the P5A joy vector (Σw=−0.06) amplifies prompt
   residue at 1.06x — fed the EXCITED prompt (V=+0.43), that residue is now
   fuel, not sabotage.
2. AUTO-RETRY CERTIFICATION (fixes draw noise): synthesize all acts, judge
   all, re-synthesize only uncertified acts (fresh draw), up to 3 rounds.
   Final speech assembles the first certified take per act (or the take
   closest to the target centroid if never certified). Judge frozen; every
   attempt a ledger row (indextts2-multi-register-v2).

Run:  .venv_tts/bin/python tts_steering/multi_register_speech.py
Play: afplay tts_steering/out/user_speech/multi_register_speech_v2.wav
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

P_NEU = str(ROOT / "own_voice/reg_neutral_v2.wav")
P_JOY = str(ROOT / "own_voice/reg_joy_v2.wav")
P_SAD = str(ROOT / "own_voice/reg_sad_v2.wav")

CENTROIDS = {"sadness": (-0.28, 0.39, -0.07), "joy": (0.30, 0.58, 0.19),
             "anger": (-0.42, 0.70, 0.41), "surprise": (0.05, 0.64, 0.26),
             "fear": (-0.21, 0.51, -0.01), "neutral": (-0.04, 0.45, 0.03)}

V = {
    "neutral":  [0.0] * 8,
    "joy":      [0.409, -0.266, 0.022, 0.131, 0.058, -0.19, -0.23, 0.003],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "fear":     [0.115, -0.286, 0.319, 0.296, 0.069, 0.165, 0.071, -0.278],
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "sadness":  [0.007, 0.063, 0.005, 0.052, 0.079, 0.784, -0.095, 0.206],
}

# (emotion, register-matched prompt, text)
ACTS = [
    ("neutral", P_NEU,
     "Let me tell you about the strangest day of my life. It started like "
     "any other morning. I made my coffee, read the paper for a while, and "
     "watched the quiet street outside my window. Nothing in the world "
     "suggested this day would be any different from the hundred ordinary "
     "days before it."),
    ("joy", P_JOY,
     "Then the phone rang, and it was the news we had been waiting months "
     "to hear. She said yes. She actually said yes! I ran down the stairs "
     "two at a time, laughing at nothing, grinning at strangers, and the "
     "whole way home I could not stop smiling."),
    ("surprise", P_NEU,  # headroom law: the spike needs a cold baseline
     "But when I reached my front door, something was wrong. The lights "
     "were already on inside? I pushed the door open slowly, and suddenly "
     "forty people jumped out from behind my furniture? My brother flew in "
     "from overseas without telling me? I could not believe my eyes."),
    ("fear", P_SAD,
     "And then, all at once, the lights went out. The whole house went "
     "black and completely silent. I heard footsteps moving slowly across "
     "the floor, coming closer and closer. My hands would not stop "
     "shaking, and I pressed my back against the cold wall, too afraid to "
     "breathe."),
    ("anger", P_JOY,
     "Then I saw who was holding the flashlight, and everything in me "
     "boiled over. You broke your promise again, after everything we "
     "talked about, after everything I forgave. I have had enough of the "
     "excuses and enough of the lies. This ends today, right now."),
    ("sadness", P_SAD,
     "After everyone finally left, the house felt emptier than it had "
     "ever felt. I walked from room to room turning off the lights, and I "
     "kept finding small things she had left behind in the drawers. "
     "Nothing in this house has felt the same since that night."),
    ("joy", P_JOY,
     "But you know what? When I think about that whole impossible day "
     "now, I have to smile all over again. Every one of those people came "
     "for me. And this weekend we are all going to be together again, "
     "around one loud, crowded, happy table. I cannot wait."),
]

MAX_ROUNDS = 3


def synth(jobs):
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if not todo:
        return
    jf = OUT / "_jobsmr2.json"
    jf.write_text(json.dumps(todo))
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
         str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                              "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=7200)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")


def dist(m, emo):
    c = CENTROIDS[emo]
    return ((m["V"] - c[0]) ** 2 + (m["A"] - c[1]) ** 2
            + (m["D"] - c[2]) ** 2) ** 0.5


def main():
    best = {}          # act index -> (wav, verdict-dict, hit, d)
    n = sum(1 for _ in open(LEDGER)) - 1
    rows = []
    t0 = time.time()
    for rnd in range(1, MAX_ROUNDS + 1):
        pending = [i for i in range(len(ACTS))
                   if i not in best or not best[i][2]]
        if not pending:
            break
        jobs, meta = [], []
        for i in pending:
            emo, prompt, text = ACTS[i]
            wav = OUT / f"mr2_a{i+1}_{emo}_t{rnd}.wav"
            jobs.append({"prompt": prompt, "text": text,
                         "vector": V[emo], "out": str(wav)})
            meta.append((i, emo, str(wav)))
        print(f"round {rnd}: synthesizing {len(jobs)} act(s) "
              f"({[ACTS[i][0] for i in pending]}) ...", flush=True)
        synth(jobs)
        res = judge([w for _, _, w in meta])
        for (i, emo, wav), m in zip(meta, res):
            hit = m["judge_family"] == emo
            d = dist(m, emo)
            n += 1
            rows.append([n, "indextts2-multi-register-v2", emo,
                         f"raw_emo_vector={V[emo]};act{i+1};round{rnd};"
                         f"reg_prompt={Path(ACTS[i][1]).stem}",
                         round(m["V"], 3), round(m["A"], 3),
                         round(m["D"], 3), m["judge_family"],
                         round(m["judge_confidence"], 2), round(d, 3),
                         int(hit)])
            cur = best.get(i)
            if cur is None or (hit and not cur[2]) or \
               (hit == cur[2] and d < cur[3]):
                best[i] = (wav, m, hit, d)
            mark = "CERTIFIED" if hit else m["judge_family"]
            print(f"  act {i+1} [{emo:8s}] -> {mark}"
                  f"@{m['judge_confidence']:.0%} V={m['V']:+.2f} "
                  f"A={m['A']:.2f} d={d:.2f}", flush=True)
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)

    sil = OUT / "_sil25.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=22050:cl=mono", "-t", "0.25", str(sil)],
                   check=True)
    lines = []
    for i in range(len(ACTS)):
        lines.append(f"file '{best[i][0]}'")
        lines.append(f"file '{sil}'")
    concat = OUT / "_concatmr2.txt"
    concat.write_text("\n".join(lines[:-1]))
    full = OUT / "multi_register_speech_v2.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "concat", "-safe",
                    "0", "-i", str(concat), "-ar", "22050", str(full)],
                   check=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(full)], capture_output=True,
        text=True).stdout)

    certified = sum(1 for i in best if best[i][2])
    print(f"\nFINAL ASSEMBLY ({dur/60:.2f} min):")
    for i in range(len(ACTS)):
        wav, m, hit, d = best[i]
        print(f"  act {i+1} [{ACTS[i][0]:8s}] {Path(wav).name:24s} "
              f"{'CERTIFIED' if hit else 'best-effort (' + m['judge_family'] + ')'}")
    print(f"\ncertified acts: {certified}/{len(ACTS)}  "
          f"total time {(time.time()-t0)/60:.0f}m  ledger total {n}")
    print(f"PLAY IT:  afplay {full}")
    print("MULTI_REGISTER_DONE")


if __name__ == "__main__":
    main()
