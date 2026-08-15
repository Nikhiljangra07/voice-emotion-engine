"""2-MINUTE NARRATION in the user's cloned voice — calm register only.

The user's finding (2026-08-13): the clone is uncanny in the prompt's own
register (calm, low-arousal) and drifts identity when pushed loud. So this
piece never leaves that register: zero vector throughout = the emovec is
pure speaker inheritance — MAXIMUM identity, no bank interference. A
narrating voice, five reflective paragraphs, 0.3s joins, one file.

Every paragraph judged + ledgered (indextts2-user-narration-dry). Judge frozen.

Run:  .venv_tts/bin/python tts_steering/user_narration.py
Play: afplay tts_steering/out/user_speech/narration_2min_dry.wav
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
PROMPT = str(ROOT / "own_voice/clone_ref_1_dry.wav")

PARAS = [
    "There is a particular kind of quiet that only exists early in the "
    "morning, before the city remembers itself. I have always liked "
    "sitting inside that quiet with a cup of coffee, watching the light "
    "change on the buildings across the street, not thinking about "
    "anything in particular, just letting the day arrive at its own pace.",

    "People ask me sometimes why I build things. I never have a clean "
    "answer for them. The truth is that most of what I make already "
    "exists somewhere in the world, made better by someone with more "
    "money and more time. But that was never really the point. The point "
    "is the making itself, the long afternoons where a problem slowly "
    "turns itself over and shows you its softer side.",

    "There is a moment in every project that I have come to wait for. It "
    "arrives without warning, usually late at night, when something that "
    "was broken for weeks suddenly works, and the screen shows you "
    "exactly what you imagined months ago. Nobody else is awake. Nobody "
    "else would even understand what they were looking at. And still, it "
    "feels like the whole room gets a little brighter.",

    "I have learned to keep a record of everything, the failures "
    "especially. The dead ends teach you more than the victories, if you "
    "write them down honestly. Years from now I will probably not "
    "remember the exact numbers, but I will remember what it felt like "
    "to find out I was wrong, and to be glad about it, because being "
    "wrong precisely is worth more than being right vaguely.",

    "So no, we did not invent anything the world was waiting for. Others "
    "got there first, with bigger machines and bigger teams. But we "
    "walked the whole road ourselves, on a quiet laptop, step by "
    "measured step, and we wrote down every stone we tripped on. We made "
    "it. That was always the main part. And tomorrow morning, inside "
    "that same early quiet, there will be something new to make.",
]


def main():
    jobs, meta = [], []
    for i, text in enumerate(PARAS, 1):
        wav = OUT / f"narrd_{i}.wav"
        if not wav.exists():
            jobs.append({"prompt": PROMPT, "text": text,
                         "vector": [0.0] * 8, "out": str(wav)})
        meta.append((i, str(wav)))
    if jobs:
        jf = OUT / "_jobsnd.json"
        jf.write_text(json.dumps(jobs))
        print(f"synthesizing {len(jobs)} narration paragraphs ...",
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

    sil = OUT / "_sil30.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=22050:cl=mono", "-t", "0.3", str(sil)],
                   check=True)
    lines = []
    for _, w in meta:
        lines.append(f"file '{w}'")
        lines.append(f"file '{sil}'")
    concat = OUT / "_concatnd.txt"
    concat.write_text("\n".join(lines[:-1]))
    full = OUT / "narration_2min_dry.wav"
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "concat", "-safe",
                    "0", "-i", str(concat), "-ar", "22050", str(full)],
                   check=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(full)], capture_output=True,
        text=True).stdout)
    print(f"single file: {full}  ({dur/60:.2f} min)", flush=True)

    res = judge([w for _, w in meta])
    rows = []
    n = sum(1 for _ in open(LEDGER)) - 1
    for (i, w), m in zip(meta, res):
        n += 1
        rows.append([n, "indextts2-user-narration-dry", "narration",
                     f"raw_emo_vector=[0]*8;para{i};calm_register;spk=user",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     "", ""])
        print(f"  para {i} -> {m['judge_family']}"
              f"@{m['judge_confidence']:.0%}  V={m['V']:+.2f} "
              f"A={m['A']:.2f}")
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"ledger total {n}")
    print(f"PLAY IT:  afplay {full}")
    print("NARRATION_DONE")


if __name__ == "__main__":
    main()
