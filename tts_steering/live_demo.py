"""LIVE-VOICE DEMO — the closed loop, end to end, on a real voice.

Ear hears you -> reads V/A/D + emotion family -> mouth replies with a
judge-proven emovec vector -> the reply is judged too (loop closure printed).

The reply is synthesized in YOUR voice by default (IndexTTS-2 zero-shot
cloning from the input recording itself — local, consented, your own audio).
Pass --speaker to use a different voice prompt.

Usage:
  # from an existing recording
  .venv_tts/bin/python tts_steering/live_demo.py --input my_clip.wav

  # record 6 seconds from the mic (macOS; grant terminal mic permission)
  .venv_tts/bin/python tts_steering/live_demo.py --record 6

  # options
  --mode mirror|respond   mirror = reply in the emotion it heard (default)
                          respond = empathetic counter-emotion
  --text "..."            override the reply text
  --speaker voice.wav     reply in this voice instead of yours

Every judged clip is a ledger row (system=indextts2-live-demo). Judge frozen.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402

OUT = HERE / "out/live_demo"
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"

# Judge-proven vectors (P4.6 map + P5A optimization, see STEERING_LOG.md)
PROVEN = {
    "joy":      [0.409, -0.266, 0.022, 0.131, 0.058, -0.19, -0.23, 0.003],
    "sadness":  [0.007, 0.063, 0.005, 0.052, 0.079, 0.784, -0.095, 0.206],
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "neutral":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# What the mouth says, per reply emotion (overridable with --text)
REPLY_TEXT = {
    "joy":      "That is wonderful to hear, I am really glad you told me.",
    "sadness":  "I hear you. That sounds heavy, and I am here with you.",
    "anger":    "Something about this clearly is not okay, and it matters.",
    "surprise": "Wait, really? I did not see that coming at all!",
    "neutral":  "Alright, I am listening. Tell me more about it.",
}

# respond mode: heard emotion -> empathetic reply emotion
RESPOND = {"joy": "joy", "surprise": "surprise", "neutral": "neutral",
           "sadness": "sadness", "anger": "neutral", "fear": "neutral",
           "disgust": "neutral"}


def record(seconds):
    wav = OUT / f"live_input_{int(time.time())}.wav"
    print(f"recording {seconds}s from mic ... speak now")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0",
         "-t", str(seconds), "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not wav.exists():
        sys.exit(f"mic capture failed (mic permission?):\n{proc.stderr[-400:]}")
    return str(wav)


def synth(text, vector, speaker, out_wav):
    jf = OUT / "_jobs.json"
    jf.write_text(json.dumps([{"prompt": speaker, "text": text,
                               "vector": vector, "out": str(out_wav)}]))
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
         str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                              "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=1800)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")


def ledger_row(system, target, control, m, d="", hit=""):
    n = sum(1 for _ in open(LEDGER)) - 1 + 1
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerow(
            [n, system, target, control, round(m["V"], 3), round(m["A"], 3),
             round(m["D"], 3), m["judge_family"],
             round(m["judge_confidence"], 2), d, hit])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--record", type=int)
    ap.add_argument("--mode", choices=["mirror", "respond"], default="mirror")
    ap.add_argument("--text")
    ap.add_argument("--speaker")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.input and not args.record:
        sys.exit("need --input file.wav or --record SECONDS")
    heard_wav = args.input or record(args.record)

    print("\n=== EAR ===")
    m = judge([heard_wav])[0]
    fam, conf = m["judge_family"], m["judge_confidence"]
    print(f"heard: {fam}@{conf:.0%}  "
          f"V={m['V']:+.2f} A={m['A']:.2f} D={m['D']:+.2f}")
    ledger_row("indextts2-live-demo", "ear-input",
               f"live_input={Path(heard_wav).name}", m)

    reply_emo = fam if args.mode == "mirror" else RESPOND.get(fam, "neutral")
    if reply_emo not in PROVEN:
        print(f"(no proven vector for '{reply_emo}' yet -> neutral)")
        reply_emo = "neutral"
    text = args.text or REPLY_TEXT[reply_emo]
    speaker = args.speaker or heard_wav
    vector = PROVEN[reply_emo]

    print(f"\n=== MOUTH ===  replying with {reply_emo} ({args.mode})")
    print(f'text: "{text}"')
    reply_wav = OUT / f"reply_{reply_emo}_{int(time.time())}.wav"
    t0 = time.time()
    synth(text, vector, speaker, reply_wav)
    print(f"synthesized in {time.time()-t0:.0f}s -> {reply_wav}")

    print("\n=== LOOP CLOSURE ===")
    r = judge([str(reply_wav)])[0]
    ok = r["judge_family"] == reply_emo
    print(f"reply judged: {r['judge_family']}@{r['judge_confidence']:.0%}  "
          f"V={r['V']:+.2f} A={r['A']:.2f}  "
          f"{'CONFIRMED' if ok else 'drifted'}")
    ledger_row("indextts2-live-demo", reply_emo,
               f"raw_emo_vector={vector};mode={args.mode}", r,
               hit=int(ok))
    print(f"\nheard {fam} -> replied {reply_emo} -> judge says "
          f"{r['judge_family']}. Play it:  afplay {reply_wav}")


if __name__ == "__main__":
    main()
