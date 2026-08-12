"""P5A HUMAN BLIND GATE — long clips, same neutral text, sealed key.

Design (fixes the VOID'd P4.6 session's weaknesses):
  * ONE fixed emotionally-neutral paragraph (~15s spoken) for EVERY clip —
    the voice alone must carry the emotion; no text leakage to the listener.
  * 8 clips: the six Ekman-6 best vectors (P4.6 anger/surprise + P5A
    joy/sadness/fear/disgust) + a zero-vector neutral + ONE duplicate
    (joy, second draw) as a consistency control.
  * Long-form: multi-sentence, per Gate-4 rule 5 (no 2-second fragments).
  * Shuffled with a fixed seed; key sealed to out/blind_p5a_key.json —
    DO NOT open the key before answering. Machine verdicts recorded in the
    key file for later comparison, never shown before the human answers.
  * Every clip a ledger row (indextts2-p5a-blind). Judge frozen.

Run:  .venv_tts/bin/python tts_steering/blind_pack_p5a.py
Then listen:  afplay tts_steering/out/blind_pack_p5a/clip_01.wav  (etc.)
"""

import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "out/blind_pack_p5a"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")

TEXT = ("The train arrives at the station every morning at eight. "
        "People step onto the platform and look for their seats. "
        "Outside the window, the fields go past one after another.")

BEST = {
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "joy":      [0.409, -0.266, 0.022, 0.131, 0.058, -0.19, -0.23, 0.003],
    "sadness":  [0.007, 0.063, 0.005, 0.052, 0.079, 0.784, -0.095, 0.206],
    "fear":     [0.115, -0.286, 0.319, 0.296, 0.069, 0.165, 0.071, -0.278],
    "disgust":  [0.237, -0.146, -0.442, -0.048, 0.36, 0.041, -0.258, 0.057],
    "neutral":  [0.0] * 8,
}

ITEMS = [(emo, vec, 0) for emo, vec in BEST.items()] + \
        [("joy", BEST["joy"], 1)]  # duplicate draw = consistency control


def main():
    jobs, meta = [], []
    for emo, vec, dr in ITEMS:
        wav = OUT / f"src_{emo}_d{dr}.wav"
        if not wav.exists():
            jobs.append({"prompt": P_NEU, "text": TEXT, "vector": vec,
                         "out": str(wav)})
        meta.append((emo, vec, dr, str(wav)))
    if jobs:
        jf = OUT / "_jobs.json"
        jf.write_text(json.dumps(jobs))
        print(f"synthesizing {len(jobs)} long clips ...", flush=True)
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

    print("machine-judging (recorded to sealed key, not shown) ...",
          flush=True)
    verdicts = judge([w for _, _, _, w in meta])

    order = list(range(len(meta)))
    random.Random(5814).shuffle(order)
    key, rows = [], []
    n = sum(1 for _ in open(LEDGER)) - 1
    for pos, idx in enumerate(order, 1):
        emo, vec, dr, src = meta[idx]
        m = verdicts[idx]
        dst = OUT / f"clip_{pos:02d}.wav"
        dst.write_bytes(Path(src).read_bytes())
        key.append({"clip": f"clip_{pos:02d}.wav", "true_emotion": emo,
                    "draw": dr, "vector": vec,
                    "machine_judge": m["judge_family"],
                    "machine_conf": round(m["judge_confidence"], 2),
                    "V": round(m["V"], 3), "A": round(m["A"], 3)})
        n += 1
        rows.append([n, "indextts2-p5a-blind", emo,
                     f"raw_emo_vector={vec};longclip;draw{dr}",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     "", int(m["judge_family"] == emo)])
    (OUT / "../blind_p5a_key.json").resolve().write_text(
        json.dumps({"seed": 5814, "text": TEXT, "key": key}, indent=2))
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n{len(meta)} clips ready in {OUT}/clip_01..{len(meta):02d}.wav")
    print("Key sealed to out/blind_p5a_key.json — do not open before "
          "answering.")
    print("ledger +%d rows (total %d)" % (len(rows), n))
    print("BLIND_PACK_DONE")


if __name__ == "__main__":
    main()
