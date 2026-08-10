"""P4.11 — THE JOY SPIKE: six clips at the last open door on S1.

Pre-registered (before any synthesis):
  WIN     = frozen e2v judge names "joy" on S1, any confidence. Nothing else.
  BUDGET  = 6 scored clips, two rounds of 3, deterministic round-2 rules below.
  Everything -> ledger rows (system=indextts2-joyspike), misses kept.

Evidence shaping the search (P4.7 ledger): on S1, happy=0.3 -> neutral@80%,
happy=0.4 -> fear@100%, happy=0.22 -> neutral@100%, happy=0.38 -> fear@100%,
happy+calm 0.4+0.4 -> neutral@100%. The verdict flips neutral->fear across
[0.3, 0.4] without passing joy. Round 1 probes:
  1. happy=0.35                 — the unexplored gap itself
  2. happy=0.30 + surprised=0.20 — low intensity + the transient the judge
                                   rewards (joy/surprise adjacency)
  3. happy=0.40 with a HAPPY speaker prompt — prompt-emotion condition
                                   (P4.2 precedent), timbre warmth without
                                   pushing the vector

Round-2 rules (deterministic, declared now):
  verdict fear      -> subtract 0.05 happy, add calm 0.15
  verdict neutral   -> add 0.05 happy
  verdict surprise  -> halve surprised component, keep happy
  verdict joy       -> STOP (win); remaining budget refines confidence upward
                       by +-0.05 happy around the winner

Run:  .venv_tts/bin/python tts_steering/joy_spike.py
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
OUT = HERE / "out/joy_spike"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"

TEXT = "The table is in the room, and the door is closed."
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
P_HAP = str(ROOT / "data/ravdess/Actor_01/03-01-03-02-01-01-01.wav")
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
CENTROID = (+0.30, 0.58, +0.19)


def vec(**kw):
    return [round(float(kw.get(k, 0.0)), 2) for k in DIMS]


def synth(jobs):
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if todo:
        jf = OUT / "_jobs.json"
        jf.write_text(json.dumps(todo))
        t0 = time.time()
        proc = subprocess.run(
            [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
             str(jf)],
            cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                                  "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=3600)
        if "WORKER_DONE" not in proc.stdout:
            raise RuntimeError(f"synth failed:\n{proc.stderr[-400:]}")
        print(f"  synthesized {len(todo)} in {time.time()-t0:.0f}s", flush=True)


def append_rows(rows):
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)


def dist(m):
    return ((m["V"] - CENTROID[0]) ** 2 + (m["A"] - CENTROID[1]) ** 2
            + (m["D"] - CENTROID[2]) ** 2) ** 0.5


def run_round(cands, rnd):
    jobs, meta = [], []
    for i, (v, prompt, note) in enumerate(cands):
        wav = OUT / f"joy_r{rnd}_{i}.wav"
        jobs.append({"prompt": prompt, "text": TEXT, "vector": v,
                     "out": str(wav)})
        meta.append((v, prompt, note, str(wav)))
    synth(jobs)
    print(f"  judging round {rnd} ...", flush=True)
    res = judge([w for _, _, _, w in meta])
    rows, out = [], []
    n = sum(1 for _ in open(LEDGER)) - 1
    for (v, prompt, note, wav), m in zip(meta, res):
        d = dist(m)
        hit = m["judge_family"] == "joy"
        n += 1
        ptag = "happy_prompt" if prompt == P_HAP else "neutral_prompt"
        rows.append([n, "indextts2-joyspike", "joy",
                     f"emo_vector={v};{ptag};{note}",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     round(d, 3), int(hit)])
        out.append((v, prompt, note, m, d, hit))
        mark = "JOY ✓✓✓" if hit else "miss"
        print(f"  {note:34s} {v} -> V={m['V']:+.2f} A={m['A']:.2f} d={d:.3f} "
              f"judge={m['judge_family']}@{m['judge_confidence']:.0%} {mark}")
    append_rows(rows)
    print(f"  ledger +{len(rows)} rows (total {n})")
    return out


def main():
    print("=" * 70 + "\nP4.11 JOY SPIKE — round 1\n" + "=" * 70)
    r1 = run_round([
        (vec(happy=0.35), P_NEU, "gap-probe"),
        (vec(happy=0.30, surprised=0.20), P_NEU, "low+transient"),
        (vec(happy=0.40), P_HAP, "happy-prompt"),
    ], 1)

    winners = [r for r in r1 if r[5]]
    print("=" * 70 + "\nround 2 (deterministic rules)\n" + "=" * 70)
    cands = []
    if winners:
        v0, p0, _, _, _, _ = winners[0]
        h = v0[0]
        cands = [(vec(happy=round(h + 0.05, 2)), p0, "confidence-up"),
                 (vec(happy=round(max(h - 0.05, 0.05), 2)), p0,
                  "confidence-down"),
                 (vec(happy=h, calm=0.15), p0, "warm-variant")]
    else:
        for v, prompt, note, m, d, _ in r1:
            fam = m["judge_family"]
            h, s = v[0], v[6]
            if fam == "fear":
                cands.append((vec(happy=round(h - 0.05, 2), calm=0.15),
                              prompt, f"r2<-{note}:fear-cooled"))
            elif fam == "neutral":
                cands.append((vec(happy=round(h + 0.05, 2),
                                  surprised=s), prompt,
                              f"r2<-{note}:warmed"))
            elif fam == "surprise":
                cands.append((vec(happy=h, surprised=round(s / 2, 2)),
                              prompt, f"r2<-{note}:less-spike"))
            else:
                cands.append((vec(happy=round(h + 0.05, 2)), prompt,
                              f"r2<-{note}:nudge"))
        cands = cands[:3]
    r2 = run_round(cands, 2)

    hits = [r for r in r1 + r2 if r[5]]
    print("=" * 70)
    if hits:
        best = max(hits, key=lambda r: r[3]["judge_confidence"])
        print(f"JOY NAILED: {best[0]} ({best[2]}) — "
              f"judge joy@{best[3]['judge_confidence']:.0%}, d={best[4]:.3f}")
    else:
        best = min(r1 + r2, key=lambda r: r[4])
        print(f"NO JOY on S1 in 6 clips. Closest: {best[0]} ({best[2]}) "
              f"d={best[4]:.3f}, judged {best[3]['judge_family']}. "
              f"S1-joy documented as closed for this mouth.")
    print("JOY_SPIKE_DONE")


if __name__ == "__main__":
    main()
