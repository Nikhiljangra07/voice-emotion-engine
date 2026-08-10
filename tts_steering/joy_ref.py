"""P4.12b — JOY BY REFERENCE: the mouth's last untested channel, aimed at joy.

Mechanism (from the P4.12 survey): emotion cloned from REFERENCE AUDIO
(`emo_audio_prompt` + `emo_alpha`) — the same philosophy as Voxtral's
voice-as-instruction and the GST family, sitting unused in our own mouth for
180 ledger rows. Speaker prompt stays the neutral Actor_01 timbre; the
reference carries only the emotion. emo_vector unused (API: mutually
exclusive).

Pre-registered (before any synthesis):
  WIN    = frozen e2v judge names "joy" on S1, any confidence.
  BUDGET = 6 scored clips, 2 rounds of 3.
  Round 1: male acted joy @ alpha 1.0 / 0.65 · natural MELD joy @ 1.0
  Round 2 rules, declared:
    any joy -> alpha +-0.2 around winner + same ref on the S3-style check
    no joy  -> female acted joy @1.0 · best-valence round-1 ref @0.8 ·
               second natural MELD joy @1.0
  Every clip a ledger row (system=indextts2-joyref), misses kept.

Run:  .venv_tts/bin/python tts_steering/joy_ref.py
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
OUT = HERE / "out/joy_ref"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"

TEXT = "The table is in the room, and the door is closed."
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
REFS = {
    "rav_happy_male": str(ROOT / "data/ravdess/Actor_01/03-01-03-02-01-01-01.wav"),
    "rav_happy_female": str(ROOT / "data/ravdess/Actor_02/03-01-03-02-01-01-02.wav"),
    "meld_joy_1": str(ROOT / "data/meld/audio/train/dia6_utt0.flac"),
    "meld_joy_2": str(ROOT / "data/meld/audio/train/dia6_utt6.flac"),
}
CENTROID = (+0.30, 0.58, +0.19)


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
            raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")
        print(f"  synthesized {len(todo)} in {time.time()-t0:.0f}s", flush=True)


def run_round(cands, rnd):
    jobs, meta = [], []
    for i, (ref, alpha) in enumerate(cands):
        wav = OUT / f"joyref_r{rnd}_{i}_{ref}_{int(alpha*100)}.wav"
        jobs.append({"prompt": P_NEU, "text": TEXT, "emo_audio": REFS[ref],
                     "emo_alpha": alpha, "out": str(wav)})
        meta.append((ref, alpha, str(wav)))
    synth(jobs)
    print(f"  judging round {rnd} ...", flush=True)
    res = judge([w for _, _, w in meta])
    rows, out = [], []
    n = sum(1 for _ in open(LEDGER)) - 1
    for (ref, alpha, wav), m in zip(meta, res):
        d = ((m["V"] - CENTROID[0]) ** 2 + (m["A"] - CENTROID[1]) ** 2
             + (m["D"] - CENTROID[2]) ** 2) ** 0.5
        hit = m["judge_family"] == "joy"
        n += 1
        rows.append([n, "indextts2-joyref", "joy",
                     f"emo_ref={ref};alpha={alpha};spk=neutral_A01",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     round(d, 3), int(hit)])
        out.append((ref, alpha, m, d, hit))
        mark = "JOY ✓✓✓" if hit else "miss"
        print(f"  {ref:18s} a={alpha:4.2f} -> V={m['V']:+.2f} A={m['A']:.2f} "
              f"d={d:.3f} judge={m['judge_family']}@"
              f"{m['judge_confidence']:.0%} {mark}")
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  ledger +{len(rows)} rows (total {n})")
    return out


def main():
    print("=" * 70 + "\nP4.12b JOY BY REFERENCE — round 1\n" + "=" * 70)
    r1 = run_round([("rav_happy_male", 1.0), ("rav_happy_male", 0.65),
                    ("meld_joy_1", 1.0)], 1)

    winners = [r for r in r1 if r[4]]
    print("=" * 70 + "\nround 2 (declared rules)\n" + "=" * 70)
    if winners:
        ref, a, _, _, _ = winners[0]
        cands = [(ref, min(a + 0.2, 1.0)), (ref, max(a - 0.2, 0.2)),
                 ("rav_happy_female" if ref != "rav_happy_female"
                  else "meld_joy_2", 1.0)]
    else:
        best_v = max(r1, key=lambda r: r[2]["V"])
        cands = [("rav_happy_female", 1.0), (best_v[0], 0.8),
                 ("meld_joy_2", 1.0)]
    r2 = run_round(cands, 2)

    hits = [r for r in r1 + r2 if r[4]]
    print("=" * 70)
    if hits:
        best = max(hits, key=lambda r: r[2]["judge_confidence"])
        print(f"JOY BY REFERENCE: NAILED — {best[0]} alpha={best[1]} "
              f"joy@{best[2]['judge_confidence']:.0%} d={best[3]:.3f}")
    else:
        bv = max(r1 + r2, key=lambda r: r[2]["V"])
        bd = min(r1 + r2, key=lambda r: r[3])
        print(f"NO JOY via reference channel. Best V={bv[2]['V']:+.2f} "
              f"({bv[0]} a={bv[1]}), best d={bd[3]:.3f} "
              f"(judged {bd[2]['judge_family']}). S1-joy closed across ALL "
              f"channels of this mouth.")
    print("JOY_REF_DONE")


if __name__ == "__main__":
    main()
