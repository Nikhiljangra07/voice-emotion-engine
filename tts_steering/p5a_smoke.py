"""P5A SMOKE — first signed, uncapped emovec-space clips, judged.

Discovery en route (correcting P4.12): normalize_emo_vec (bias + 0.8 cap) is
WEBUI-ONLY. The inference path takes emo_vector raw:
    emovec = Σ w_i · bank_i  +  (1 − Σw) · speaker_emovec
Negative and >1 coefficients flow straight through — the Phase 5A search
space was open all along; only OUR harness clamps ever restricted it.

Four probes on s07 ("We are going to see them again this weekend." — the
sentence where plain happy=0.35 already earned a joy verdict):
  1. parity     happy +0.35                  (must reproduce the joy verdict)
  2. thesis     happy +0.60, angry −0.30     (SUBTRACT TENSION — first
                                              negative coefficient ever)
  3. beyond     happy +1.10                  (past the webui's 0.8 cap zone)
  4. warm blend happy +0.45, calm +0.35, angry −0.15

Win condition (smoke, not a gate): clips synthesize without artifacts and the
judge still reads them sanely; thesis clip judged joy = thesis confirmed.
Ledger rows: system=indextts2-p5a-smoke.

Run:  .venv_tts/bin/python tts_steering/p5a_smoke.py
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
OUT = HERE / "out/p5a"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
TEXT = "We are going to see them again this weekend."
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
JOY_C = (+0.30, 0.58, +0.19)


def vec(**kw):
    order = ["happy", "angry", "sad", "afraid", "disgusted",
             "melancholic", "surprised", "calm"]
    return [round(float(kw.get(k, 0.0)), 2) for k in order]


PROBES = [
    ("parity_h035", vec(happy=0.35)),
    ("thesis_h06_am03", vec(happy=0.60, angry=-0.30)),
    ("beyond_h110", vec(happy=1.10)),
    ("blend_h045_c035_am015", vec(happy=0.45, calm=0.35, angry=-0.15)),
]


def main():
    jobs, meta = [], []
    for name, v in PROBES:
        wav = OUT / f"smoke_{name}.wav"
        jobs.append({"prompt": P_NEU, "text": TEXT, "vector": v,
                     "out": str(wav)})
        meta.append((name, v, str(wav)))
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if todo:
        jf = OUT / "_jobs.json"
        jf.write_text(json.dumps(todo))
        print(f"synthesizing {len(todo)} probes ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
             str(jf)],
            cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                                  "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=3600)
        if "WORKER_DONE" not in proc.stdout:
            raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    print("judging ...", flush=True)
    res = judge([w for _, _, w in meta])
    rows = []
    n = sum(1 for _ in open(LEDGER)) - 1
    for (name, v, wav), m in zip(meta, res):
        d = ((m["V"] - JOY_C[0]) ** 2 + (m["A"] - JOY_C[1]) ** 2
             + (m["D"] - JOY_C[2]) ** 2) ** 0.5
        hit = m["judge_family"] == "joy"
        n += 1
        rows.append([n, "indextts2-p5a-smoke", "joy",
                     f"raw_emo_vector={v};sent=s07;uncapped_signed",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     round(d, 3), int(hit)])
        mark = "JOY ✓" if hit else m["judge_family"]
        print(f"  {name:24s} {v} -> V={m['V']:+.2f} A={m['A']:.2f} "
              f"d={d:.3f} judge={mark}@{m['judge_confidence']:.0%}")
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"ledger +{len(rows)} rows (total {n})")
    print("P5A_SMOKE_DONE")


if __name__ == "__main__":
    main()
