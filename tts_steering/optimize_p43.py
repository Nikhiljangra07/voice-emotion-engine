"""P4.3 — THE OPTIMIZER: closed-loop emotion-vector search, no human between iterations.

The machine reads its own history, proposes its next vectors, synthesizes, measures,
and converges — or honestly fails. Design laws:

  * STEER on WavLM V/A/D distance to the target centroid (the steering signal).
  * Declare a HIT only when the INDEPENDENT judge (frozen emotion2vec, different
    vector space) names the target family. Steering and judging never share a model.
  * DETERMINISTIC proposals — rule-based, no randomness; every move explainable.
  * Every synthesized clip becomes a ledger row. Misses are kept.
  * Budget-capped: MAX_ROUNDS rounds, <=4 candidates per emotion per round.

Run:  .venv_tts/bin/python tts_steering/optimize_p43.py
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402  (subprocess-only bridge, no engine imports)

ROOT = HERE.parent
OUT = HERE / "out/p43"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
TEXT = "The table is in the room, and the door is closed."
PROMPT = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")  # neutral timbre

DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
CENTROIDS = {  # parent WRITEUP §5.4, fit on 137k MSP clips
    "sadness": (-0.28, 0.39, -0.07),
    "joy":     (+0.30, 0.58, +0.19),
    "anger":   (-0.42, 0.70, +0.41),
}
MAX_ROUNDS = 3
CALM = DIMS.index("calm")


def vec(**kw) -> list[float]:
    return [round(float(kw.get(k, 0.0)), 2) for k in DIMS]


def clamp(v: list[float]) -> list[float]:
    v = [min(max(x, 0.0), 1.4) for x in v]
    s = sum(v)
    if s > 1.5:                       # API constraint: sum <= 1.5
        v = [round(x * 1.5 / s, 2) for x in v]
    return [round(x, 2) for x in v]


# Round-1 seeds: warm starts from the P4.2 ledger (best known + informed guesses).
SEEDS = {
    "anger":   [vec(angry=0.7),                       # 0.8 overshot arousal slightly
                vec(angry=0.8, calm=0.2)],            # damp arousal, keep anger
    "sadness": [vec(melancholic=1.0),                 # best knob, push harder
                vec(melancholic=1.2),
                vec(melancholic=0.8, calm=0.3),       # deepen low-arousal
                vec(melancholic=0.7, sad=0.4)],       # blend the weak knob in
    "joy":     [vec(happy=0.4),                       # LESS shout, maybe more joy
                vec(happy=0.4, calm=0.4),
                vec(surprised=0.5, happy=0.3),        # judge read happy as surprise —
                vec(happy=0.3, surprised=0.3, calm=0.3)],  # meet it halfway
}


def propose(target: str, best_vec: list[float], best_m: dict,
            tried: set) -> list[list[float]]:
    """Deterministic neighborhood of the current best, guided by the PAD error."""
    c = CENTROIDS[target]
    cands: list[list[float]] = []
    dom = max(range(8), key=lambda i: best_vec[i])
    for scale in (1.25, 0.75):                      # push/pull the dominant knob
        w = list(best_vec); w[dom] = round(w[dom] * scale, 2)
        cands.append(clamp(w))
    a_err = best_m["A"] - c[1]
    if a_err > 0.08:                                 # too aroused -> add calm
        w = list(best_vec); w[CALM] = round(min(w[CALM] + 0.3, 0.6), 2)
        cands.append(clamp(w))
    elif a_err < -0.08 and best_vec[CALM] > 0:       # too flat -> remove calm
        w = list(best_vec); w[CALM] = round(max(w[CALM] - 0.3, 0.0), 2)
        cands.append(clamp(w))
    fresh = []
    for w in cands:
        key = tuple(w)
        if key not in tried:
            tried.add(key)
            fresh.append(w)
    return fresh[:4]


def synthesize(jobs: list[dict]) -> None:
    jf = OUT / "_jobs.json"
    jf.write_text(json.dumps(jobs))
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"), str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=3600)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth worker failed:\n{proc.stderr[-600:]}")


def dist(m: dict, target: str) -> float:
    c = CENTROIDS[target]
    return math.sqrt((m["V"] - c[0]) ** 2 + (m["A"] - c[1]) ** 2
                     + (m["D"] - c[2]) ** 2)


def main() -> None:
    state = {t: {"best": None, "hit": None, "tried": set()} for t in CENTROIDS}
    ledger_rows = []
    n_prev = sum(1 for _ in open(LEDGER)) - 1 if LEDGER.exists() else 0
    it = n_prev

    for rnd in range(1, MAX_ROUNDS + 1):
        jobs, meta = [], []
        for tgt, st in state.items():
            if st["hit"] and rnd > 1:      # judge already satisfied -> stop spending
                continue
            if rnd == 1:
                cands = []
                for w in SEEDS[tgt]:
                    st["tried"].add(tuple(w))
                    cands.append(w)
            else:
                cands = propose(tgt, st["best"]["vector"], st["best"], st["tried"])
            for i, w in enumerate(cands):
                name = f"r{rnd}_{tgt}_{i}"
                jobs.append({"prompt": PROMPT, "text": TEXT, "vector": w,
                             "out": str(OUT / f"{name}.wav")})
                meta.append({"name": name, "target": tgt, "vector": w})
        if not jobs:
            break
        print(f"\n=== ROUND {rnd}: {len(jobs)} candidates ===", flush=True)
        synthesize(jobs)
        results = judge([j["out"] for j in jobs])

        for m, r in zip(meta, results):
            d = dist(r, m["target"])
            hit = r["judge_family"] == m["target"]
            it += 1
            rec = {"vector": m["vector"], "V": r["V"], "A": r["A"], "D": r["D"],
                   "dist": d, "family": r["judge_family"],
                   "conf": r["judge_confidence"], "name": m["name"]}
            st = state[m["target"]]
            if st["best"] is None or d < st["best"]["dist"]:
                st["best"] = rec
            if hit and (st["hit"] is None or d < st["hit"]["dist"]):
                st["hit"] = rec
            ledger_rows.append([it, "indextts2-p43", m["target"],
                                f"emo_vector={m['vector']};spk=ravdess_neutral_A01",
                                round(r["V"], 3), round(r["A"], 3), round(r["D"], 3),
                                r["judge_family"], round(r["judge_confidence"], 2),
                                round(d, 3), int(hit)])
            mark = "HIT" if hit else "    "
            print(f"  {m['name']:22s} {str(m['vector']):46s} d={d:.3f} "
                  f"judge={r['judge_family']}@{r['judge_confidence']:.0%} {mark}",
                  flush=True)

    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(ledger_rows)

    print("\n" + "=" * 74)
    summary = {}
    for tgt, st in state.items():
        b, h = st["best"], st["hit"]
        summary[tgt] = {"best_dist": round(b["dist"], 3), "best_vector": b["vector"],
                        "converged": h is not None,
                        "hit": None if h is None else
                        {"vector": h["vector"], "dist": round(h["dist"], 3),
                         "conf": h["conf"], "clip": h["name"]}}
        status = (f"CONVERGED  {h['name']}  d={h['dist']:.3f}  conf={h['conf']:.0%}"
                  if h else f"NOT CONVERGED  best d={b['dist']:.3f} ({b['family']})")
        print(f"{tgt:9s} {status}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nledger rows added: {len(ledger_rows)} (total {it}) -> {LEDGER}")
    print("OPTIMIZER_DONE")


if __name__ == "__main__":
    main()
