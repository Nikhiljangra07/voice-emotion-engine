"""P5A — SADNESS OPTIMIZATION: evolution search in the mouth's native emovec space.

Second Phase-5 run, protocol identical to p5a_joy_opt.py (pre-registered):
  * TRAIN sentences: s08 + s09 (the two proven somber carriers from P4.6 —
    s08 also holds the all-time sadness record, sad=0.4 -> d=0.085).
    2 draws per candidate (one per sentence).
  * Score per clip: 2·(judge==sadness)·confidence + max(0, 0.5 − dVAD)
    against the MSP sadness centroid (−0.28, 0.39, −0.07).
  * Optimizer: seeded (mu=3, lambda=8) evolution strategy, 10 generations,
    diagonal sigma 1.2x/0.85x adaptation, floor 0.03. Deterministic ->
    resumable via clip cache. Judge retry x3 with 60s cooldown (joy-run fix).
  * Hull (identical to joy run): coef in [-0.6, +1.0], sum|c| <= 1.6,
    sum c in [-0.2, +1.1]. Init mean = proven mel-route region at the hull
    edge: [0, 0, 0.15, 0, 0, 0.75, 0, 0.2].
  * HELD-OUT one-shot (no iteration): best vector on s10 (somber, never got a
    sadness verdict in P4.6 — a real test) + two NEVER-SEEN somber sentences
    + one WARM control (s07). The control probes the P4.6 smearing risk:
    mel1.0+calm0.3 painted sadness onto warm sentences; a good vector must
    NOT turn "we'll see them again this weekend" sad.
  * Judge frozen. Every clip a ledger row (indextts2-p5a-sad). Cap:
    160 optimization clips + 8 held-out.

Run:  .venv_tts/bin/python tts_steering/p5a_sad_opt.py
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
OUT = HERE / "out/p5a_sad"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
SAD_C = (-0.28, 0.39, -0.07)

TRAIN = {"s08": "She never came back after that day.",
         "s09": "The house has been empty for a long time now."}
HELDOUT = {"s10": "He put the old photograph back in the drawer.",
           "h03": "The old clock in the hallway stopped ticking years ago.",
           "h04": "Nothing was the same after they moved away.",
           "s07": "We are going to see them again this weekend."}  # control

MU, LAM, GENS = 3, 8, 10
INIT_MEAN = [0.0, 0.0, 0.15, 0.0, 0.0, 0.75, 0.0, 0.2]
INIT_SIGMA = [0.12] * 8
RNG = random.Random(20260812)


def clamp_hull(c):
    c = [min(max(x, -0.6), 1.0) for x in c]
    l1 = sum(abs(x) for x in c)
    if l1 > 1.6:
        c = [x * 1.6 / l1 for x in c]
    s = sum(c)
    if s > 1.1:
        c = [x - (s - 1.1) / 8 for x in c]
    elif s < -0.2:
        c = [x + (-0.2 - s) / 8 for x in c]
    return [round(x, 3) for x in c]


def synth(jobs):
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if not todo:
        return
    jf = OUT / "_jobs.json"
    jf.write_text(json.dumps(todo))
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
         str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                              "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=7200)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")


def score_clip(m):
    d = ((m["V"] - SAD_C[0]) ** 2 + (m["A"] - SAD_C[1]) ** 2
         + (m["D"] - SAD_C[2]) ** 2) ** 0.5
    s = (2.0 * m["conf"] if m["family"] == "sadness" else 0.0) \
        + max(0.0, 0.5 - d)
    return s, d


def main():
    results = (json.loads((OUT / "results.json").read_text())
               if (OUT / "results.json").exists() else {})
    ledgered = set(json.loads((OUT / "ledgered.json").read_text())) \
        if (OUT / "ledgered.json").exists() else set()

    def evaluate(tag, cands, sents_draws):
        """cands: list of coef lists. sents_draws: [(sid, draw_idx), ...]"""
        jobs, meta = [], []
        for ci, c in enumerate(cands):
            for sid, dr in sents_draws:
                wav = OUT / f"{tag}_c{ci}_{sid}_d{dr}.wav"
                text = TRAIN.get(sid) or HELDOUT[sid]
                jobs.append({"prompt": P_NEU, "text": text, "vector": c,
                             "out": str(wav)})
                meta.append((ci, c, sid, dr, str(wav)))
        synth(jobs)
        new = [w for _, _, _, _, w in meta if w not in results]
        if new:
            for attempt in range(3):
                try:
                    verdicts = judge(new)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  judge failed ({e}); retry in 60s", flush=True)
                    time.sleep(60)
            for wav, m in zip(new, verdicts):
                results[wav] = {"V": m["V"], "A": m["A"], "D": m["D"],
                                "family": m["judge_family"],
                                "conf": m["judge_confidence"]}
            (OUT / "results.json").write_text(json.dumps(results))
        rows, n = [], sum(1 for _ in open(LEDGER)) - 1
        scores = [0.0] * len(cands)
        for ci, c, sid, dr, wav in meta:
            m = results[wav]
            s, d = score_clip(m)
            scores[ci] += s / len(sents_draws)
            if wav not in ledgered:
                n += 1
                hit = m["family"] == "sadness"
                rows.append([n, "indextts2-p5a-sad", "sadness",
                             f"raw_emo_vector={c};sent={sid};draw{dr};{tag}",
                             round(m["V"], 3), round(m["A"], 3),
                             round(m["D"], 3), m["family"],
                             round(m["conf"], 2), round(d, 3), int(hit)])
                ledgered.add(wav)
        if rows:
            with open(LEDGER, "a", newline="") as f:
                csv.writer(f).writerows(rows)
            (OUT / "ledgered.json").write_text(json.dumps(sorted(ledgered)))
        return scores

    mean, sigma = list(INIT_MEAN), list(INIT_SIGMA)
    best_ever, best_score = None, -1.0
    t0 = time.time()
    for g in range(1, GENS + 1):
        cands = [clamp_hull([mean[i] + RNG.gauss(0, sigma[i])
                             for i in range(8)]) for _ in range(LAM)]
        if g == 1:
            cands[0] = clamp_hull(list(INIT_MEAN))  # anchor the mel route
        scores = evaluate(f"g{g:02d}", cands,
                          [("s08", 0), ("s09", 1)])
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])
        gen_best = ranked[0]
        improved = gen_best[0] > best_score
        if improved:
            best_score, best_ever = gen_best[0], gen_best[1]
        elite = [c for _, c in ranked[:MU]]
        mean = [sum(e[i] for e in elite) / MU for i in range(8)]
        sigma = [min(max(s * (1.2 if improved else 0.85), 0.03), 0.3)
                 for s in sigma]
        sad_hits = sum(
            1 for ci in range(LAM) for sid, dr in [("s08", 0), ("s09", 1)]
            if results.get(str(OUT / f"g{g:02d}_c{ci}_{sid}_d{dr}.wav"),
                           {}).get("family") == "sadness")
        print(f"gen {g:2d}: sadness verdicts {sad_hits}/16  "
              f"best={gen_best[0]:.3f} "
              f"(ever {best_score:.3f}) vec={gen_best[1]} "
              f"sigma~{sum(sigma)/8:.3f} elapsed {(time.time()-t0)/60:.0f}m",
              flush=True)

    print(f"\nOPTIMIZATION DONE. best={best_score:.3f} vec={best_ever}")

    print("\nHELD-OUT ONE-SHOT (pre-registered, no iteration):", flush=True)
    ho = [(sid, dr) for sid in HELDOUT for dr in (0, 1)]
    evaluate("heldout", [best_ever], ho)
    hits = 0
    for sid in HELDOUT:
        for dr in (0, 1):
            m = results[str(OUT / f"heldout_c0_{sid}_d{dr}.wav")]
            hit = m["family"] == "sadness"
            hits += hit and sid != "s07"
            print(f"  {sid} draw{dr}: {m['family']}@{m['conf']:.0%} "
                  f"V={m['V']:+.2f} {'SAD ✓' if hit else ''}")
    print(f"\nheld-out sadness verdicts (excl. control): {hits}/6")
    json.dump({"best_vec": best_ever, "best_score": best_score},
              open(OUT / "best.json", "w"))
    print("P5A_SAD_DONE")


if __name__ == "__main__":
    main()
