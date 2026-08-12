"""P5A — DISGUST OPTIMIZATION: the last unexplored bank, with a structural twist.

Pre-registered protocol, with one forced deviation from joy/sad/fear:
**the frozen judge CANNOT emit "disgust"** — its family list is hard-coded to
[anger, fear, joy, neutral, sadness, surprise] (scripts/retrieval_namer.py
FAMILIES). Zero disgust verdicts in 911 ledger rows is structural, not
acoustic. The judge stays frozen (the law), so this run scores on the two
frozen instruments that DO cover disgust:

  * Score per clip: 2·P_wavlm(disgust) + max(0, 0.5 − dVAD) against the MSP
    disgust centroid (−0.34, 0.59, +0.24) — derived from 1,325 Train labels
    with the established formula. P_wavlm from the WavLM adaptor's
    distribution (frozen, has a disgust class). The 6-way judge verdict is
    RECORDED for the ledger but cannot score — documented limitation.
  * "Hit" = WavLM top-1 == disgust.
  * TRAIN: s11 (reproach exclaim) + d01 (new, disgust-affine). 2 draws.
  * Optimizer/hull/budget identical to the other 5A runs. Seed: the native
    disgusted bank + a pinch of angry (its dimensional kin):
    [0, 0.15, 0, 0, 0.6, 0, 0, 0].
  * HELD-OUT one-shot: two NEVER-SEEN disgust sentences + s16 + flat control
    s03, 2 draws each, no iteration.
  * Every clip a ledger row (indextts2-p5a-disgust). Judge frozen. Ever.

Run:  .venv_tts/bin/python tts_steering/p5a_disgust_opt.py
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
OUT = HERE / "out/p5a_disgust"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
DIS_C = (-0.34, 0.59, +0.24)

TRAIN = {"s11": "I can't believe you did that again!",
         "d01": "That smell coming from the fridge is absolutely foul."}
HELDOUT = {"d02": "I am not touching that, it looks disgusting.",
           "d03": "How can you even eat something like this?",
           "s16": "Stop it right now.",
           "s03": "The report is on the desk in the main office."}  # control

MU, LAM, GENS = 3, 8, 10
INIT_MEAN = [0.0, 0.15, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0]
INIT_SIGMA = [0.12] * 8
RNG = random.Random(20260814)


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
    d = ((m["V"] - DIS_C[0]) ** 2 + (m["A"] - DIS_C[1]) ** 2
         + (m["D"] - DIS_C[2]) ** 2) ** 0.5
    s = 2.0 * m["pdis"] + max(0.0, 0.5 - d)
    return s, d


def main():
    results = (json.loads((OUT / "results.json").read_text())
               if (OUT / "results.json").exists() else {})
    ledgered = set(json.loads((OUT / "ledgered.json").read_text())) \
        if (OUT / "ledgered.json").exists() else set()

    def evaluate(tag, cands, sents_draws):
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
                dist = m.get("wavlm_distribution", {})
                results[wav] = {"V": m["V"], "A": m["A"], "D": m["D"],
                                "family": m["judge_family"],
                                "conf": m["judge_confidence"],
                                "pdis": float(dist.get("disgust", 0.0)),
                                "wavlm_top": m.get("wavlm_emotion", "")}
            (OUT / "results.json").write_text(json.dumps(results))
        rows, n = [], sum(1 for _ in open(LEDGER)) - 1
        scores = [0.0] * len(cands)
        for ci, c, sid, dr, wav in meta:
            m = results[wav]
            s, d = score_clip(m)
            scores[ci] += s / len(sents_draws)
            if wav not in ledgered:
                n += 1
                hit = m["wavlm_top"] == "disgust"
                rows.append([n, "indextts2-p5a-disgust", "disgust",
                             f"raw_emo_vector={c};sent={sid};draw{dr};{tag};"
                             f"pdis={m['pdis']:.2f};wavlm={m['wavlm_top']}",
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
            cands[0] = clamp_hull(list(INIT_MEAN))  # anchor the disgusted bank
        scores = evaluate(f"g{g:02d}", cands,
                          [("s11", 0), ("d01", 1)])
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])
        gen_best = ranked[0]
        improved = gen_best[0] > best_score
        if improved:
            best_score, best_ever = gen_best[0], gen_best[1]
        elite = [c for _, c in ranked[:MU]]
        mean = [sum(e[i] for e in elite) / MU for i in range(8)]
        sigma = [min(max(s * (1.2 if improved else 0.85), 0.03), 0.3)
                 for s in sigma]
        dis_hits = sum(
            1 for ci in range(LAM) for sid, dr in [("s11", 0), ("d01", 1)]
            if results.get(str(OUT / f"g{g:02d}_c{ci}_{sid}_d{dr}.wav"),
                           {}).get("wavlm_top") == "disgust")
        print(f"gen {g:2d}: wavlm-disgust top1 {dis_hits}/16  "
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
            hit = m["wavlm_top"] == "disgust"
            hits += hit and sid != "s03"
            print(f"  {sid} draw{dr}: wavlm={m['wavlm_top']} "
                  f"pdis={m['pdis']:.2f} judge6={m['family']}"
                  f"@{m['conf']:.0%} V={m['V']:+.2f} "
                  f"{'DISGUST ✓' if hit else ''}")
    print(f"\nheld-out wavlm-disgust top1 (excl. control): {hits}/6")
    json.dump({"best_vec": best_ever, "best_score": best_score},
              open(OUT / "best.json", "w"))
    print("P5A_DISGUST_DONE")


if __name__ == "__main__":
    main()
