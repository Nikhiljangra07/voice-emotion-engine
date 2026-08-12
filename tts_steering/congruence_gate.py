"""CONGRUENCE GATE — testing the carrier/resonance law of the emotion space.

HYPOTHESIS (from the P5A held-out maps + the human blind gate):
  * CARRIER emotions (anger, surprise, fear): the voice broadcasts them over
    any words. Congruent text should add LITTLE.
  * RESONANCE emotions (joy, sadness, disgust): the voice can only tune a
    string the text must provide. Congruent text should add A LOT.

DESIGN (2 text conditions x 6 emotions x 2 draws, long-form):
  * FLAT arm: the emotionally-neutral train paragraph (draw 0 reused from
    the blind pack; draw 1 synthesized fresh).
  * CONGRUENT arm: one emotion-matched paragraph per emotion (below),
    2 draws each. Same vectors as the blind pack (the six best).
  * Mechanism note: the judges are ACOUSTIC (emotion2vec kNN, WavLM). A
    congruent-text gain is text->prosody: the GPT acts the words. That IS
    the law being tested — text is a steering input, not a semantic leak.

PRE-REGISTERED PREDICTIONS (before any synthesis):
  joy      flat 0/2 -> congruent >=1/2   (big gain)      [resonance]
  sadness  flat 0/2 -> congruent >=1/2   (gain, or V on-centroid) [resonance]
  disgust  flat low -> congruent gain on WavLM/v2 instruments    [resonance]
  anger    flat 2/2 -> congruent 2/2     (no change)     [carrier]
  surprise flat 2/2 -> congruent 2/2     (no change)     [carrier]
  fear     flat ?   -> congruent >= flat (pack long-clip missed; P5A short
           clips carried — long-form fear is the open cell)   [carrier?]

LAW GAIN METRIC: interaction = mean(resonance gain) - mean(carrier gain).
Law is REAL if interaction is clearly positive with carriers ~unchanged.

Every clip a ledger row (indextts2-congruence). Judge frozen (v1); WavLM
distribution recorded; disgust additionally scored by judge-v2 offline.

Run:  .venv_tts/bin/python tts_steering/congruence_gate.py
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
OUT = HERE / "out/congruence"
OUT.mkdir(parents=True, exist_ok=True)
PACK = HERE / "out/blind_pack_p5a"
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")

FLAT = ("The train arrives at the station every morning at eight. "
        "People step onto the platform and look for their seats. "
        "Outside the window, the fields go past one after another.")

CONGRUENT = {
    "joy": ("You will not believe the news I got this morning. "
            "She said yes, and everyone is coming to celebrate with us "
            "this weekend. I could not stop smiling the whole way home."),
    "sadness": ("The house has been empty since they moved away. "
                "I keep finding small things they left behind in the "
                "drawers. Nothing has felt the same this whole year."),
    "disgust": ("Something in the fridge has gone completely rotten. "
                "The smell spread through the whole kitchen overnight. "
                "I had to throw everything out and scrub the shelves twice."),
    "anger": ("You broke your promise again after everything we talked "
              "about. I have had enough of these excuses every single "
              "time. This ends today, right now."),
    "fear": ("I keep hearing footsteps behind us on this empty street. "
             "My hands will not stop shaking and it is getting darker. "
             "Please stay close and do not look back."),
    "surprise": ("Wait, they actually called my name from the stage? "
                 "Out of hundreds of people, they picked me? I honestly "
                 "cannot believe this is happening right now."),
}

BEST = {
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "joy":      [0.409, -0.266, 0.022, 0.131, 0.058, -0.19, -0.23, 0.003],
    "sadness":  [0.007, 0.063, 0.005, 0.052, 0.079, 0.784, -0.095, 0.206],
    "fear":     [0.115, -0.286, 0.319, 0.296, 0.069, 0.165, 0.071, -0.278],
    "disgust":  [0.237, -0.146, -0.442, -0.048, 0.36, 0.041, -0.258, 0.057],
}
CARRIER = ("anger", "surprise", "fear")
RESONANCE = ("joy", "sadness", "disgust")


def main():
    jobs, meta = [], []
    for emo, vec in BEST.items():
        # FLAT draw 0: reuse the blind-pack clip (identical vector+text)
        pack_src = PACK / f"src_{emo}_d0.wav"
        flat0 = OUT / f"flat_{emo}_d0.wav"
        if not flat0.exists() and pack_src.exists():
            flat0.write_bytes(pack_src.read_bytes())
        meta.append((emo, "flat", 0, str(flat0)))
        # FLAT draw 1: fresh
        flat1 = OUT / f"flat_{emo}_d1.wav"
        if not flat1.exists():
            jobs.append({"prompt": P_NEU, "text": FLAT, "vector": vec,
                         "out": str(flat1)})
        meta.append((emo, "flat", 1, str(flat1)))
        # CONGRUENT draws 0,1
        for dr in (0, 1):
            cw = OUT / f"cong_{emo}_d{dr}.wav"
            if not cw.exists():
                jobs.append({"prompt": P_NEU, "text": CONGRUENT[emo],
                             "vector": vec, "out": str(cw)})
            meta.append((emo, "cong", dr, str(cw)))

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

    print("judging 24 clips ...", flush=True)
    verdicts = judge([w for _, _, _, w in meta])
    res, rows = {}, []
    n = sum(1 for _ in open(LEDGER)) - 1
    for (emo, arm, dr, wav), m in zip(meta, verdicts):
        pdis = float(m.get("wavlm_distribution", {}).get("disgust", 0.0))
        hit = m["judge_family"] == emo
        whit = m.get("wavlm_emotion", "") == emo
        res[(emo, arm, dr)] = (m["judge_family"], m["judge_confidence"],
                               m["V"], m["A"], hit, whit, pdis)
        n += 1
        rows.append([n, "indextts2-congruence", emo,
                     f"raw_emo_vector={BEST[emo]};arm={arm};draw{dr};"
                     f"wavlm={m.get('wavlm_emotion','')};pdis={pdis:.2f}",
                     round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                     m["judge_family"], round(m["judge_confidence"], 2),
                     "", int(hit)])
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"\n{'emotion':9s} {'type':10s} {'FLAT (d0,d1)':28s} CONGRUENT (d0,d1)")
    gains = {}
    for emo in BEST:
        typ = "carrier" if emo in CARRIER else "resonance"
        cells = {}
        for arm in ("flat", "cong"):
            parts, hits = [], 0
            for dr in (0, 1):
                fam, conf, V, A, hit, whit, _ = res[(emo, arm, dr)]
                parts.append(f"{fam}@{conf:.0%}{'✓' if hit else ''}")
                hits += hit
            cells[arm] = (", ".join(parts), hits)
        gains[emo] = cells["cong"][1] - cells["flat"][1]
        print(f"{emo:9s} {typ:10s} {cells['flat'][0]:28s} {cells['cong'][0]}")
    cg = sum(gains[e] for e in CARRIER) / len(CARRIER)
    rg = sum(gains[e] for e in RESONANCE) / len(RESONANCE)
    print(f"\nmean gain (hits/2): carriers {cg:+.2f}   resonance {rg:+.2f}")
    print(f"LAW INTERACTION (resonance - carrier): {rg - cg:+.2f}")
    json.dump({str(k): v for k, v in res.items()},
              open(OUT / "results.json", "w"))
    print(f"ledger +{len(rows)} rows (total {n})")
    print("CONGRUENCE_DONE")


if __name__ == "__main__":
    main()
