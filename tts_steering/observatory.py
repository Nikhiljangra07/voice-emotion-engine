"""OBSERVATORY — run the Phase-4 loop unchanged, but narrate every internal step.

Nothing in the pipeline is modified: same synth worker, same frozen bridge, same
centroids, same rules. This wrapper only *watches* and explains. Clips generated
here still obey the ledger law (system=observatory-demo).

Run:  .venv_tts/bin/python tts_steering/observatory.py
"""

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import wavlm_vad, e2v_family  # noqa: E402  (the two halves of judge())

ROOT = HERE.parent
OUT = HERE / "out/observatory"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"

TEXT = "The table is in the room, and the door is closed."
PROMPT = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
CENTROIDS = {"sadness": (-0.28, 0.39, -0.07), "joy": (+0.30, 0.58, +0.19),
             "anger": (-0.42, 0.70, +0.41)}

DEMOS = [  # the two most instructive configs from the 130-row ledger
    ("anger",   [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     "the converged winner: judge-confirmed HIT at d=0.207 in P4.3"),
    ("sadness", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3],
     "the famous 'miss': best acoustics of all 5 systems (d=0.134), judge said "
     "neutral — blind human ears later ruled it SAD (the ear's blind spot)"),
]


def h(title):
    print(f"\n{'='*74}\n  {title}\n{'='*74}", flush=True)


def main():
    h("STAGE 0 — THE CAST")
    print(f"""
  MOUTH   IndexTTS-2 (vendor env, py3.11, MPS)   — speaks on command
  BRIDGE  bridge.py                              — subprocess -> CLI -> JSON,
                                                   the ONLY link to the engine
  EAR-1   fine-tuned WavLM  (.venv_diar)         — places audio in V/A/D space
                                                   -> the STEERING signal
  EAR-2   frozen emotion2vec (.venv_diar)        — names the emotion family
                                                   -> the VERDICT (HIT/miss)
  Text (constant, semantically neutral): "{TEXT}"
  Voice prompt (constant, neutral timbre): RAVDESS Actor_01 neutral
  Law reminder: steering and verdict use DIFFERENT models in DIFFERENT vector
  spaces — the anti-circularity design. And the judge is FROZEN, forever.""")

    jobs, meta = [], []
    for emo, vec, why in DEMOS:
        h(f"STAGE 1 — CONTROL VECTOR for target '{emo.upper()}'")
        print(f"\n  Why this config: {why}\n")
        print("  The mouth's 8 sliders (0-1.4 each, sum <= 1.5):")
        for d, v in zip(DIMS, vec):
            bar = "#" * int(v * 20)
            print(f"    {d:12s} {v:4.1f}  {bar}")
        wav = OUT / f"obs_{emo}.wav"
        jobs.append({"prompt": PROMPT, "text": TEXT, "vector": vec,
                     "out": str(wav)})
        meta.append((emo, vec, wav))

    h("STAGE 2 — SYNTHESIS (the mouth speaks)")
    print(f"\n  Command the loop actually runs (vendor env, no shared imports):")
    print(f"    cd {VENDOR}")
    print(f"    PYTHONPATH=$PWD .venv/bin/python synth_worker.py <jobs.json>\n")
    jf = OUT / "_jobs.json"
    jf.write_text(json.dumps(jobs))
    t0 = time.time()
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"), str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=3600)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth failed:\n{proc.stderr[-400:]}")
    for emo, vec, wav in meta:
        kb = wav.stat().st_size // 1024
        print(f"  {wav.name}: {kb} KB synthesized")
    print(f"  total synthesis time: {time.time()-t0:.0f}s "
          f"(cached clips are skipped — the resumability law)")

    clips = [str(w) for _, _, w in meta]

    h("STAGE 3 — EAR-1: WavLM places each clip in PAD space (steering signal)")
    print("""
  What happens inside the bridge call:
    .venv_diar/bin/python -m scripts.predict_wavlm_ft --json <clips>
  The engine preprocesses (16 kHz mono), embeds with fine-tuned WavLM-large,
  and regresses three numbers per clip — the coordinates of the notebook sketch:
    V valence   -1 unpleasant .. +1 pleasant
    A arousal    0 calm       .. 1 activated
    D dominance -1 overwhelmed.. +1 in control""")
    t0 = time.time()
    vad = wavlm_vad(clips)
    print(f"\n  ({time.time()-t0:.0f}s)")
    for (emo, vec, wav), r in zip(meta, vad):
        c = CENTROIDS[emo]
        pt = (r["valence"], r["arousal"], r["dominance"])
        d = math.sqrt(sum((pt[i] - c[i]) ** 2 for i in range(3)))
        print(f"""
  {wav.name}  ->  V={pt[0]:+.2f}  A={pt[1]:+.2f}  D={pt[2]:+.2f}   (wavlm's own family read: {r['emotion']})
    target centroid ({emo}, fit on 137k MSP clips): V={c[0]:+.2f} A={c[1]:+.2f} D={c[2]:+.2f}
    distance = sqrt((dV)^2+(dA)^2+(dD)^2) = {d:.3f}
    -> this DISTANCE is what the optimizer minimizes. It never sees the verdict.""")

    h("STAGE 4 — EAR-2: frozen emotion2vec names the family (the verdict)")
    print("""
  What happens inside:
    .venv_diar/bin/python -m scripts.adaptors predict --backbone emotion2vec_plus_large
  Retrieval, not regression: embed the clip, find nearest labeled neighbors by
  cosine (the 'Shazam for emotion'), vote. Confidence = vote share, NOT evidence.""")
    t0 = time.time()
    fam = e2v_family(clips)
    print(f"\n  ({time.time()-t0:.0f}s)")

    h("STAGE 5 — THE RULE: who gets to declare success")
    rows = []
    n = sum(1 for _ in open(LEDGER)) - 1
    for (emo, vec, wav), r, f in zip(meta, vad, fam):
        c = CENTROIDS[emo]
        pt = (r["valence"], r["arousal"], r["dominance"])
        d = math.sqrt(sum((pt[i] - c[i]) ** 2 for i in range(3)))
        hit = f["family"] == emo
        n += 1
        rows.append([n, "observatory-demo", emo,
                     f"emo_vector={vec};spk=ravdess_neutral_A01",
                     round(pt[0], 3), round(pt[1], 3), round(pt[2], 3),
                     f["family"], round(f["confidence"], 2), round(d, 3), int(hit)])
        verdict = "HIT ✓" if hit else "miss ✗"
        print(f"""
  {wav.name}
    steering says: distance {d:.3f} to {emo}
    judge says:    {f['family']} @ {f['confidence']:.0%}
    HIT only if judge names the target -> {verdict}""")
        if not hit and emo == "sadness":
            print("""    ^ THIS is the documented blind spot: acoustics sit nearly on the
      sadness centroid, humans hear sadness (blind protocol, 3/3 controls),
      but frozen e2v locks synthetic sadness to 'neutral'. We documented it
      instead of retraining the judge — the judge-frozen law.""")

    with open(LEDGER, "a", newline="") as fp:
        csv.writer(fp).writerows(rows)

    h("STAGE 6 — THE LEDGER (the loop's memory)")
    print(f"\n  Every clip becomes a row — hits AND misses. Rows just appended:")
    for row in rows:
        print(f"    {row}")
    print(f"\n  Ledger total: {n} rows. This file is the transfer map's raw data,")
    print(f"  the Phase-5 training corpus, and the judge-characterization evidence")
    print(f"  — all at once: {LEDGER}")

    h("STAGE 7 — WHAT THE OPTIMIZER WOULD DO NEXT (deterministic, no RNG)")
    for (emo, vec, wav), r in zip(meta, vad):
        c = CENTROIDS[emo]
        v_err, a_err = r["valence"] - c[0], r["arousal"] - c[1]
        if abs(a_err) >= abs(v_err):
            move = ("arousal too HIGH -> add calm" if a_err > 0.08 else
                    "arousal too LOW -> push dominant knob x1.25" if a_err < -0.08
                    else "arousal on target -> scale dominant knob +/-25% and re-judge")
        else:
            move = ("valence too NEGATIVE -> try a warmer knob mix" if v_err < 0
                    else "valence on target-ish -> intensify")
        print(f"  {emo:8s} V_err={v_err:+.2f} A_err={a_err:+.2f} -> {move}")

    print("\nOBSERVATORY_DONE")


if __name__ == "__main__":
    main()
