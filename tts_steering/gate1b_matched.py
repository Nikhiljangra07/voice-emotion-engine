"""P4.9b GATE 1-bis — matched-text control for the Fourier fingerprint gate.

Gate 1 failed with knob effects measured on S1 while RAVDESS directions come
from "Kids are talking by the door" — contour equations are text-sensitive, so
the transfer gap may be the sentence, not the fingerprint idea. Control: the
mouth speaks THE RAVDESS SENTENCE for calibration (12 unscored clips: 8 single
knobs @0.8 + zero baseline; disclosed calibration synthesis, judge never
involved), then Gate 1 reruns with text held constant on both sides.

Run:  .venv_tts/bin/python tts_steering/gate1b_matched.py
"""

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENDOR = HERE / "vendor/index-tts"
OUT = HERE / "out/fourier_calib"
OUT.mkdir(parents=True, exist_ok=True)

TEXT = "Kids are talking by the door"        # RAVDESS statement 01
PROMPT = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
TARGETS = ["anger", "sadness", "joy", "surprise"]
SANE = {"anger": {"angry"}, "surprise": {"surprised"},
        "joy": {"happy"}, "sadness": {"sad", "melancholic"}}


def synth_calib() -> dict:
    jobs, clips = [], {}
    for i, k in enumerate(DIMS):
        v = [0.0] * 8
        v[i] = 0.8
        wav = OUT / f"kids_{k}_08.wav"
        jobs.append({"prompt": PROMPT, "text": TEXT, "vector": v,
                     "out": str(wav)})
        clips[k] = str(wav)
    basewav = OUT / "kids_baseline.wav"
    jobs.append({"prompt": PROMPT, "text": TEXT, "vector": [0.0] * 8,
                 "out": str(basewav)})
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if todo:
        jf = OUT / "_jobs.json"
        jf.write_text(json.dumps(todo))
        print(f"synthesizing {len(todo)} matched-text calibration clips ...",
              flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
             str(jf)],
            cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                                  "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=3600)
        if "WORKER_DONE" not in proc.stdout:
            raise RuntimeError(f"synth failed:\n{proc.stderr[-400:]}")
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return clips | {"__baseline__": str(basewav)}


def main() -> None:
    clips = synth_calib()

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    from fourier_gate import TARGETS as _T  # noqa: F401  (env check)
    from fourier_gate import WEIGHTS, fingerprint, wcos
    from src.utils.dataset_loader import load_ravdess

    print("fitting RAVDESS directions (statement-01 clips ONLY — fully "
          "matched text) ...", flush=True)
    fps, labels = [], []
    for s in load_ravdess(ROOT / "data/ravdess"):
        # RAVDESS filename field 5 = statement (01 = "Kids are talking...")
        if s.label in TARGETS and Path(s.path).name.split("-")[4] == "01":
            fp = fingerprint(str(s.path))
            if fp is not None:
                fps.append(fp); labels.append(s.label)
    nfps = []
    for p in sorted(glob.glob(str(ROOT / "data/ravdess/Actor_*/*.wav"))):
        n = p.split("/")[-1].split("-")
        if n[2] in ("01", "02") and n[4] == "01":
            fp = fingerprint(p)
            if fp is not None:
                nfps.append(fp)
    X, y, Xn = np.stack(fps), np.array(labels), np.stack(nfps)
    print(f"  {len(y)} emotion + {len(Xn)} neutral clips (statement 01)")
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1e-9
    Z, Zn = (X - mu) / sd, (Xn - mu) / sd
    dirs = {f: Z[y == f].mean(axis=0) - Zn.mean(axis=0) for f in TARGETS}

    base = fingerprint(clips["__baseline__"])
    eff = {}
    for k in DIMS:
        fp = fingerprint(clips[k])
        eff[k] = ((fp - mu) / sd - (base - mu) / sd) / 0.8

    for wname in ("deriv", "shape", "combined"):
        W = WEIGHTS[wname]
        print(f"\nGATE 1-bis (matched text, W_{wname}):")
        passed = {}
        for tgt in TARGETS:
            ranked = sorted(((wcos(eff[k], dirs[tgt], W), k) for k in DIMS),
                            reverse=True)
            top = ranked[0][1]
            ok = top in SANE[tgt]
            passed[tgt] = ok
            print(f"  {tgt:8s} -> " + " | ".join(
                f"{k} {s:+.2f}" for s, k in ranked[:4])
                + f"   top: {top} {'✓' if ok else '✗'}")
        n_ok = sum(passed.values())
        verdict = n_ok >= 3 and passed["anger"]
        print(f"  -> {n_ok}/4 sane (bar 3/4, anger mandatory): "
              f"{'PASS' if verdict else 'FAIL'}")
        if wname == "deriv":
            print("GATE1B_" + ("PASS" if verdict else "FAIL"))


if __name__ == "__main__":
    main()
