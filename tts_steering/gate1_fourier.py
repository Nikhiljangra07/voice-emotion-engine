"""P4.9b GATE 1 — knob-choice sanity in the Fourier-fingerprint basis.

Uses the weighting that passed Gate 0 (W_deriv — the user's "derivation in the
equations"). Directions fit from ALL RAVDESS actors (validation already done on
held-out actors in fourier_gate.py). Same pre-registered bar as P4.8:
3/4 targets must rank a sane knob first, anger mandatory.

Run:  venv/bin/python tts_steering/gate1_fourier.py     ($0, no synthesis)
"""

import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from fourier_gate import (  # noqa: E402
    BASELINE, CALIB, K, TARGETS, WEIGHTS, fingerprint, wcos)
from src.utils.dataset_loader import load_ravdess  # noqa: E402

SANE = {"anger": {"angry"}, "surprise": {"surprised"},
        "joy": {"happy"}, "sadness": {"sad", "melancholic"}}
W = WEIGHTS["deriv"]


def main() -> None:
    print("fitting directions from ALL RAVDESS actors ...", flush=True)
    fps, labels = [], []
    for s in load_ravdess(ROOT / "data/ravdess"):
        if s.label in TARGETS:
            fp = fingerprint(str(s.path))
            if fp is not None:
                fps.append(fp); labels.append(s.label)
    nfps = []
    for p in sorted(glob.glob(str(ROOT / "data/ravdess/Actor_*/*.wav"))):
        if p.split("/")[-1].split("-")[2] in ("01", "02"):
            fp = fingerprint(p)
            if fp is not None:
                nfps.append(fp)
    X, y, Xn = np.stack(fps), np.array(labels), np.stack(nfps)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1e-9
    Z, Zn = (X - mu) / sd, (Xn - mu) / sd
    neutral = Zn.mean(axis=0)
    dirs = {f: Z[y == f].mean(axis=0) - neutral for f in TARGETS}

    base = fingerprint(BASELINE)
    acc: dict = {}
    for p, (k, lvl) in CALIB.items():
        fp = fingerprint(p)
        acc.setdefault(k, []).append(((fp - mu) / sd - (base - mu) / sd) / lvl)
    eff = {k: np.mean(v, axis=0) for k, v in acc.items()}

    print("\nGATE 1 (Fourier basis, W_deriv) — knob ranking per target:\n")
    passed = {}
    for tgt in TARGETS:
        ranked = sorted(((wcos(e, dirs[tgt], W), k) for k, e in eff.items()),
                        reverse=True)
        top = ranked[0][1]
        ok = top in SANE[tgt]
        passed[tgt] = ok
        print(f"  {tgt:8s} -> " + " | ".join(
            f"{k} {s:+.2f}" for s, k in ranked[:4]))
        print(f"           top: {top}  {'SANE ✓' if ok else 'INSANE ✗'}\n")
    n_ok = sum(passed.values())
    verdict = n_ok >= 3 and passed["anger"]
    print(f"GATE 1: {n_ok}/4 sane (bar: 3/4, anger mandatory) -> "
          f"{'PASS' if verdict else 'FAIL'}")
    print("GATE1F_" + ("PASS" if verdict else "FAIL"))


if __name__ == "__main__":
    main()
