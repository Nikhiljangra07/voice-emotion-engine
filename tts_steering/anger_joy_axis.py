"""P4.13 — THE ANGER↔JOY AXIS: how little difference does the job?

User's hunch: joy and anger share high pitch + fast delivery; the separation
must be a small, specific difference — "a little bit of engineering." This
script measures that difference three ways, all from data already on disk:

  1. REAL SPEECH: per-feature effect size (Cohen's d) between joy and anger in
     MSP (natural, 74k rows) and RAVDESS (acted) — which of the 111 features
     actually carry the separation, and do the two corpora agree?
  2. THE MOUTH: project every calibration knob-effect onto the joy-anger axis —
     is there ANY knob that moves joy-ward?
  3. THE JUDGE'S RECEIPTS: forensics on the only judge-confirmed synthetic joy
     (ElevenLabs [excited][laughs], d=0.144) vs our judge-confirmed anger and
     our warmest near-joy (joyref V=+0.02) — where exactly do they differ on
     the axis features?

Run:  venv/bin/python tts_steering/anger_joy_axis.py     ($0)
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.features.feature_vector import (  # noqa: E402
    build_feature_vector, feature_names, to_array)

NAMES = feature_names()


def load_msp():
    X = np.load(ROOT / "out/features_msp_train.npy")
    y = []
    with open(ROOT / "out/meta_msp_train.csv") as f:
        for r in csv.DictReader(f):
            y.append(r["emotion"])
    return X, np.array(y)


def main() -> None:
    Xm, ym = load_msp()
    Xr = np.load(ROOT / "out/features_ravdess.npy")
    yr = np.load(ROOT / "out/labels_ravdess.npy", allow_pickle=True).astype(str)

    med = np.median(Xm, axis=0)
    iqr = np.percentile(Xm, 75, axis=0) - np.percentile(Xm, 25, axis=0)
    iqr[iqr < 1e-9] = 1e-9

    def z(X):
        return (X - med) / iqr

    Zm, Zr = z(Xm), z(Xr)

    def cohens_d(Z, y):
        j, a = Z[y == "joy"], Z[y == "anger"]
        pooled = np.sqrt((j.var(axis=0) + a.var(axis=0)) / 2)
        pooled[pooled < 1e-9] = 1e-9
        return (j.mean(axis=0) - a.mean(axis=0)) / pooled

    dm, dr = cohens_d(Zm, ym), cohens_d(Zr, yr)
    print(f"joy vs anger: MSP n={int((ym=='joy').sum())}+"
          f"{int((ym=='anger').sum())}, RAVDESS n=192+192")

    Dm = Zm[ym == "joy"].mean(axis=0) - Zm[ym == "anger"].mean(axis=0)
    Dr = Zr[yr == "joy"].mean(axis=0) - Zr[yr == "anger"].mean(axis=0)
    cos_dirs = float(Dm @ Dr / (np.linalg.norm(Dm) * np.linalg.norm(Dr)))
    print(f"axis agreement natural~acted (cosine): {cos_dirs:+.2f}\n")

    order = np.argsort(-np.abs(dm))
    print("TOP 15 separators in NATURAL speech (Cohen's d, + = more in joy):")
    print(f"  {'feature':46s} {'MSP d':>7s} {'RAV d':>7s}  agree?")
    agree = 0
    for i in order[:15]:
        s = np.sign(dm[i]) == np.sign(dr[i])
        agree += s
        print(f"  {NAMES[i][:46]:46s} {dm[i]:+7.2f} {dr[i]:+7.2f}  "
              f"{'✓' if s else '✗'}")
    print(f"  sign agreement: {agree}/15")

    # pooled unit axis (agreeing dims only, weighted by min |d|)
    mask = np.sign(dm) == np.sign(dr)
    w = np.where(mask, np.minimum(np.abs(dm), np.abs(dr)), 0.0)
    D = np.sign(dm) * w
    D = D / np.linalg.norm(D)

    # ---- 2. the mouth's knobs on this axis ----
    feats = json.loads((HERE / "out/abc_p47/features.json").read_text())
    SWEEP = HERE / "out/sweep_p42"
    P47 = HERE / "out/abc_p47"
    CAL = {"angry": [("angry_08.wav", .8), ("angry_12.wav", 1.2)],
           "happy": [("joy_08.wav", .8), ("joy_12.wav", 1.2)],
           "sad": [("sad_04.wav", .4), ("sad_12.wav", 1.2)],
           "melancholic": [("mel_08.wav", .8)], "calm": [("calm_08.wav", .8)]}
    base = (np.array(feats[str(SWEEP / "baseline_zero.wav")]) - med) / iqr
    print("\nKNOB MOVEMENT along the joy-anger axis (+ = joy-ward, per unit):")
    for k, clips in CAL.items():
        effs = []
        for fn, lvl in clips:
            v = (np.array(feats[str(SWEEP / fn)]) - med) / iqr
            effs.append((v - base) / lvl)
        proj = float(np.mean([e @ D for e in effs]))
        print(f"  {k:12s} {proj:+.3f}")
    for k, fn in [("afraid", "probe_afraid_08.wav"),
                  ("disgusted", "probe_disgusted_08.wav"),
                  ("surprised", "probe_surprised_08.wav")]:
        v = (np.array(feats[str(P47 / fn)]) - med) / iqr
        proj = float(((v - base) / 0.8) @ D)
        print(f"  {k:12s} {proj:+.3f}")

    # ---- 3. forensics on the judge's receipts ----
    print("\nCLIP FORENSICS — position on the axis (z-score projection):")
    clips = {
        "EL joy HIT (d=0.144)": HERE / "out/p44b/elevenlabs_joy_r2_4.wav",
        "EL joy HIT (d=0.277)": HERE / "out/p44b/elevenlabs_joy_r1_2.wav",
        "OUR anger HIT": HERE / "out/abc_p47/abcA_anger_r1_0.wav",
        "OUR best near-joy": HERE /
            "out/joy_ref/joyref_r1_0_rav_happy_male_100.wav",
        "OUR baseline": SWEEP / "baseline_zero.wav",
    }
    vecs = {}
    for label, p in clips.items():
        try:
            v = (to_array(build_feature_vector(str(p))) - med) / iqr
        except Exception as e:
            print(f"  {label}: EXTRACTION FAILED ({e})")
            continue
        vecs[label] = v
        print(f"  {label:24s} axis position {float(v @ D):+.3f}")

    if "EL joy HIT (d=0.144)" in vecs and "OUR best near-joy" in vecs:
        gap = vecs["EL joy HIT (d=0.144)"] - vecs["OUR best near-joy"]
        contrib = gap * D
        top = np.argsort(-contrib)[:8]
        print("\nWHAT EL'S JOY HAS THAT OURS LACKS "
              "(top axis-weighted feature gaps):")
        for i in top:
            print(f"  {NAMES[i][:46]:46s} gap={gap[i]:+6.2f} z "
                  f"(axis weight {D[i]:+.3f})")
    print("\nANGER_JOY_AXIS_DONE")


if __name__ == "__main__":
    main()
