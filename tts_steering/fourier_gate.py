"""P4.9b — THE FOURIER FINGERPRINT GATE (the user's idea, taken literally).

Previous gates summarized trajectories into statistics (slope means, spike
counts) — still integrating the shape away. Here the contour ITSELF becomes the
equation, exactly like the reels:

    F0(t)  ≈ c0/2 + Σ_{k=1..K} c_k cos(π k t)      t ∈ [0,1]  (DCT-II series)
    RMS(t) ≈ d0/2 + Σ_{k=1..K} d_k cos(π k t)

The coefficient vector IS the emotional fingerprint of the gesture. "Adding
derivation in the equation" is exact calculus, not statistics: d/dt of the k-th
harmonic multiplies it by k, so a derivative-weighted inner product (w_k = k²)
compares the FLUCTUATION of two gestures analytically.

Three weightings, declared BEFORE running (disclosed multiplicity, no tuning):
  W_shape    w_k = 1        — compare the gestures
  W_deriv    w_k = k², k>=1 — compare the gestures' derivatives (fluctuation)
  W_combined normalized sum of both

Steps:
  A. Fit fingerprints per emotion from RAVDESS (prominent foundation), actors
     split even/odd -> held-out validation of emotion separation. Print the
     fitted equations.
  B. Knob-effect collinearity in fingerprint space (calibration clips on disk).
     PRE-REGISTERED PASS BAR: all three hot pairs (happy/angry/surprised)
     < 0.80 under at least one declared weighting.

Run:  venv/bin/python tts_steering/fourier_gate.py     ($0, no synthesis)
"""

import glob
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.utils.dataset_loader import load_ravdess  # noqa: E402

SWEEP = HERE / "out/sweep_p42"
P47 = HERE / "out/abc_p47"
K = 10                      # harmonics per contour
N = 100                     # time-normalized samples
TARGETS = ["anger", "sadness", "joy", "surprise"]
HOT = ["happy", "angry", "surprised"]

CALIB = {
    str(SWEEP / "angry_08.wav"): ("angry", 0.8),
    str(SWEEP / "angry_12.wav"): ("angry", 1.2),
    str(SWEEP / "joy_08.wav"): ("happy", 0.8),
    str(SWEEP / "joy_12.wav"): ("happy", 1.2),
    str(SWEEP / "sad_04.wav"): ("sad", 0.4),
    str(SWEEP / "sad_12.wav"): ("sad", 1.2),
    str(SWEEP / "mel_08.wav"): ("melancholic", 0.8),
    str(SWEEP / "calm_08.wav"): ("calm", 0.8),
    str(P47 / "probe_afraid_08.wav"): ("afraid", 0.8),
    str(P47 / "probe_disgusted_08.wav"): ("disgusted", 0.8),
    str(P47 / "probe_surprised_08.wav"): ("surprised", 0.8),
}
BASELINE = str(SWEEP / "baseline_zero.wav")


def contours(path: str):
    """Time-normalized F0 (semitones re 55 Hz) + RMS contours, or None."""
    try:
        snd = parselmouth.Sound(path)
        f0 = snd.to_pitch(time_step=0.01).selected_array["frequency"]
    except Exception:
        return None
    voiced = f0 > 0
    if voiced.sum() < 10:
        return None
    st = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-6) / 55.0), np.nan)
    i0 = int(np.argmax(voiced))
    i1 = len(voiced) - int(np.argmax(voiced[::-1])) - 1
    seg = st[i0:i1 + 1]
    idx = np.arange(len(seg))
    good = ~np.isnan(seg)
    seg = np.interp(idx, idx[good], seg[good])       # bridge unvoiced gaps
    f0c = np.interp(np.linspace(0, len(seg) - 1, N), idx, seg)
    y, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.025),
                              hop_length=int(sr * 0.010))[0]
    rms = rms / max(rms.max(), 1e-9)
    rmsc = np.interp(np.linspace(0, len(rms) - 1, N),
                     np.arange(len(rms)), rms)
    return f0c, rmsc


_n = np.arange(N)
_BASIS = np.stack([np.cos(np.pi * k * (_n + 0.5) / N) for k in range(K + 1)])


def dct(x: np.ndarray) -> np.ndarray:
    return (2.0 / N) * (_BASIS @ x)


def fingerprint(path: str):
    c = contours(path)
    if c is None:
        return None
    return np.concatenate([dct(c[0]), dct(c[1])])    # 2*(K+1) = 22 dims


KVEC = np.concatenate([np.arange(K + 1), np.arange(K + 1)]).astype(float)
W_SHAPE = np.ones(2 * (K + 1))
W_DERIV = KVEC ** 2                                  # d/dt kills c0, scales by k
WEIGHTS = {"shape": W_SHAPE,
           "deriv": W_DERIV,
           "combined": W_SHAPE / W_SHAPE.sum() + W_DERIV / W_DERIV.sum()}


def wcos(a, b, w):
    na = math.sqrt(float((w * a * a).sum()))
    nb = math.sqrt(float((w * b * b).sum()))
    return float((w * a * b).sum()) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def main() -> None:
    # ---------------- A. foundation fingerprints (RAVDESS, prominent) --------
    samples = [(s.path, s.label) for s in load_ravdess(ROOT / "data/ravdess")
               if s.label in TARGETS]
    neutral_files = sorted(
        f for f in glob.glob(str(ROOT / "data/ravdess/Actor_*/*.wav"))
        if f.split("/")[-1].split("-")[2] in ("01", "02"))
    print(f"extracting contours: {len(samples)} emotion + "
          f"{len(neutral_files)} neutral clips ...", flush=True)

    def actor(p: str) -> int:
        return int(Path(p).parent.name.split("_")[1])

    fps, labels, actors = [], [], []
    for p, lab in samples:
        fp = fingerprint(str(p))
        if fp is not None:
            fps.append(fp); labels.append(lab); actors.append(actor(str(p)))
    nfps, nactors = [], []
    for p in neutral_files:
        fp = fingerprint(p)
        if fp is not None:
            nfps.append(fp); nactors.append(actor(p))
    X = np.stack(fps); y = np.array(labels); act = np.array(actors)
    Xn = np.stack(nfps); actn = np.array(nactors)
    print(f"  usable: {len(y)} emotion, {len(Xn)} neutral")

    # standardize dims on TRAIN actors only (even) — no leakage
    tr, te = act % 2 == 0, act % 2 == 1
    trn = actn % 2 == 0
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0); sd[sd < 1e-9] = 1e-9
    Z, Zn = (X - mu) / sd, (Xn - mu) / sd
    neutral = Zn[trn].mean(axis=0)
    fams = {f: Z[tr & (y == f)].mean(axis=0) for f in TARGETS}
    dirs = {f: fams[f] - neutral for f in TARGETS}

    # ---- print the fitted equations (raw coefficient space, first 6 terms) --
    print("\nTHE FINGERPRINT EQUATIONS (RAVDESS, t in [0,1]):")
    for f in TARGETS + ["neutral"]:
        c = (X[y == f].mean(axis=0) if f != "neutral" else Xn.mean(axis=0))
        f0t = " ".join(f"{c[k]:+.2f}cos({k}πt)" for k in range(1, 7))
        rmt = " ".join(f"{c[K+1+k]:+.2f}cos({k}πt)" for k in range(1, 7))
        print(f"  {f:8s} F0(t)[st] = {c[0]/2:6.2f} {f0t}")
        print(f"           E(t)      = {c[K+1]/2:6.2f} {rmt}")

    # ---- held-out separation (odd actors), per declared weighting ----------
    print("\nheld-out separation (odd actors, nearest-direction, "
          "chance 25% / hot 33%):")
    ach = Z[te] - neutral
    yte = y[te]
    val = {}
    for wname, w in WEIGHTS.items():
        S = np.stack([[wcos(a, dirs[f], w) for f in TARGETS] for a in ach])
        pred = np.array(TARGETS)[S.argmax(axis=1)]
        acc = float((pred == yte).mean())
        hot_m = np.isin(yte, ["anger", "joy", "surprise"])
        S3 = np.stack([[wcos(a, dirs[f], w)
                        for f in ["anger", "joy", "surprise"]]
                       for a in ach[hot_m]])
        p3 = np.array(["anger", "joy", "surprise"])[S3.argmax(axis=1)]
        hacc = float((p3 == yte[hot_m]).mean())
        val[wname] = {"acc4": acc, "hot3": hacc}
        print(f"  W_{wname:9s} 4-way {acc:.1%}   hot-3-way {hacc:.1%}")

    # ---------------- B. knob effects in fingerprint space -------------------
    base = fingerprint(BASELINE)
    acc_k: dict = {}
    for p, (k, lvl) in CALIB.items():
        fp = fingerprint(p)
        acc_k.setdefault(k, []).append(((fp - mu) / sd - (base - mu) / sd) / lvl)
    eff = {k: np.mean(v, axis=0) for k, v in acc_k.items()}
    knobs = ["happy", "angry", "surprised", "afraid", "sad", "melancholic",
             "disgusted", "calm"]

    verdicts = {}
    for wname, w in WEIGHTS.items():
        print(f"\nknob-effect collinearity — W_{wname}:")
        print("            " + "".join(f"{k[:6]:>8s}" for k in knobs))
        for a in knobs:
            print(f"{a:12s}" + "".join(
                f"{wcos(eff[a], eff[b], w):+8.2f}" for b in knobs))
        pairs = {f"{a}|{b}": wcos(eff[a], eff[b], w)
                 for i, a in enumerate(HOT) for b in HOT[i + 1:]}
        ok = all(v < 0.80 for v in pairs.values())
        verdicts[wname] = {"pairs": pairs, "pass": ok}
        print("  hot pairs: " + "  ".join(
            f"{p}={v:+.2f}{'✓' if v < 0.80 else '✗'}"
            for p, v in pairs.items()))

    any_pass = any(v["pass"] for v in verdicts.values())
    print(f"\nGATE (pre-registered: all hot pairs < 0.80 under >=1 of the 3 "
          f"declared weightings): {'PASS' if any_pass else 'FAIL'}")
    (HERE / "out/fourier_gate.json").write_text(json.dumps(
        {"validation_heldout": val,
         "knob_verdicts": {k: {"pairs": v["pairs"], "pass": v["pass"]}
                           for k, v in verdicts.items()},
         "pass": any_pass}, indent=1))
    print("FOURIER_GATE_" + ("PASS" if any_pass else "FAIL"))


if __name__ == "__main__":
    main()
