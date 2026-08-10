"""P4.9d — THE EPICYCLE GATE: 3-axis Fourier with cross-axis circles.

User's idea: the 1-axis series was the shadow; make it a curve in 3-D acoustic
space and put "circles on specific points" — the epicycle (complex-Fourier)
form, whose CROSS-AXIS PHASE terms encode what no single-axis series can:
does pitch lead energy or lag it, does the gesture rotate clockwise
(spike-then-fall) or counter-clockwise (swell) in each plane.

The three axes (chosen to attack the two documented blockages):
  X  F0(t)   semitones      — melody (as before)
  Y  E(t)    norm. RMS      — energy (as before)
  Z  HNR(t)  dB             — VOICE QUALITY over time: breathiness/instability
                              dynamics, the exact ingredient the sadness
                              diagnosis said was missing.

Per clip, per axis: complex spectrum c_k (k=1..K) on the common time grid.
Features:
  |c_k| per axis                      (3K)   — the shapes
  Im(c_k^a conj(c_k^b)) per pair      (3K)   — signed rotation (lead/lag) —
                                              time-shift INVARIANT, new physics
  Re(c_k^a conj(c_k^b)) per pair      (3K)   — in-phase coupling
Weightings declared before running: W_shape (ones) and W_deriv (harmonic index
k on every dim). Two declared, pass under either; no tuning after.

PRE-REGISTERED PASS (declared now): matched-text knob sanity with contrast
scoring must reach >=3/4 sane INCLUDING anger, AND at least one of
sadness/surprise must flip to sane (that is the point — "open the blockages").
Fear/disgust cannot be unblocked by any representation (judge vocabulary) and
are out of scope, stated upfront.

Run:  venv/bin/python tts_steering/epicycle_gate.py     ($0, existing clips)
"""

import glob
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

N = 100
K = 10
TARGETS = ["anger", "sadness", "joy", "surprise"]
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
HOT = ["happy", "angry", "surprised"]
SANE = {"anger": {"angry"}, "surprise": {"surprised"},
        "joy": {"happy"}, "sadness": {"sad", "melancholic"}}
CALIB = HERE / "out/fourier_calib"


def tracks(path: str):
    """(F0_st, E, HNR) on a common N-point normalized time grid, or None."""
    try:
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch(time_step=0.01)
        harm = snd.to_harmonicity(time_step=0.01)
    except Exception:
        return None
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    if voiced.sum() < 10:
        return None
    st = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-6) / 55.0), np.nan)
    i0 = int(np.argmax(voiced))
    i1 = len(voiced) - int(np.argmax(voiced[::-1])) - 1
    seg = st[i0:i1 + 1]
    idx = np.arange(len(seg))
    good = ~np.isnan(seg)
    seg = np.interp(idx, idx[good], seg[good])
    grid = np.linspace(0, len(seg) - 1, N)
    x = np.interp(grid, idx, seg)

    h = harm.values[0].copy()
    h[h < -50] = np.nan                              # silence sentinel
    hi = np.arange(len(h))
    hg = ~np.isnan(h)
    if hg.sum() < 5:
        return None
    h = np.interp(hi, hi[hg], h[hg])
    z = np.interp(np.linspace(0, len(h) - 1, N), hi, h)

    y_, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y_, frame_length=int(sr * 0.025),
                              hop_length=int(sr * 0.010))[0]
    rms = rms / max(rms.max(), 1e-9)
    y = np.interp(np.linspace(0, len(rms) - 1, N), np.arange(len(rms)), rms)
    return x, y, z


def epicycle_fp(path: str):
    t = tracks(path)
    if t is None:
        return None
    spectra = []
    for a in t:
        c = np.fft.rfft(a - a.mean())[1:K + 1] / N   # harmonics 1..K
        spectra.append(c)
    feats = [np.abs(c) for c in spectra]             # 3 x K magnitudes
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        cross = spectra[i] * np.conj(spectra[j])
        feats.append(np.imag(cross))                 # rotation (lead/lag)
    for i, j in pairs:
        cross = spectra[i] * np.conj(spectra[j])
        feats.append(np.real(cross))                 # in-phase coupling
    return np.concatenate(feats)                     # 9K = 90 dims


KH = np.tile(np.arange(1, K + 1), 9).astype(float)
WEIGHTS = {"shape": np.ones(9 * K), "deriv": KH}


def wcos(a, b, w):
    na = np.sqrt(float((w * a * a).sum()))
    nb = np.sqrt(float((w * b * b).sum()))
    return float((w * a * b).sum()) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def main() -> None:
    from src.utils.dataset_loader import load_ravdess

    # ---- RAVDESS statement-01 (matched text with kids_* calibration) --------
    print("extracting 3-axis epicycle fingerprints (RAVDESS statement-01) ...",
          flush=True)
    fps, labels, actors = [], [], []
    for s in load_ravdess(ROOT / "data/ravdess"):
        nm = Path(s.path).name.split("-")
        if s.label in TARGETS and nm[4] == "01":
            fp = epicycle_fp(str(s.path))
            if fp is not None:
                fps.append(fp); labels.append(s.label)
                actors.append(int(Path(s.path).parent.name.split("_")[1]))
    nfps, nactors = [], []
    for p in sorted(glob.glob(str(ROOT / "data/ravdess/Actor_*/*.wav"))):
        nm = p.split("/")[-1].split("-")
        if nm[2] in ("01", "02") and nm[4] == "01":
            fp = epicycle_fp(p)
            if fp is not None:
                nfps.append(fp)
                nactors.append(int(Path(p).parent.name.split("_")[1]))
    X, y, act = np.stack(fps), np.array(labels), np.array(actors)
    Xn, actn = np.stack(nfps), np.array(nactors)
    print(f"  {len(y)} emotion + {len(Xn)} neutral clips")

    # ---- held-out separation vs 2-axis baseline (even fit / odd test) ------
    tr, te = act % 2 == 0, act % 2 == 1
    trn = actn % 2 == 0
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0); sd[sd < 1e-9] = 1e-9
    Z, Zn = (X - mu) / sd, (Xn - mu) / sd
    neutral = Zn[trn].mean(axis=0)
    dirs = {f: Z[tr & (y == f)].mean(axis=0) - neutral for f in TARGETS}
    print("\nheld-out separation (odd actors, nearest-direction):")
    for wname, w in WEIGHTS.items():
        S = np.stack([[wcos(a, dirs[f], w) for f in TARGETS]
                      for a in Z[te] - neutral])
        pred = np.array(TARGETS)[S.argmax(axis=1)]
        acc = float((pred == y[te]).mean())
        rec = {f: float((pred[y[te] == f] == f).mean()) for f in TARGETS}
        print(f"  W_{wname:6s} 4-way {acc:.1%} (chance 25%)  " +
              " ".join(f"{f[:4]} {r:.0%}" for f, r in rec.items()))

    # ---- knob effects (matched-text kids clips) ----------------------------
    base = epicycle_fp(str(CALIB / "kids_baseline.wav"))
    eff = {}
    for k in DIMS:
        fp = epicycle_fp(str(CALIB / f"kids_{k}_08.wav"))
        eff[k] = ((fp - mu) / sd - (base - mu) / sd) / 0.8

    print("\nknob distinguishability (hot pairs + the blocked knobs):")
    for wname, w in WEIGHTS.items():
        hp = {f"{a}|{b}": wcos(eff[a], eff[b], w)
              for i, a in enumerate(HOT) for b in HOT[i + 1:]}
        extra = wcos(eff["sad"], eff["surprised"], w)
        print(f"  W_{wname:6s} " + "  ".join(
            f"{p}={v:+.2f}" for p, v in hp.items())
            + f"  sad|surprised={extra:+.2f}")

    # ---- THE PRE-REGISTERED GATE: contrast sanity, all-actor directions ----
    mu2 = X.mean(axis=0)
    sd2 = X.std(axis=0); sd2[sd2 < 1e-9] = 1e-9
    Z2 = (X - mu2) / sd2
    fams = {f: Z2[y == f].mean(axis=0) for f in TARGETS}
    base2 = (epicycle_fp(str(CALIB / "kids_baseline.wav")) - mu2) / sd2
    eff2 = {}
    for k in DIMS:
        fp = (epicycle_fp(str(CALIB / f"kids_{k}_08.wav")) - mu2) / sd2
        eff2[k] = (fp - base2) / 0.8

    print("\nPRE-REGISTERED GATE — contrast sanity (matched text):")
    final = {}
    for wname, w in WEIGHTS.items():
        print(f"  W_{wname}:")
        passed = {}
        for tgt in TARGETS:
            cons = [fams[tgt] - fams[g] for g in TARGETS if g != tgt]
            ranked = sorted(((min(wcos(eff2[k], c, w) for c in cons), k)
                             for k in DIMS), reverse=True)
            top = ranked[0][1]
            ok = top in SANE[tgt]
            passed[tgt] = ok
            print(f"    {tgt:8s} -> " + " | ".join(
                f"{k} {s:+.2f}" for s, k in ranked[:4])
                + f"   top: {top} {'✓' if ok else '✗'}")
        n_ok = sum(passed.values())
        unblocked = passed["sadness"] or passed["surprise"]
        verdict = n_ok >= 3 and passed["anger"] and unblocked
        final[wname] = verdict
        print(f"    -> {n_ok}/4 sane, anger {'✓' if passed['anger'] else '✗'}, "
              f"blockage flipped {'✓' if unblocked else '✗'} : "
              f"{'PASS' if verdict else 'FAIL'}")
    print("\nEPICYCLE_GATE_" + ("PASS" if any(final.values()) else "FAIL"))


if __name__ == "__main__":
    main()
