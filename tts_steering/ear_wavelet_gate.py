"""P4.10 — THE EAR GATE: Gabor/wavelet micro-gestures for the blind emotions.

Direction shift, user-approved: stop sharpening the mouth's compass (wall
relocated to the mouth in P4.9d) and aim the "quantum math" — Gabor's acoustic
quanta / uncertainty-principle localization — at the EAR's two blind families:

  FEAR     tremor: 3-10 Hz micro-fluctuation of pitch/energy/voice quality —
           invisible to global, time-normalized harmonics by construction.
  SURPRISE transient: millisecond-scale spikes whose ABSOLUTE suddenness the
           global series smears away.

Features (absolute time, 10 ms grid, per axis F0/E/HNR — 48 dims):
  tremor band  3-10 Hz : band RMS, dominant modulation freq, burst count/s,
                         max burst amplitude
  spike band  10-40 Hz : same four stats
  Gabor scales 50/100/200/400 ms : max |x - G_s(x)| / std, response rate
(All per-axis stats normalized by that axis's own std -> level-invariant.)

Benchmark protocol (identical for every basis — self-contained, no
cross-referencing older runs):
  corpora   RAVDESS full (6 fams + neutral) · CREMA-D full (5 fams + neutral)
            · MSP sample (6 fams + neutral, all fear/disgust kept, seed 42)
  split     held-out SPEAKERS (even fit / odd test) within each corpus
  classifier nearest class centroid on standardized dims (declared, same for all)
  bases     A) 1-axis DCT (22) B) 3-axis epicycle (90) C) wavelet (48)
            D) epicycle+wavelet (138)

PRE-REGISTERED PASS (declared before running, one shot):
  mean held-out FEAR recall of D >= B + 10 points, AND
  mean held-out SURPRISE recall of D >= B - 5 points.
If it passes, this becomes an EAR-V2 CANDIDATE feature set (Project 1, future,
versioned). The frozen judge is not touched — judge-frozen law absolute.

Run:  venv/bin/python tts_steering/ear_wavelet_gate.py      ($0, local audio)
"""

import csv
import glob
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OUT = HERE / "out/ear_wavelet"
OUT.mkdir(parents=True, exist_ok=True)
FAMS = ["anger", "sadness", "joy", "surprise", "fear", "disgust", "neutral"]
FS = 100.0                     # 10 ms grid
K = 10
N = 100
CREMA_MAP = {"ANG": "anger", "SAD": "sadness", "HAP": "joy", "NEU": "neutral",
             "FEA": "fear", "DIS": "disgust"}
MSP_PER_FAM = 1000


def tracks_abs(path: str):
    """F0(st), E, HNR on the ABSOLUTE 10 ms grid (no time normalization)."""
    try:
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch(time_step=0.01)
        harm = snd.to_harmonicity(time_step=0.01)
    except Exception:
        return None
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    if voiced.sum() < 20:
        return None
    st = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-6) / 55.0), np.nan)
    i0 = int(np.argmax(voiced))
    i1 = len(voiced) - int(np.argmax(voiced[::-1])) - 1
    seg = st[i0:i1 + 1]
    if len(seg) < 80:                       # < 0.8 s voiced span: too short
        return None
    idx = np.arange(len(seg))
    good = ~np.isnan(seg)
    x = np.interp(idx, idx[good], seg[good])

    h = harm.values[0].copy()
    h[h < -50] = np.nan
    hg = ~np.isnan(h)
    if hg.sum() < 5:
        return None
    hi = np.arange(len(h))
    h = np.interp(hi, hi[hg], h[hg])
    # align HNR to the voiced span (same 10 ms clock, praat offsets ~equal)
    z = h[min(i0, len(h) - 1):min(i1 + 1, len(h))]
    if len(z) < 40:
        return None

    y_, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y_, frame_length=int(sr * 0.025),
                              hop_length=int(sr * 0.010))[0]
    rms = rms / max(rms.max(), 1e-9)
    y = rms[min(i0, len(rms) - 1):min(i1 + 1, len(rms))]
    if len(y) < 40:
        return None
    return x, y, z


def _band_stats(x: np.ndarray, lo: float, hi: float) -> list:
    nyq = FS / 2
    hi = min(hi, nyq * 0.95)
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    sd = max(float(np.std(x)), 1e-9)
    try:
        bp = filtfilt(b, a, x)
    except ValueError:
        return [0.0, 0.0, 0.0, 0.0]
    rmsb = float(np.sqrt(np.mean(bp ** 2))) / sd
    spec = np.abs(np.fft.rfft(bp))
    freqs = np.fft.rfftfreq(len(bp), 1 / FS)
    m = (freqs >= lo) & (freqs <= hi)
    dom = float(freqs[m][np.argmax(spec[m])]) if m.any() else 0.0
    env = np.abs(bp) / sd
    mad = max(float(np.median(np.abs(env - np.median(env)))), 1e-9)
    peaks, _ = find_peaks(env, height=3 * mad)
    dur = len(x) / FS
    return [rmsb, dom, len(peaks) / dur, float(env.max())]


def feats_wavelet(t) -> np.ndarray:
    out = []
    for x in t:
        sd = max(float(np.std(x)), 1e-9)
        out += _band_stats(x, 3.0, 10.0)
        out += _band_stats(x, 10.0, 40.0)
        dur = len(x) / FS
        for scale_ms in (50, 100, 200, 400):
            sig = scale_ms / 10.0 / 2.355        # FWHM -> sigma in samples
            resp = np.abs(x - gaussian_filter1d(x, sig)) / sd
            mad = max(float(np.median(np.abs(resp - np.median(resp)))), 1e-9)
            pk, _ = find_peaks(resp, height=3 * mad)
            out += [float(resp.max()), len(pk) / dur]
    return np.array(out)                          # 3*(8+8) = 48


def _grid(x):
    return np.interp(np.linspace(0, len(x) - 1, N), np.arange(len(x)), x)


def feats_dct(t) -> np.ndarray:
    n = np.arange(N)
    basis = np.stack([np.cos(np.pi * k * (n + 0.5) / N) for k in range(K + 1)])
    return np.concatenate([(2 / N) * (basis @ _grid(a)) for a in t[:2]])  # 22


def feats_epicycle(t) -> np.ndarray:
    spectra = [np.fft.rfft(_grid(a) - _grid(a).mean())[1:K + 1] / N
               for a in t]
    feats = [np.abs(c) for c in spectra]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        feats.append(np.imag(spectra[i] * np.conj(spectra[j])))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        feats.append(np.real(spectra[i] * np.conj(spectra[j])))
    return np.concatenate(feats)                  # 90


def extract(name: str, items: list) -> tuple:
    cache = OUT / f"{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return (d["D"], d["E"], d["Wv"], d["y"].astype(str),
                d["spk"].astype(int), int(d["failed"]))
    D, E, Wv, ys, spks, failed = [], [], [], [], [], 0
    for i, (p, lab, spk) in enumerate(items):
        t = tracks_abs(str(p))
        if t is None:
            failed += 1
            continue
        try:
            D.append(feats_dct(t)); E.append(feats_epicycle(t))
            Wv.append(feats_wavelet(t)); ys.append(lab); spks.append(spk)
        except Exception:
            failed += 1
        if (i + 1) % 1000 == 0:
            print(f"  {name}: {i+1}/{len(items)}", flush=True)
    D, E, Wv = np.stack(D), np.stack(E), np.stack(Wv)
    y, spk = np.array(ys), np.array(spks)
    np.savez(cache, D=D, E=E, Wv=Wv, y=y, spk=spk, failed=failed)
    return D, E, Wv, y, spk, failed


def recall_table(X, y, spk) -> dict:
    tr, te = spk % 2 == 0, spk % 2 == 1
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0); sd[sd < 1e-9] = 1e-9
    Z = (X - mu) / sd
    fams = [f for f in FAMS if (y[tr] == f).any()]
    C = np.stack([Z[tr & (y == f)].mean(axis=0) for f in fams])
    d2 = ((Z[te][:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    pred = np.array(fams)[d2.argmin(axis=1)]
    return {f: float((pred[y[te] == f] == f).mean())
            for f in fams if (y[te] == f).any()}


def main() -> None:
    from src.utils.dataset_loader import load_ravdess

    rav = [(str(s.path), s.label,
            int(Path(s.path).parent.name.split("_")[1]))
           for s in load_ravdess(ROOT / "data/ravdess")]
    rav += [(p, "neutral", int(Path(p).parent.name.split("_")[1]))
            for p in sorted(glob.glob(str(ROOT / "data/ravdess/Actor_*/*.wav")))
            if p.split("/")[-1].split("-")[2] in ("01", "02")]

    crema = []
    for p in sorted(glob.glob(str(ROOT / "data/crema_d/audios/*.wav"))):
        parts = Path(p).name.split("_")
        if parts[2] in CREMA_MAP:
            crema.append((p, CREMA_MAP[parts[2]], int(parts[0])))

    rng = random.Random(42)
    by_fam: dict = {}
    with open(ROOT / "out/meta_msp_train.csv") as f:
        for r in csv.DictReader(f):
            if r["emotion"] in FAMS:
                by_fam.setdefault(r["emotion"], []).append(
                    (r["filename"], int(r["speaker"]) if
                     r["speaker"].isdigit() else hash(r["speaker"]) % 10000))
    msp = []
    for fam, files in by_fam.items():
        take = files if fam in ("fear", "disgust") \
            else rng.sample(files, min(MSP_PER_FAM, len(files)))
        for fn, spk in take:
            msp.append((str(ROOT / "data/msp_podcast/Audios" / fn), fam, spk))

    BASES = ["dct", "epicycle", "wavelet", "epi+wav"]
    results: dict = {b: {} for b in BASES}
    for name, items in [("ravdess", rav), ("crema", crema), ("msp", msp)]:
        print(f"extracting {name}: {len(items)} clips ...", flush=True)
        D, E, Wv, y, spk, failed = extract(name, items)
        print(f"  {name}: {len(y)} ok, {failed} failed/short", flush=True)
        for bname, X in [("dct", D), ("epicycle", E), ("wavelet", Wv),
                         ("epi+wav", np.hstack([E, Wv]))]:
            results[bname][name] = recall_table(X, y, spk)

    print("\nHELD-OUT SPEAKER RECALL BY BASIS "
          "(nearest-centroid, identical protocol):")
    corp = ["ravdess", "crema", "msp"]
    for f in FAMS:
        print(f"\n  {f.upper()}")
        for b in BASES:
            row = "  ".join(
                f"{c}:{results[b][c][f]:.0%}" if f in results[b][c] else
                f"{c}:  --" for c in corp)
            means = [results[b][c][f] for c in corp if f in results[b][c]]
            print(f"    {b:9s} {row}   mean {np.mean(means):.1%}")

    def mean_recall(b, f):
        vals = [results[b][c][f] for c in corp if f in results[b][c]]
        return float(np.mean(vals))

    fear_gain = mean_recall("epi+wav", "fear") - mean_recall("epicycle", "fear")
    surp_drop = mean_recall("epicycle", "surprise") - mean_recall("epi+wav",
                                                                 "surprise")
    print(f"\nPRE-REGISTERED CHECK: fear gain (epi+wav vs epicycle) = "
          f"{fear_gain:+.1%} (need >= +10pp); surprise drop = {surp_drop:+.1%} "
          f"(need <= 5pp)")
    ok = fear_gain >= 0.10 and surp_drop <= 0.05
    print("EAR_WAVELET_GATE_" + ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
