"""P4.9 GATE 0 — do DERIVATIVES give the compass grip on the hot knobs?

The wall (P4.8): in static classical features, the mouth's happy/angry/surprised
knobs are ~95% collinear — same acoustic action, indistinguishable, unsteerable.
The hypothesis (user's): the difference lives in the FLUCTUATION — the temporal
shape static functionals integrate away (surprise = spike, anger = plateau,
joy = melody).

This gate costs $0: dynamics features are computed from the calibration clips
already on disk (same fixed sentence S1, same voice, so text-dependence of
fluctuation features cancels exactly for knob comparison).

PRE-REGISTERED PASS BAR: in the combined basis (32 static-z + dynamics-z), all
three hot-knob pairwise cosines (happy/angry/surprised) drop below 0.80.

Run:  venv/bin/python tts_steering/gate0_dynamics.py
"""

import json
import math
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "out/sweep_p42"
P47 = HERE / "out/abc_p47"

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
HOT = ["happy", "angry", "surprised"]

DYN_NAMES = [
    "f0_slope_mean", "f0_abs_slope_mean", "f0_slope_std", "f0_curvature",
    "f0_spike_rate", "f0_rise_fall_ratio", "f0_final_slope",
    "rms_mod_rate", "rms_mod_depth", "rms_slope_mean", "rms_attack_asym",
    "rms_curvature", "voiced_run_mean_s", "voicing_switch_rate",
]


def dynamics(path: str) -> np.ndarray:
    """14 fluctuation features from the F0 and energy TRAJECTORIES."""
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(time_step=0.01)
    f0 = pitch.selected_array["frequency"]          # Hz, 0 = unvoiced
    voiced = f0 > 0
    y, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.025),
                              hop_length=int(sr * 0.010))[0]
    dur = len(y) / sr

    # --- F0 contour derivatives (semitones, voiced frames, 10 ms grid) ---
    st = np.full_like(f0, np.nan)
    st[voiced] = 12 * np.log2(f0[voiced] / 55.0)
    runs, cur = [], []                               # voiced runs of semitone values
    for v, s in zip(voiced, st):
        if v:
            cur.append(s)
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    d1 = np.concatenate([np.diff(r) for r in runs if len(r) > 1]) * 100  # st/s
    d2 = np.concatenate([np.diff(r, 2) for r in runs if len(r) > 2]) * 100
    stv = st[voiced]
    prom = np.std(stv) if len(stv) > 3 else 1.0      # self-normalized prominence
    peaks, _ = find_peaks(stv, prominence=max(prom, 1e-3))
    last = runs[-1] if runs and len(runs[-1]) > 3 else np.array([0.0, 0.0])
    tail = last[-max(2, len(last) // 5):]
    f0_feats = [
        float(np.mean(d1)) if len(d1) else 0.0,
        float(np.mean(np.abs(d1))) if len(d1) else 0.0,
        float(np.std(d1)) if len(d1) else 0.0,
        float(np.mean(np.abs(d2))) if len(d2) else 0.0,
        len(peaks) / dur,
        float((d1 > 0).mean() / max((d1 < 0).mean(), 1e-3)) if len(d1) else 1.0,
        float(np.polyfit(np.arange(len(tail)), tail, 1)[0] * 100),
    ]

    # --- energy envelope derivatives (10 ms grid) ---
    r = rms / max(rms.max(), 1e-9)
    dr = np.diff(r) * 100                            # 1/s
    rprom = np.std(r)
    rpeaks, _ = find_peaks(r, prominence=max(rprom, 1e-3))
    rise, fall = dr[dr > 0], -dr[dr < 0]
    rms_feats = [
        len(rpeaks) / dur,
        float(np.std(r) / max(np.mean(r), 1e-9)),
        float(np.mean(dr)),
        float(np.mean(rise) / max(np.mean(fall), 1e-9))
        if len(rise) and len(fall) else 1.0,
        float(np.mean(np.abs(np.diff(r, 2)))) * 100,
    ]

    # --- voicing rhythm ---
    switches = int(np.sum(np.abs(np.diff(voiced.astype(int)))))
    run_lens = [len(rr) * 0.01 for rr in runs]
    voi_feats = [float(np.mean(run_lens)) if run_lens else 0.0, switches / dur]

    return np.array(f0_feats + rms_feats + voi_feats)


def wcos(a, b, w):
    na = math.sqrt(float((w * a * a).sum()))
    nb = math.sqrt(float((w * b * b).sum()))
    return float((w * a * b).sum()) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def matrix(effects: dict, w: np.ndarray, title: str, knobs: list) -> dict:
    print(f"\n{title}")
    print("            " + "".join(f"{k[:6]:>8s}" for k in knobs))
    out = {}
    for a in knobs:
        print(f"{a:12s}" + "".join(
            f"{wcos(effects[a], effects[b], w):+8.2f}" for b in knobs))
        for b in knobs:
            if a < b:
                out[f"{a}|{b}"] = wcos(effects[a], effects[b], w)
    return out


def main() -> None:
    # dynamics effects (per unit slider) vs baseline
    dyn = {p: dynamics(p) for p in list(CALIB) + [BASELINE]}
    base_d = dyn[BASELINE]
    acc: dict = {}
    for p, (k, lvl) in CALIB.items():
        acc.setdefault(k, []).append((dyn[p] - base_d) / lvl)
    eff_d = {k: np.mean(v, axis=0) for k, v in acc.items()}
    # scale-normalize each dynamics dim by spread across knob effects
    E = np.stack(list(eff_d.values()))
    scale = np.std(E, axis=0)
    scale[scale < 1e-9] = 1e-9
    eff_dz = {k: v / scale for k, v in eff_d.items()}
    w_dyn = np.ones(len(DYN_NAMES))

    # static effects (from P4.7 cache, MSP z-space) — the failed basis, for combo
    feats = json.loads((P47 / "features.json").read_text())
    s = json.loads((HERE / "out/scaffold_msp.json").read_text())
    idx, wst = s["feature_index_in_111"], np.array(s["weights"])
    med, iqr = np.array(s["global_median"]), np.array(s["global_iqr"])

    def z(p):
        f = np.array(feats[p])
        return (f[idx] - med) / iqr

    base_s = z(BASELINE)
    acc_s: dict = {}
    for p, (k, lvl) in CALIB.items():
        acc_s.setdefault(k, []).append((z(p) - base_s) / lvl)
    eff_s = {k: np.mean(v, axis=0) for k, v in acc_s.items()}

    knobs = ["happy", "angry", "surprised", "afraid", "sad", "melancholic",
             "disgusted", "calm"]
    matrix(eff_s, wst, "STATIC basis (P4.8 — the wall):", knobs)
    matrix(eff_dz, w_dyn, "DYNAMICS basis (new — the fluctuation fingerprint):",
           knobs)

    # combined: static (its weights, scaled to unit total) + dynamics (equal,
    # scaled to unit total) — each basis contributes half the geometry
    eff_c = {k: np.concatenate([eff_s[k], eff_dz[k]]) for k in knobs}
    w_c = np.concatenate([wst / wst.sum(), w_dyn / w_dyn.sum()])
    pairs = matrix(eff_c, w_c, "COMBINED basis (static + dynamics):", knobs)

    hot_pairs = {p: v for p, v in pairs.items()
                 if all(any(h in p.split("|") for h in HOT)
                        for _ in [0]) and
                 set(p.split("|")).issubset(set(HOT))}
    print("\nhot-knob pairs in COMBINED basis:")
    for p, v in hot_pairs.items():
        print(f"  {p}: {v:+.3f}  {'< 0.80 ✓' if v < 0.80 else '>= 0.80 ✗'}")
    ok = all(v < 0.80 for v in hot_pairs.values())
    print(f"\nGATE 0 (pre-registered bar: all three hot pairs < 0.80): "
          f"{'PASS' if ok else 'FAIL'}")

    np.save(HERE / "out/gate0_dyn_effects.npy",
            np.stack([eff_dz[k] for k in knobs]))
    (HERE / "out/gate0_dynamics.json").write_text(json.dumps(
        {"dyn_names": DYN_NAMES, "knobs": knobs,
         "dyn_effects": {k: eff_dz[k].tolist() for k in knobs},
         "combined_hot_pairs": hot_pairs, "pass": bool(ok)}, indent=1))
    print("GATE0_" + ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
