"""Rigorous diarization eval: Path A vs Path B on controlled conversations.

For each constructed conversation (exact ground truth), measures:
  - k-detection: does AUTO speaker-count match the truth?
  - attribution accuracy: FORCED to the true count, what fraction of windows are
    assigned to the right speaker (best label permutation via Hungarian match)?
Attribution isolates embedding quality from the k-estimation step.

Path A runs in-process (classical features); Path B runs the isolated neural
diarizer as a subprocess. Both use the same 2s/1s window grid.

Usage: python -m scripts.eval_diarization
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.dimensional.diarization import SpeakerDiarizer
from src.dimensional.trajectory import TrajectoryEngine, window_bounds
from src.features.feature_vector import feature_names
from src.preprocessing import TARGET_SR, load_audio

CONVO_DIR = Path("own_voice/test_convos")
OUT = Path("out")
DIAR_PY = ".venv_diar/bin/python"
MIN_TURN = 1.5
WIN, HOP = 2.0, 1.0


def true_per_window(bounds, sr, segments):
    """True speaker per window = segment with max temporal overlap (None if none)."""
    out = []
    for s, e in bounds:
        ws, we = s / sr, e / sr
        best, best_ov = None, 0.0
        for seg in segments:
            ov = min(we, seg["end"]) - max(ws, seg["start"])
            if ov > best_ov:
                best_ov, best = ov, seg["speaker"]
        out.append(best)
    return out


def attribution_accuracy(pred, true):
    """Best-permutation match accuracy over windows with valid (pred, true)."""
    pairs = [(p, t) for p, t in zip(pred, true) if t is not None and p >= 0]
    if not pairs:
        return 0.0
    preds, trues = zip(*pairs)
    lp, lt = sorted(set(preds)), sorted(set(trues))
    pi = {l: i for i, l in enumerate(lp)}
    ti = {l: i for i, l in enumerate(lt)}
    C = np.zeros((len(lp), len(lt)))
    for p, t in pairs:
        C[pi[p], ti[t]] += 1
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / len(pairs))


def run_neural(wav: str, out_csv: Path, speakers: int | None) -> np.ndarray:
    cmd = [DIAR_PY, "-m", "scripts.diarize_neural", "--input", wav,
           "--out", str(out_csv), "--window", str(WIN), "--hop", str(HOP),
           "--min-turn", str(MIN_TURN)]
    if speakers is not None:
        cmd += ["--speakers", str(speakers)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    with open(out_csv) as f:
        return np.array([int(r["speaker"]) for r in csv.DictReader(f)])


def main() -> int:
    convos = sorted(CONVO_DIR.glob("*.json"))
    if not convos:
        print("No test conversations. Run scripts.build_test_conversations first.")
        return 1
    eng = TrajectoryEngine.from_saved("models/dim_svr_msp_final",
                                      "models/namer_msp_final",
                                      window_s=WIN, hop_s=HOP)
    names = feature_names()

    print(f"{'conversation':<26}{'true_k':>7}{'A auto':>7}{'B auto':>7}"
          f"{'A attr%':>9}{'B attr%':>9}")
    print("-" * 65)
    rows = []
    for jf in convos:
        truth = json.loads(jf.read_text())
        wav = str(jf.with_suffix(".wav"))
        true_k = truth["n_speakers"]

        y, sr = load_audio(wav, sr=TARGET_SR)
        bounds = window_bounds(len(y), sr, WIN, HOP)
        tpw = true_per_window(bounds, sr, truth["segments"])

        # Path A: features once, then auto + forced.
        _, _, X, _, _ = eng._extract_windows(wav)
        kA = SpeakerDiarizer(n_speakers=None, min_turn_s=MIN_TURN).fit_predict(X, names)
        a_auto = len(set(int(l) for l in kA if l >= 0))
        aA = SpeakerDiarizer(n_speakers=true_k, min_turn_s=MIN_TURN).fit_predict(X, names)
        attrA = 1.0 if true_k == 1 else attribution_accuracy(aA, tpw)

        # Path B: neural auto + forced.
        labB_auto = run_neural(wav, OUT / f"_eval_{truth['name']}_auto.csv", None)
        b_auto = len(set(int(l) for l in labB_auto))
        labB = run_neural(wav, OUT / f"_eval_{truth['name']}_k.csv", true_k)
        attrB = 1.0 if true_k == 1 else attribution_accuracy(labB, tpw)

        print(f"{truth['name']:<26}{true_k:>7}{a_auto:>7}{b_auto:>7}"
              f"{attrA*100:>8.0f}%{attrB*100:>8.0f}%")
        rows.append({"convo": truth["name"], "true_k": true_k,
                     "A_auto_k": a_auto, "B_auto_k": b_auto,
                     "A_attr": round(attrA, 3), "B_attr": round(attrB, 3)})

    # Aggregate (exclude 1-spk from attribution average — it's trivially 100%).
    multi = [r for r in rows if r["true_k"] > 1]
    kA_ok = sum(r["A_auto_k"] == r["true_k"] for r in rows)
    kB_ok = sum(r["B_auto_k"] == r["true_k"] for r in rows)
    print("-" * 65)
    print(f"k-detection correct:  Path A {kA_ok}/{len(rows)}   Path B {kB_ok}/{len(rows)}")
    print(f"mean attribution (multi-speaker):  "
          f"Path A {np.mean([r['A_attr'] for r in multi])*100:.0f}%   "
          f"Path B {np.mean([r['B_attr'] for r in multi])*100:.0f}%")

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "diarization_eval.csv", index=False)
    print(f"\nSaved → {OUT / 'diarization_eval.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
