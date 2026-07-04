"""Train the FUSED V/A/D engine (classical 111 + WavLM 768) for the main pipeline.

Consumes cached SSL embeddings (from scripts.extract_ssl, run in the isolated env)
+ our classical features, fuses them, and trains the dimensional regressor — all in
the MAIN venv (no torch here; embeddings arrive as .npy). Trains both classical-only
and fused on the SAME scaled rows so the lift is measured apples-to-apples, then
saves the fused model for the main pipeline.

Usage: python -m scripts.train_fused
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.dimensional.metrics import DIMENSIONS
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import feature_names

OUT = Path("out")
MODELS = Path("models")
SSL_NAMES = [f"wavlm_{i}" for i in range(768)]


def load(split: str):
    idx = np.load(OUT / f"ssl_{split}_idx.npy")
    Xc = np.load(OUT / f"features_msp_{split}.npy")[idx]
    E = np.load(OUT / f"ssl_{split}.npy")
    Y = np.load(OUT / f"targets_msp_{split}.npy")[idx]
    return Xc, E, Y


def evaluate(reg, Xte, Yte, label, lines):
    rep = reg.evaluate(Xte, Yte)
    cc = [rep[d]["ccc"] for d in DIMENSIONS]
    lines.append(f"  {label:<10} V {cc[0]:.3f}  A {cc[1]:.3f}  D {cc[2]:.3f}"
                 f"   mean {np.mean(cc):.3f}")
    return cc


def main() -> int:
    for f in ("ssl_train.npy", "ssl_train_idx.npy", "ssl_test1.npy", "ssl_test1_idx.npy"):
        if not (OUT / f).exists():
            print(f"Missing {f}. Run scripts.extract_ssl first (isolated env).")
            return 1

    Xc_tr, E_tr, Y_tr = load("train")
    Xc_te, E_te, Y_te = load("test1")
    Xf_tr, Xf_te = np.hstack([Xc_tr, E_tr]), np.hstack([Xc_te, E_te])
    cnames, fnames = feature_names(), feature_names() + SSL_NAMES
    print(f"train {Xc_tr.shape[0]} / test {Xc_te.shape[0]} | classical {len(cnames)} "
          f"feats, fused {len(fnames)} feats")

    lines = [f"FUSED V/A/D ENGINE — train {len(Y_tr)} / test1 {len(Y_te)} (held-out)",
             "Ridge + calibration; CCC per dim", ""]
    # Classical-only baseline on identical rows.
    reg_c = DimensionalRegressor(model="ridge", calibrate=True).fit(Xc_tr, Y_tr, cnames)
    evaluate(reg_c, Xc_te, Y_te, "classical", lines)
    # Fused.
    reg_f = DimensionalRegressor(model="ridge", calibrate=True).fit(Xf_tr, Y_tr, fnames)
    fcc = evaluate(reg_f, Xf_te, Y_te, "fused", lines)
    lines.append("")
    lines.append(f"Fused mean CCC {np.mean(fcc):.3f} → saved models/dim_fused_msp")

    reg_f.save(MODELS / "dim_fused_msp")
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "fused_engine_result.txt").write_text(text)
    print("\nSaved → out/fused_engine_result.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
