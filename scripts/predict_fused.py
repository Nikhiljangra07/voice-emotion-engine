"""Integrated FUSED prediction on real clips (main pipeline orchestrates SSL).

End-to-end demo of the fused engine in the main pipeline: for each clip it gets
classical features (main venv) + WavLM embedding (subprocess → isolated env),
fuses them, predicts V/A/D with the fused model, and names the emotion. The heavy
SSL dependency stays quarantined; the main pipeline just calls it.

Usage: python -m scripts.predict_fused --inputs own_voice/001.wav own_voice/008.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.dimensional.loader import normalize_vad_msp
from src.dimensional.namer import CentroidNamer
from src.dimensional.regressors import DimensionalRegressor
from src.features.feature_vector import build_feature_vector, to_array

DIAR_PY = ".venv_diar/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--regressor", default="models/dim_fused_msp")
    ap.add_argument("--namer", default="models/namer_msp_final")
    args = ap.parse_args()

    # 1. Classical features (main venv).
    print("Extracting classical features...")
    classical = np.vstack([to_array(build_feature_vector(f)) for f in args.inputs])

    # 2. SSL embeddings (subprocess → isolated env).
    print("Extracting WavLM embeddings (isolated env)...")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        emb_path = tf.name
    subprocess.run([DIAR_PY, "-m", "scripts.embed_files",
                    "--inputs", *args.inputs, "--out", emb_path], check=True)
    ssl = np.load(emb_path)

    # 3. Fuse → predict → name.
    fused = np.hstack([classical, ssl])
    reg = DimensionalRegressor.load(args.regressor)
    namer = CentroidNamer.load(args.namer)
    vad = np.clip(reg.predict(fused), 1.0, 7.0)
    pad = np.array([normalize_vad_msp(*r) for r in vad])

    print(f"\n{'clip':<28}{'V':>6}{'A':>6}{'D':>6}  emotion (ambiguous?)")
    for f, p in zip(args.inputs, pad):
        r = namer.predict(p)
        amb = " ~" if r["ambiguous"] else ""
        print(f"{Path(f).name:<28}{p[0]:>6.2f}{p[1]:>6.2f}{p[2]:>6.2f}  "
              f"{r['emotion']}{amb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
