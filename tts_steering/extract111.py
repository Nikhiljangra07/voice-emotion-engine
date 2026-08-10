"""Layer-1 feature bridge for P4.7 — runs in the ORIGINAL engine venv (`venv/`),
which carries opensmile + parselmouth. Called as a subprocess by abc_p47.py
(same isolation pattern as bridge.py: no shared imports across environments).

Usage:  venv/bin/python tts_steering/extract111.py <out.json> <wav> [<wav> ...]
Writes: {wav_path: [111 floats], ...}   — canonical to_array() ordering,
        the SAME ordering as out/features_msp_*.npy. Fails loud on any error.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features.feature_vector import build_feature_vector, to_array  # noqa: E402

out_path, wavs = sys.argv[1], sys.argv[2:]
result = {}
for w in wavs:
    result[w] = [float(x) for x in to_array(build_feature_vector(w))]
Path(out_path).write_text(json.dumps(result))
print(f"EXTRACT111_DONE {len(result)} clips")
