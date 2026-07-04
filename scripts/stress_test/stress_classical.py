"""Stress harness — CLASSICAL pipeline (src.pipeline.run), one file at a time.

Verdict per clip:
  PREDICTED     — pipeline returned a result; we then check it for NaN/garbage
  EXPLICIT_ERR  — pipeline raised a clear typed error (the CORRECT outcome for junk)
  CRASH         — unhandled/unclear exception (BAD)
"""
import json, math, sys, traceback
from pathlib import Path

STRESS = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd()))
from src.pipeline import run  # noqa: E402

def is_bad_number(x):
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))

def scan_nan(obj, path=""):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += scan_nan(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += scan_nan(v, f"{path}[{i}]")
    elif is_bad_number(obj):
        bad.append(path)
    return bad

rows = []
for f in sorted(STRESS.glob("*.wav")):
    rec = {"clip": f.name}
    try:
        r = run(str(f))
        nans = scan_nan(r)
        rec["verdict"] = "PREDICTED" + ("_WITH_NAN!" if nans else "")
        rec["emotion"] = r.get("emotion")
        rec["confidence"] = round(float(r.get("confidence", -1)), 3)
        rec["valence"] = round(float(r.get("valence", 0)), 3)
        rec["arousal"] = round(float(r.get("arousal", 0)), 3)
        rec["low_conf_flag"] = bool(r.get("low_confidence", False))
        rec["nan_fields"] = nans
    except Exception as e:
        et = type(e).__name__
        msg = str(e)[:140]
        # explicit, typed, informative errors are the DESIRED outcome for junk
        explicit = et in ("AudioError", "RuntimeError", "FileNotFoundError", "ValueError") and msg
        rec["verdict"] = "EXPLICIT_ERR" if explicit else "CRASH"
        rec["error_type"] = et
        rec["error_msg"] = msg
    rows.append(rec)
    v = rec["verdict"]
    extra = rec.get("emotion") or rec.get("error_type")
    print(f"{f.name:32s} {v:14s} {extra} "
          f"{('conf='+str(rec.get('confidence'))) if 'confidence' in rec else rec.get('error_msg','')[:80]}",
          flush=True)

out = Path(sys.argv[2])
out.write_text(json.dumps(rows, indent=1))
print(f"\nsaved -> {out}")
