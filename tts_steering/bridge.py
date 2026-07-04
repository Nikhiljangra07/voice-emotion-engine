"""THE BRIDGE — the only connection between tts_steering and the Voice Emotion Engine.

Isolation law (from the charter): this project never imports engine code. The engine
is consumed exactly as an external user would consume it — its CLIs, as subprocesses,
returning JSON/text over stdout. If the engine changes shape, only this file changes.

Two engine endpoints are bridged:
  * wavlm_vad(clips)   -> V/A/D point + 8-class emotion distribution   (steering signal)
  * e2v_family(clips)  -> 6-family label + confidence via frozen emotion2vec (the JUDGE)

Anti-circularity: the steering loop optimizes against ONE backbone's output and is
scored by the OTHER (separate vector spaces; engine's anti-mix law). judge() returns
both so the caller can enforce that separation explicitly.

Self-test:  .venv_tts/bin/python tts_steering/bridge.py <clip.wav> [more.wav ...]
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PY = REPO_ROOT / ".venv_diar" / "bin" / "python"

_FAMILY_LINE = re.compile(
    r"^(?P<clip>.+\.wav): (?P<emotion>\w+)\s+conf=(?P<conf>\d+)%(?P<amb> \(ambiguous\))?"
)


class BridgeError(RuntimeError):
    """Engine call failed — surfaced loudly, never swallowed (engine law #6)."""


def _run(args: list[str], timeout: int = 600) -> str:
    proc = subprocess.run(
        [str(ENGINE_PY), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise BridgeError(
            f"engine call failed (exit {proc.returncode}): {args[:3]}...\n"
            f"stderr tail: {proc.stderr[-500:]}"
        )
    return proc.stdout


def wavlm_vad(clips: list[str | Path]) -> list[dict]:
    """Fine-tuned WavLM: clip -> V/A/D (PAD, V/D in [-1,1], A in [0,1]) + 8-class dist.

    This is the STEERING signal (distance-to-target in PAD space).
    """
    out = _run(["-m", "scripts.predict_wavlm_ft", "--json",
                "--inputs", *[str(c) for c in clips]])
    start, end = out.find("["), out.rfind("]")
    if start == -1 or end == -1:
        raise BridgeError(f"no JSON in predict_wavlm_ft output: {out[:200]!r}")
    rows = json.loads(out[start:end + 1])
    if len(rows) != len(clips):
        raise BridgeError(f"asked {len(clips)} clips, got {len(rows)} results")
    return rows


def e2v_family(clips: list[str | Path], k: int = 5) -> list[dict]:
    """Frozen emotion2vec retrieval: clip -> family + confidence + ambiguity.

    This is the JUDGE (independent vector space from the steering signal).
    """
    out = _run(["-m", "scripts.adaptors", "predict",
                "--backbone", "emotion2vec_plus_large", "--k", str(k),
                *[str(c) for c in clips]])
    results = []
    for line in out.splitlines():
        m = _FAMILY_LINE.match(line.strip())
        if m:
            results.append({
                "clip": Path(m["clip"]).name,
                "family": m["emotion"],
                "confidence": int(m["conf"]) / 100.0,
                "ambiguous": bool(m["amb"]),
            })
    if len(results) != len(clips):
        raise BridgeError(
            f"asked {len(clips)} clips, parsed {len(results)} family lines.\n"
            f"raw tail: {out[-400:]!r}")
    return results


def judge(clips: list[str | Path]) -> list[dict]:
    """Both signals per clip, merged. Caller enforces steer-vs-judge separation."""
    vad = wavlm_vad(clips)
    fam = e2v_family(clips)
    merged = []
    for v, f in zip(vad, fam):
        merged.append({
            "clip": v["clip"],
            "V": v["valence"], "A": v["arousal"], "D": v["dominance"],
            "intensity": v["intensity"],
            "wavlm_emotion": v["emotion"],
            "wavlm_distribution": v["distribution"],
            "judge_family": f["family"],
            "judge_confidence": f["confidence"],
            "judge_ambiguous": f["ambiguous"],
        })
    return merged


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: bridge.py clip.wav [clip2.wav ...]")
    for r in judge(sys.argv[1:]):
        print(f"{r['clip']:24s} V={r['V']:+.2f} A={r['A']:.2f} D={r['D']:+.2f} "
              f"| wavlm={r['wavlm_emotion']:9s} | judge(e2v)={r['judge_family']}"
              f"@{r['judge_confidence']:.0%}{' (amb)' if r['judge_ambiguous'] else ''}")
