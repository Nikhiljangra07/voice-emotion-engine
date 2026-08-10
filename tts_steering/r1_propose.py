"""P4.7 ARM B — DeepSeek R1 as the mathematical proposer (via OpenRouter).

The model sees the slider spec, the target centroid, and the arm's OWN judged
history (never the other arms'), and returns the next emo_vector(s). Honest-
failure rule: if R1 cannot produce parseable vectors after 2 attempts, the
round is FORFEIT for arm B and logged as such — we never silently substitute
the deterministic rules, that would contaminate the comparison.

Budget guard: cumulative spend tracked in out/abc_p47/r1_spend.json,
hard stop at $2.50 (leaves margin in the $3.60 OpenRouter balance).
"""

import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND_FILE = HERE / "out/abc_p47/r1_spend.json"
MODEL = "deepseek/deepseek-r1-0528"
HARD_STOP_USD = 2.50
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]


def _key() -> str:
    for line in (HERE / ".keys.env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.partition("=")[2].strip()
    raise RuntimeError("OPENROUTER_API_KEY missing from .keys.env")


def _spent() -> float:
    return json.loads(SPEND_FILE.read_text())["usd"] if SPEND_FILE.exists() else 0.0


def _add_spend(usd: float) -> None:
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps({"usd": _spent() + usd}))


class R1CallError(RuntimeError):
    """Network / API failure — counts as a failed attempt, never crashes the run."""


def _call(prompt: str, max_tokens: int = 12000) -> tuple[str, float]:
    if _spent() >= HARD_STOP_USD:
        raise RuntimeError(f"R1 budget hard-stop reached (${_spent():.2f})")
    body = json.dumps({
        "model": MODEL, "max_tokens": max_tokens, "temperature": 0.0,
        "usage": {"include": True},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_key()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except Exception as e:                       # timeout, DNS, 5xx, bad JSON
        raise R1CallError(f"{type(e).__name__}: {e}") from e
    usage = resp.get("usage", {})
    usd = float(usage.get("cost", 0.0)) or (
        usage.get("prompt_tokens", 0) * 0.5e-6
        + usage.get("completion_tokens", 0) * 2.15e-6)
    _add_spend(usd)
    msg = resp["choices"][0]["message"]
    # R1 on OpenRouter splits chain-of-thought into `reasoning`; the final
    # answer may live in either field. Parse both, content last (wins ties).
    text = (msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
    return text, usd


def _parse_vectors(text: str, n: int) -> list[list[float]]:
    """Find JSON arrays of exactly 8 numbers; take the last n (post-reasoning)."""
    found = []
    for m in re.finditer(r"\[[\d\s.,+-eE]+\]", text):
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list) and len(arr) == 8 \
                and all(isinstance(x, (int, float)) for x in arr):
            v = [min(max(float(x), 0.0), 1.4) for x in arr]
            s = sum(v)
            if s > 1.5:
                v = [round(x * 1.5 / s, 2) for x in v]
            found.append([round(x, 2) for x in v])
    return found[-n:] if len(found) >= n else []


def propose(target: str, centroid: tuple, history: list[dict],
            n_vectors: int = 2) -> list[list[float]]:
    """history rows: {vector, V, A, D, distance, judge_family, judge_confidence}"""
    hist_txt = "\n".join(
        f"  vector={h['vector']} -> measured V={h['V']:+.2f} A={h['A']:.2f} "
        f"D={h['D']:+.2f}, distance={h['distance']:.3f}, "
        f"independent judge said: {h['judge_family']}@{h['judge_confidence']:.0%}"
        for h in history) or "  (none yet — this is the first round)"
    prompt = f"""You are steering a text-to-speech system's emotion control vector.

The control is 8 sliders, order [{', '.join(DIMS)}], each in [0, 1.4], sum <= 1.5.
The synthesized clip is measured by a speech-emotion model as a point (V, A, D):
valence in [-1,1], arousal in [0,1], dominance in [-1,1].

TARGET: make the clip land on the '{target}' centroid V={centroid[0]:+.2f} \
A={centroid[1]:.2f} D={centroid[2]:+.2f} (fit from 137k real human clips), AND be \
named '{target}' by an independent frozen judge.

History of this arm's own attempts (vector -> measurement):
{hist_txt}

Reason mathematically about the mapping from sliders to (V,A,D) using the history,
then propose the {n_vectors} most promising NEW vectors (not repeats).
End your answer with exactly {n_vectors} JSON arrays of 8 numbers, one per line,
nothing after them."""
    raw_dir = HERE / "out/abc_p47/r1_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2, 3):
        try:
            text, usd = _call(prompt)
        except R1CallError as e:
            print(f"    [R1] {target}: attempt {attempt} network fail ({e})")
            continue
        n_raw = len(list(raw_dir.glob(f"{target}_*.txt")))
        (raw_dir / f"{target}_{n_raw:02d}.txt").write_text(text)
        vecs = _parse_vectors(text, n_vectors)
        print(f"    [R1] {target}: attempt {attempt}, ${usd:.3f}, "
              f"parsed {len(vecs)} vectors (total spend ${_spent():.2f})")
        if vecs:
            return vecs
    print(f"    [R1] {target}: FORFEIT this round (3 attempts, no vectors)")
    return []
