"""Tests for the torch-free live-ear support layer: NameSmoother,
flicker_rate, and the Affectogram renderer. These run in the classical
venv (no torch) — scripts/affectogram.py must stay importable there."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.affectogram import (NameSmoother, flicker_rate,  # noqa: E402
                                 render)


# ── NameSmoother ────────────────────────────────────────────────────

def test_smoother_majority_wins():
    sm = NameSmoother(k=5)
    for x in ["anger", "anger", "anger", "joy", "anger"]:
        out = sm.update(x)
    assert out == "anger"


def test_smoother_suppresses_single_flicker():
    """One deviant window inside a stable run must not change the name."""
    sm = NameSmoother(k=5)
    for x in ["anger", "anger", "anger"]:
        sm.update(x)
    assert sm.update("surprise") == "anger"
    assert sm.update("anger") == "anger"


def test_smoother_honors_genuine_change():
    """A sustained new emotion takes over once it holds the majority."""
    sm = NameSmoother(k=5)
    for x in ["anger"] * 5:
        sm.update(x)
    outs = [sm.update("joy") for _ in range(5)]
    assert outs[-1] == "joy"
    assert outs[0] == "anger"  # not instant — that's the point


def test_smoother_sticky_tiebreak():
    """On a tie, the current smoothed name wins (hysteresis)."""
    sm = NameSmoother(k=4)
    sm.update("anger"); sm.update("anger")
    sm.update("joy")
    out = sm.update("joy")  # buffer: anger,anger,joy,joy — tie
    assert out == "anger"


def test_smoother_gap_reset():
    """A long gated gap resets state — names don't cross scene breaks."""
    sm = NameSmoother(k=5, max_gap=3)
    for x in ["anger"] * 5:
        sm.update(x)
    for _ in range(4):  # > max_gap
        sm.gate()
    assert sm.state is None
    assert sm.update("joy") == "joy"


def test_smoother_short_gap_keeps_state():
    sm = NameSmoother(k=5, max_gap=3)
    for x in ["anger"] * 5:
        sm.update(x)
    for _ in range(3):  # == max_gap, not over
        sm.gate()
    assert sm.update("joy") == "anger"  # majority still anger


# ── flicker_rate ────────────────────────────────────────────────────

def test_flicker_zero_when_stable():
    assert flicker_rate(["joy"] * 10) == 0.0


def test_flicker_one_when_alternating():
    assert flicker_rate(["joy", "anger"] * 5) == 1.0


def test_flicker_ignores_gaps():
    """Only strictly adjacent named pairs count."""
    assert flicker_rate(["joy", None, "anger"]) == 0.0


def test_flicker_empty():
    assert flicker_rate([]) == 0.0
    assert flicker_rate([None, None]) == 0.0


# ── Affectogram renderer ────────────────────────────────────────────

def _rows(n=40, with_raw=True):
    rng = np.random.default_rng(20260818)
    rows = []
    for i in range(n):
        t = round(i * 1.5, 2)
        if i % 7 == 0:
            rows.append({"t": t, "V": None, "A": None, "D": None,
                         "emotion": "no-speech", "speech_prob": 0.1,
                         "ms": 9.0})
            continue
        r = {"t": t, "V": round(float(rng.uniform(-1, 1)), 3),
             "A": round(float(rng.uniform(0, 1)), 3),
             "D": round(float(rng.uniform(-1, 1)), 3),
             "emotion": "anger" if i % 2 else "joy",
             "ambiguous": bool(i % 3 == 0),
             "speech_prob": 0.9, "ms": 250.0}
        if with_raw:
            r["emotion_raw"] = "surprise" if i % 5 == 0 else r["emotion"]
        rows.append(r)
    return rows


def test_render_writes_png(tmp_path):
    out = tmp_path / "x_affectogram.png"
    render(_rows(), "x", out, params={"window": 3.0, "stride": 1.5,
                                      "speech_gate": 0.5, "smooth_k": 5})
    assert out.exists() and out.stat().st_size > 50_000


def test_render_pre_smoothing_format(tmp_path):
    """Old sessions (no emotion_raw key) must render via offline
    smoothing, not crash."""
    out = tmp_path / "old_affectogram.png"
    render(_rows(with_raw=False), "old", out)
    assert out.exists() and out.stat().st_size > 50_000


def test_render_all_gated(tmp_path):
    """A session where the gate suppressed everything still renders."""
    rows = [{"t": i * 1.5, "V": None, "A": None, "D": None,
             "emotion": "no-speech", "speech_prob": 0.0, "ms": 9.0}
            for i in range(10)]
    out = tmp_path / "silent_affectogram.png"
    render(rows, "silent", out)
    assert out.exists()


def test_render_missing_dominance(tmp_path):
    """Rows without D (hypothetical foreign source) must not crash."""
    rows = _rows()
    for r in rows:
        r.pop("D", None)
    out = tmp_path / "nod_affectogram.png"
    render(rows, "nod", out)
    assert out.exists()
