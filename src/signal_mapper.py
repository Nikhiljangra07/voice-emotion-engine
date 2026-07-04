"""Signal mapper: Ekman-6 predictions → valence, arousal, expressionStrength.

Converts classifier output into the dimensional format that the companion backend's
backend (Layer A) consumes. The bridge, pressure engine, escalation
engine, mood engine — they all consume these three numbers.

Sources:
    Valence/Arousal/Dominance centroids:
        Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013).
        "Norms of valence, arousal, and dominance for 13,915 English lemmas."
        Behavior Research Methods, 45, 1191-1207.
        Mean ratings for the emotion nouns (1-9) normalized: valence → [-1,+1],
        arousal → [0,1], dominance → [-1,+1]. Interim LITERATURE PRIOR —
        replaced by data-derived centroids in Phase 2 / P2.3 (see
        TRAJECTORY_ENGINE.md Law 3).

    ExpressionStrength baselines:
        Global default: 288 RAVDESS neutral+calm speech files.
        Per-speaker: calibrated from the speaker's own neutral audio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.speaker_baseline import SpeakerBaseline

# ── Ekman-6 → dimensional (V, A, D) mapping ─────────────────────────
# Source: Warriner et al. (2013), authoritative ratings CSV (crr.ugent.be /
# JULIELab XANEW mirror). Mean ratings for the emotion NOUNS on the 1-9
# scale (columns V.Mean.Sum / A.Mean.Sum / D.Mean.Sum):
#
#     emotion    V(1-9)  A(1-9)  D(1-9)
#     anger       2.50    5.93    5.14
#     disgust     3.32    5.00    4.84
#     fear        2.93    6.14    3.32
#     joy         8.21    5.55    7.00
#     sadness     2.40    2.81    3.84
#     surprise    7.44    6.57    5.17
#
# Normalization (bipolar valence/dominance, unipolar arousal):
#   valence   = (V - 5) / 4   -> [-1, +1]
#   arousal   = (A - 1) / 8   -> [ 0,  1]
#   dominance = (D - 5) / 4   -> [-1, +1]   (>0 dominant, <0 submissive)
#
# Note: anger (D=+0.035) sits above fear (D=-0.42) on dominance — the
# key separation PAD/Mehrabian predict (anger=approach, fear=submit).
#
# THESE ARE AN INTERIM LITERATURE PRIOR. Per TRAJECTORY_ENGINE.md Law 3,
# production centroids will be computed from a dimensionally-annotated
# dataset (P2.3) and replace this table. The previous 2-D values were
# unsourced (did not match Warriner for any word form); corrected and
# extended to 3-D from the verified CSV on 2026-06-23.

_EKMAN_DIMENSIONS: dict[str, dict[str, float]] = {
    "anger":    {"valence": -0.6250, "arousal": 0.6163, "dominance":  0.0350},
    "disgust":  {"valence": -0.4200, "arousal": 0.5000, "dominance": -0.0400},
    "fear":     {"valence": -0.5175, "arousal": 0.6425, "dominance": -0.4200},
    "joy":      {"valence":  0.8025, "arousal": 0.5688, "dominance":  0.5000},
    "sadness":  {"valence": -0.6500, "arousal": 0.2263, "dominance": -0.2900},
    "surprise": {"valence":  0.6100, "arousal": 0.6963, "dominance":  0.0425},
}

# ── Global ExpressionStrength baselines ──────────────────────────────
# Source: 288 RAVDESS neutral+calm files (all 24 actors).
# Used as fallback when no per-speaker baseline is provided.

_GLOBAL_BASELINES: dict[str, tuple[float, float]] = {
    "jitter":         (0.0166, 0.0047),
    "shimmer":        (1.1545, 0.1972),
    "hnr":            (6.9702, 2.2798),
    "pitch_variance": (0.0908, 0.0449),
}

# ExpressionStrength weights — data-driven from RF feature importance.
# Source: RandomForest trained to separate 288 RAVDESS neutral+calm files
# from 1,152 emotional files using only these 4 features. RF accuracy: 100%.
# Jitter is the strongest separator (1.49x higher in emotional speech).
_ES_WEIGHTS: dict[str, float] = {
    "jitter":         0.35,
    "hnr":            0.23,
    "pitch_variance": 0.21,
    "shimmer":        0.21,
}

# Minimum classifier confidence for trusting the prediction.
# Below this, valence/arousal fall back to a neutral blend.
_CONFIDENCE_THRESHOLD: float = 0.30

# Neutral blend: equal-weight average of all Ekman-6 centroids.
_NEUTRAL_VALENCE: float = float(np.mean(
    [d["valence"] for d in _EKMAN_DIMENSIONS.values()]
))
_NEUTRAL_AROUSAL: float = float(np.mean(
    [d["arousal"] for d in _EKMAN_DIMENSIONS.values()]
))
_NEUTRAL_DOMINANCE: float = float(np.mean(
    [d["dominance"] for d in _EKMAN_DIMENSIONS.values()]
))


def map_signal(
    prediction: dict[str, Any],
    speaker_baseline: SpeakerBaseline | None = None,
) -> dict[str, Any]:
    """Map a classifier prediction to the Layer A signal format.

    Args:
        prediction: Dict from Predictor.predict() with keys:
            emotion, confidence, ekman6_weights, features.
        speaker_baseline: Optional per-speaker baseline. If provided,
            expressionStrength is measured as deviation from THIS
            speaker's neutral voice. If None, uses global RAVDESS
            baseline (population average).

    Returns:
        Dict with: valence, arousal, dominance, expressionStrength,
        confidence, emotion, ekman6_weights, features.

        valence/arousal/dominance form the (V,A,D) point in PAD space
        (Phase 2). dominance is bipolar [-1,+1]; >0 dominant, <0 submissive.
    """
    weights = prediction["ekman6_weights"]
    features = prediction["features"]
    confidence = prediction["confidence"]
    low_confidence = confidence < _CONFIDENCE_THRESHOLD

    # ── Valence, Arousal & Dominance (the (V,A,D) point) ──────────
    if low_confidence:
        # Classifier is guessing — use neutral blend instead of
        # amplifying an uncertain prediction into dimensional space.
        valence = _NEUTRAL_VALENCE
        arousal = _NEUTRAL_AROUSAL
        dominance = _NEUTRAL_DOMINANCE
    else:
        # Probability-weighted blend of Ekman-6 centroids.
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        total_weight = 0.0
        for emotion, prob in weights.items():
            if emotion in _EKMAN_DIMENSIONS:
                dims = _EKMAN_DIMENSIONS[emotion]
                valence += prob * dims["valence"]
                arousal += prob * dims["arousal"]
                dominance += prob * dims["dominance"]
                total_weight += prob
        if total_weight > 0:
            valence /= total_weight
            arousal /= total_weight
            dominance /= total_weight

    # ── Expression Strength ──────────────────────────────────────
    if speaker_baseline is not None:
        expression_strength = _compute_es_per_speaker(
            features, speaker_baseline
        )
    else:
        expression_strength = _compute_es_global(features)

    result: dict[str, Any] = {
        "valence": round(float(valence), 4),
        "arousal": round(float(arousal), 4),
        "dominance": round(float(dominance), 4),
        "expressionStrength": round(expression_strength, 4),
        "confidence": confidence,
        "emotion": prediction["emotion"],
        "ekman6_weights": prediction["ekman6_weights"],
        "features": prediction["features"],
    }

    if low_confidence:
        result["low_confidence"] = True

    return result


def _compute_es_global(features: dict[str, Any]) -> float:
    """ExpressionStrength using global RAVDESS baselines."""
    es = 0.0
    for feat_key, (baseline_mean, baseline_std) in _GLOBAL_BASELINES.items():
        feat_val = features.get(feat_key, baseline_mean)
        weight = _ES_WEIGHTS[feat_key]
        deviation = abs(feat_val - baseline_mean) / (2.0 * baseline_std + 1e-9)
        es += weight * min(deviation, 1.0)
    return float(np.clip(es, 0.0, 1.0))


def _compute_es_per_speaker(
    features: dict[str, Any],
    baseline: SpeakerBaseline,
) -> float:
    """ExpressionStrength using per-speaker baseline.

    Deviation is measured from the speaker's own neutral values.
    The global std is still used for scaling (how much deviation
    counts as "expressive") since we can't compute std from a
    single calibration sample.
    """
    speaker_vals = baseline.baselines
    es = 0.0
    for feat_key, (_, global_std) in _GLOBAL_BASELINES.items():
        feat_val = features.get(feat_key, speaker_vals[feat_key])
        speaker_mean = speaker_vals[feat_key]
        weight = _ES_WEIGHTS[feat_key]
        deviation = abs(feat_val - speaker_mean) / (2.0 * global_std + 1e-9)
        es += weight * min(deviation, 1.0)
    return float(np.clip(es, 0.0, 1.0))
