"""P4.7 ARM C — the MSP equation scaffold ("training wheels").

Idea (user's, 2026-08-08): don't steer toward a bare V/A/D point — steer toward
the ACOUSTIC RECIPE of each emotion, fit from real certified human speech
(MSP-Podcast, 169k labeled clips). The scaffold is, per emotion family, an
equation over the engine's own Layer-1 features:

    score(clip, family) = weighted match between
        (z(clip) - z(neutral_reference))          <- what OUR clip changed
    and (mean_z(family) - mean_z(neutral))_MSP    <- how REAL emotion differs
                                                     from real neutral

Differencing against neutral on BOTH sides cancels channel/synthesizer offsets
to first order (MSP is podcast audio; our mouth is a clean TTS — absolute
feature values are not comparable, but emotion-induced *changes* are).

Laws respected:
- MSP audio/features are used ONLY to fit statistics (read-only). Nothing is
  redistributed, nothing trains the mouth. License-safe.
- No magic numbers: feature selection is Fisher-score ranked, weights are
  data-derived, exclusions are by *category* (text/timing/absolute-level
  features that cannot transfer from free conversation to one fixed sentence).
- Validated on the held-out MSP dev split BEFORE any steering use.

Run:  venv/bin/python tts_steering/scaffold_msp.py
Out:  tts_steering/out/scaffold_msp.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.features.feature_vector import feature_names  # noqa: E402

FAMILIES = ["neutral", "anger", "sadness", "joy", "surprise"]
TOP_K = 32  # kept features, ranked by Fisher discriminability

# Category exclusions — features that depend on WHAT is said / how long the
# utterance is / absolute recording level, none of which transfer from MSP's
# free conversation to our one fixed sentence. Substring match, documented.
EXCLUDE_SUBSTR = [
    "speech_rate", "pause_ratio", "pause_count", "mean_pause_s", "tempo_bpm",
    "voiced_fraction", "attack_time_s", "decay_time_s", "attack_slope",
    "sustain_ratio",                       # utterance-structure (our prosody)
    "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "VoicedSegmentLength",
    "UnvoicedSegmentLength",               # eGeMAPS timing
    "equivalentSoundLevel",                # absolute level = channel, not emotion
]


def load_split(split: str):
    X = np.load(ROOT / f"out/features_msp_{split}.npy")
    labels = []
    with open(ROOT / f"out/meta_msp_{split}.csv") as f:
        for r in csv.DictReader(f):
            labels.append(r["emotion"])
    assert len(labels) == X.shape[0], f"{split}: meta/matrix row mismatch"
    return X, np.array(labels)


def main() -> None:
    names = feature_names()
    keep_idx = [i for i, n in enumerate(names)
                if not any(s in n for s in EXCLUDE_SUBSTR)]
    kept = [names[i] for i in keep_idx]
    print(f"features: {len(names)} total -> {len(kept)} candidates "
          f"({len(names) - len(kept)} excluded by category)")

    X, y = load_split("train")
    mask = np.isin(y, FAMILIES)
    X, y = X[mask][:, keep_idx], y[mask]
    print(f"train rows in the 5 families: {X.shape[0]}")

    # Robust global scaler (median / IQR) — outlier-tolerant, podcast audio is messy
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr[iqr < 1e-9] = 1e-9
    Z = (X - med) / iqr

    # Fisher score: between-class variance of family means / mean within-class var
    means = {f: Z[y == f].mean(axis=0) for f in FAMILIES}
    variances = {f: Z[y == f].var(axis=0) for f in FAMILIES}
    grand = np.mean([means[f] for f in FAMILIES], axis=0)
    between = np.mean([(means[f] - grand) ** 2 for f in FAMILIES], axis=0)
    within = np.mean([variances[f] for f in FAMILIES], axis=0)
    fisher = between / (within + 1e-9)

    order = np.argsort(fisher)[::-1][:TOP_K]
    sel_names = [kept[i] for i in order]
    weights = fisher[order] / fisher[order].sum()
    print(f"\ntop 12 discriminative features (of {TOP_K} kept):")
    for i in range(12):
        print(f"  {sel_names[i]:48s} fisher={fisher[order[i]]:.3f} "
              f"w={weights[i]:.3f}")

    # The scaffold: per-family template + direction-from-neutral, selected dims
    templates = {f: means[f][order] for f in FAMILIES}
    directions = {f: (means[f] - means["neutral"])[order]
                  for f in FAMILIES if f != "neutral"}

    # ---- validation on held-out dev: nearest weighted template, 5-way ----
    Xd, yd = load_split("dev")
    dmask = np.isin(yd, FAMILIES)
    Zd = ((Xd[dmask][:, keep_idx] - med) / iqr)[:, order]
    yd = yd[dmask]
    dists = np.stack([np.sqrt(((Zd - templates[f]) ** 2 * weights).sum(axis=1))
                      for f in FAMILIES], axis=1)
    pred = np.array(FAMILIES)[dists.argmin(axis=1)]
    acc = float((pred == yd).mean())
    print(f"\nheld-out dev validation ({len(yd)} clips, 5-way, chance=20%):")
    print(f"  overall nearest-template accuracy: {acc:.1%}")
    recalls = {}
    for f in FAMILIES:
        r = float((pred[yd == f] == f).mean())
        recalls[f] = r
        print(f"  {f:9s} recall {r:.1%}   (n={int((yd == f).sum())})")

    out = {
        "provenance": {
            "source": "MSP-Podcast train split, statistics only (license-safe)",
            "n_train": int(X.shape[0]), "n_dev_validation": int(len(yd)),
            "families": FAMILIES, "top_k": TOP_K,
            "selection": "Fisher score over category-filtered features",
            "scaler": "robust median/IQR",
            "excluded_categories": EXCLUDE_SUBSTR,
        },
        "feature_names": sel_names,
        "feature_index_in_111": [int(keep_idx[i]) for i in order],
        "weights": weights.tolist(),
        "global_median": med[order].tolist(),
        "global_iqr": iqr[order].tolist(),
        "templates_z": {f: templates[f].tolist() for f in FAMILIES},
        "directions_from_neutral_z": {f: d.tolist()
                                      for f, d in directions.items()},
        "validation": {"dev_accuracy_5way": acc, "chance": 0.2,
                       "per_family_recall": recalls},
    }
    dest = HERE / "out/scaffold_msp.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nSCAFFOLD_SAVED {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
