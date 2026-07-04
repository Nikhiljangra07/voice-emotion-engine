"""Dimensional dataset loading + functional VAD mapping.

Real dimensionally-annotated corpora (IEMOCAP, MSP-Podcast, CMU-MOSEI) are not
yet on disk — they are academic-gated and access is pending (see
SESSION_LOG_2026-06-22.md §7). These loaders define the interface P2.3 will
fill in, and raise ``NotImplementedError`` with guidance until data lands.

``vad_from_categorical`` is the **smoke-test path**: it maps existing
categorical labels to (V,A,D) via the signal_mapper centroids so the whole
Layer-2/Layer-4 pipeline can be wired and tested *today* with no new data.

    HONESTY NOTE: this mapping is CIRCULAR — training a regressor to predict
    label-derived coordinates and then checking they map back to labels
    validates the PLUMBING only, never the science. Never report results from
    it (TRAJECTORY_ENGINE.md Laws 6 & 9). Use it for wiring/smoke tests only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.dimensional.metrics import DIMENSIONS
from src.signal_mapper import _EKMAN_DIMENSIONS


@dataclass
class DimensionalSample:
    """One audio clip with its dimensional + categorical annotation.

    valence/arousal/dominance are stored in the dataset's NATIVE scale (for
    MSP-Podcast that is SAM 1-7). Use ``normalize_vad_msp`` to convert to the
    PAD-plane convention ([-1,1]/[0,1]/[-1,1]) for the namer/visualization.
    """

    path: str
    valence: float
    arousal: float
    dominance: float
    emotion: str
    dataset: str
    speaker: str | None = None
    split: str | None = None


# MSP-Podcast primary-emotion codes/words -> our Ekman-6 names.
# Single letters are the consensus codes (A/S/H/U/F/D); full words appear in
# some exports. Contempt (C), Neutral (N), Other (O), No-agreement (X) are NOT
# Ekman-6 -> mapped to None (kept for V/A/D regression, dropped for the namer).
_MSP_EKMAN: dict[str, str] = {
    "a": "anger", "anger": "anger", "angry": "anger",
    "s": "sadness", "sadness": "sadness", "sad": "sadness",
    "h": "joy", "happiness": "joy", "happy": "joy",
    "u": "surprise", "surprise": "surprise",
    "f": "fear", "fear": "fear",
    "d": "disgust", "disgust": "disgust",
}

# Full primary-emotion names for ALL MSP codes (for the display label on
# non-Ekman rows kept in the regression set). Ekman-6 rows use the Ekman name.
_MSP_FULL: dict[str, str] = {
    "a": "anger", "s": "sadness", "h": "happiness", "u": "surprise",
    "f": "fear", "d": "disgust", "c": "contempt", "n": "neutral",
    "o": "other", "x": "no_agreement",
}

_SPLIT_ALIASES: dict[str, set[str]] = {
    "train": {"train", "training"},
    "development": {"development", "dev", "validation"},
    "test1": {"test1", "test 1", "test_1"},
    "test2": {"test2", "test 2", "test_2"},
    "test3": {"test3", "test 3", "test_3"},
}


def _canonical_split(value: str) -> str:
    """Resolve a split label/alias to its canonical key (e.g. 'dev' -> 'development')."""
    v = str(value).strip().lower()
    for canon, aliases in _SPLIT_ALIASES.items():
        if v == canon or v in aliases:
            return canon
    return v


def msp_emotion_to_ekman(raw: str) -> str | None:
    """Map an MSP primary-emotion code/word to an Ekman-6 name, or None."""
    return _MSP_EKMAN.get(str(raw).strip().lower())


def normalize_vad_msp(
    valence: float, arousal: float, dominance: float
) -> tuple[float, float, float]:
    """MSP SAM (1-7) -> PAD-plane convention.

    valence  = (V - 4) / 3  -> [-1, +1]   (4 = neutral midpoint)
    arousal  = (A - 1) / 6  -> [ 0,  1]
    dominance= (D - 4) / 3  -> [-1, +1]
    """
    return ((valence - 4.0) / 3.0, (arousal - 1.0) / 6.0, (dominance - 4.0) / 3.0)


def vad_from_categorical(labels: list[str]) -> np.ndarray:
    """Map categorical Ekman labels to (V,A,D) via signal_mapper centroids.

    SMOKE TEST ONLY — circular, not a scientific result (see module docstring).

    Args:
        labels: list of Ekman-6 emotion names.

    Returns:
        (n, 3) array, columns = valence, arousal, dominance.

    Raises:
        ValueError: if a label is not a known Ekman-6 emotion.
    """
    rows = []
    for lab in labels:
        if lab not in _EKMAN_DIMENSIONS:
            raise ValueError(
                f"Unknown emotion '{lab}'. Known: {sorted(_EKMAN_DIMENSIONS)}."
            )
        dims = _EKMAN_DIMENSIONS[lab]
        rows.append([dims[d] for d in DIMENSIONS])
    return np.asarray(rows, dtype=float)


def _not_available(name: str, page: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{name} is not on disk yet (academic-gated, access pending — see "
        f"SESSION_LOG_2026-06-22.md §7). Once obtained, implement this loader "
        f"to return list[DimensionalSample] with native V/A/D. Source: {page}"
    )


def load_iemocap(root: str | Path) -> list[DimensionalSample]:
    """Load IEMOCAP (categorical + native V/A/D). STUB until data is granted."""
    raise _not_available("IEMOCAP", "https://sail.usc.edu/iemocap/")


def load_cmu_mosei(root: str | Path) -> list[DimensionalSample]:
    """Load CMU-MOSEI (sentiment=valence proxy + categories). STUB until downloaded.

    Note: CMU-MOSEI is OPEN (no academic gate) and provides valence + categories
    but NOT native arousal/dominance — fill those as NaN when implemented.
    """
    raise _not_available(
        "CMU-MOSEI", "http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/"
    )


def _find_col(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    """Find a column by case-insensitive alias match."""
    lower = {c.strip().lower(): c for c in columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_msp_podcast(
    root: str | Path,
    split: str | None = None,
    ekman6_only: bool = False,
    require_vad: bool = True,
) -> list[DimensionalSample]:
    """Load MSP-Podcast from ``Labels/labels_consensus.csv`` + ``Audios/``.

    Native V/A/D are SAM 1-7 (consensus = mean of >=5 annotators). Test3 has
    masked filenames / withheld labels and is skipped automatically.

    Args:
        root: corpus folder containing ``Labels/`` and ``Audios/``.
        split: optional partition filter — "train", "development", "test1",
            "test2" (accepts "dev"). None = all available (excl. Test3).
        ekman6_only: keep only rows whose primary emotion maps to Ekman-6.
        require_vad: drop rows without valid numeric valence/arousal/dominance.

    Returns:
        list[DimensionalSample] (valence/arousal/dominance in native 1-7).

    Raises:
        FileNotFoundError: if the labels CSV isn't found under ``root``.
        ValueError: if expected V/A/D columns can't be located.
    """
    import math

    import pandas as pd

    root = Path(root)
    labels_csv = root / "Labels" / "labels_consensus.csv"
    audios = root / "Audios"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Expected MSP labels at {labels_csv}. Point `root` at the corpus "
            "folder that contains Labels/ and Audios/."
        )

    df = pd.read_csv(labels_csv)
    cols = list(df.columns)
    c_file = _find_col(cols, ("filename", "file_name", "file", "fileid"))
    c_val = _find_col(cols, ("emoval", "valence", "val", "v"))
    c_act = _find_col(cols, ("emoact", "arousal", "activation", "act", "a"))
    c_dom = _find_col(cols, ("emodom", "dominance", "dom", "d"))
    c_emo = _find_col(cols, ("emoclass", "emotion", "class", "emoclass_major"))
    c_spk = _find_col(cols, ("spkrid", "speakerid", "speaker", "spk_id"))
    c_split = _find_col(cols, ("split_set", "split", "set", "partition"))

    missing = [
        name for name, c in
        [("filename", c_file), ("valence", c_val),
         ("arousal", c_act), ("dominance", c_dom)]
        if c is None
    ]
    if missing:
        raise ValueError(
            f"labels_consensus.csv is missing expected columns {missing}. "
            f"Found columns: {cols}. Add the real names to _find_col aliases."
        )

    want_canon = _canonical_split(split) if split else None

    samples: list[DimensionalSample] = []
    for _, row in df.iterrows():
        # Skip Test3 (masked filenames / withheld labels) and apply split filter.
        row_canon = _canonical_split(row[c_split]) if c_split else ""
        if row_canon == "test3":
            continue
        if want_canon and row_canon != want_canon:
            continue

        try:
            v = float(row[c_val]); a = float(row[c_act]); d = float(row[c_dom])
        except (ValueError, TypeError):
            if require_vad:
                continue
            v = a = d = math.nan
        if require_vad and (math.isnan(v) or math.isnan(a) or math.isnan(d)):
            continue

        raw_emo = str(row[c_emo]).strip() if c_emo else ""
        emo = msp_emotion_to_ekman(raw_emo)
        if ekman6_only and emo is None:
            continue
        display = emo if emo else _MSP_FULL.get(raw_emo.lower(), raw_emo.lower())

        samples.append(DimensionalSample(
            path=str(audios / str(row[c_file])),
            valence=v, arousal=a, dominance=d,
            emotion=display,
            dataset="msp-podcast",
            speaker=str(row[c_spk]) if c_spk else None,
            split=str(row[c_split]) if c_split else None,
        ))
    return samples


def vad_matrix(
    samples: list[DimensionalSample], normalize: bool = False
) -> "np.ndarray":
    """Stack samples' (valence, arousal, dominance) into an (n, 3) array.

    Args:
        samples: list of DimensionalSample.
        normalize: if True, apply ``normalize_vad_msp`` (1-7 -> PAD plane).
    """
    rows = []
    for s in samples:
        if normalize:
            rows.append(list(normalize_vad_msp(s.valence, s.arousal, s.dominance)))
        else:
            rows.append([s.valence, s.arousal, s.dominance])
    return np.asarray(rows, dtype=float)
