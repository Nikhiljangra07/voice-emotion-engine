"""Dimensional PAD trajectory engine (Phase 2).

Layer 2 (three V/A/D regressors), Layer 4 (centroid namer -> distribution),
CCC metrics, and dimensional dataset loaders. See TRAJECTORY_ENGINE.md.
"""

from src.dimensional.loader import (
    DimensionalSample,
    load_cmu_mosei,
    load_iemocap,
    load_msp_podcast,
    msp_emotion_to_ekman,
    normalize_vad_msp,
    vad_from_categorical,
    vad_matrix,
)
from src.dimensional.metrics import (
    DIMENSIONS,
    ccc,
    dimensional_report,
    pearson,
    regression_report,
    rmse,
)
from src.dimensional.diarization import (
    SPEAKER_FEATURE_NAMES,
    SpeakerDiarizer,
    speaker_feature_indices,
)
from src.dimensional.namer import CentroidNamer
from src.dimensional.regressors import DimensionalRegressor
from src.dimensional.trajectory import (
    TrajectoryEngine,
    TrajectoryPoint,
    ema_smooth,
    trajectory_to_rows,
    window_bounds,
)

__all__ = [
    "DIMENSIONS",
    "ccc",
    "rmse",
    "pearson",
    "regression_report",
    "dimensional_report",
    "DimensionalRegressor",
    "CentroidNamer",
    "TrajectoryEngine",
    "TrajectoryPoint",
    "SpeakerDiarizer",
    "speaker_feature_indices",
    "SPEAKER_FEATURE_NAMES",
    "window_bounds",
    "ema_smooth",
    "trajectory_to_rows",
    "DimensionalSample",
    "vad_from_categorical",
    "load_iemocap",
    "load_cmu_mosei",
    "load_msp_podcast",
    "msp_emotion_to_ekman",
    "normalize_vad_msp",
    "vad_matrix",
]
