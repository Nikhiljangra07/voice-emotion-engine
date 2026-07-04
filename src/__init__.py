"""Voice Emotion Engine — emotional signal extraction from voice.

Public API:
    from src import preprocess, build_feature_vector, feature_names
    from src import FeatureNormalizer
    from src import load_ravdess, load_crema_d, load_all, EKMAN6_LABELS
"""

from src.preprocessing import preprocess, load_audio, normalize, trim_silence, TARGET_SR
from src.features import (
    build_feature_vector,
    build_feature_vector_from_array,
    feature_names,
    to_array,
    FeatureNormalizer,
)
from src.utils import (
    EKMAN6_LABELS,
    AudioSample,
    load_all,
    load_crema_d,
    load_ravdess,
    to_dataframe,
)

__all__ = [
    # Preprocessing
    "preprocess",
    "load_audio",
    "normalize",
    "trim_silence",
    "TARGET_SR",
    # Features
    "build_feature_vector",
    "build_feature_vector_from_array",
    "feature_names",
    "to_array",
    "FeatureNormalizer",
    # Dataset loading
    "EKMAN6_LABELS",
    "AudioSample",
    "load_all",
    "load_crema_d",
    "load_ravdess",
    "to_dataframe",
]
