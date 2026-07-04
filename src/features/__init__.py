"""Feature extraction layer (Layer 1 — dataset-agnostic).

Public API:
    from src.features import build_feature_vector, feature_names, to_array
    from src.features import extract_opensmile, extract_voice_quality, extract_prosody
    from src.features import FeatureNormalizer
"""

from src.features.feature_vector import (
    build_feature_vector,
    build_feature_vector_from_array,
    feature_names,
    to_array,
)
from src.features.normalize import FeatureNormalizer
from src.features.opensmile_extractor import (
    extract_features as extract_opensmile,
    extract_features_from_array as extract_opensmile_from_array,
)
from src.features.prosody import (
    extract_prosody,
    extract_prosody_from_array,
)
from src.features.voice_quality import (
    extract_voice_quality,
    extract_voice_quality_from_array,
)

__all__ = [
    # Combined (primary entry points)
    "build_feature_vector",
    "build_feature_vector_from_array",
    "feature_names",
    "to_array",
    # Normalization
    "FeatureNormalizer",
    # Individual extractors
    "extract_opensmile",
    "extract_opensmile_from_array",
    "extract_voice_quality",
    "extract_voice_quality_from_array",
    "extract_prosody",
    "extract_prosody_from_array",
]
