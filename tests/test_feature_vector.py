"""Tests for src.features.feature_vector (combined feature vector)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.features.feature_vector import (
    build_feature_vector,
    feature_names,
    to_array,
    validate_feature_vector,
)
from src.preprocessing import TARGET_SR


def _make_speech_wav(sr: int = 16_000, duration: float = 2.0) -> Path:
    """Create a temporary WAV with a speech-like signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    vibrato = 5.0 * np.sin(2 * np.pi * 5.5 * t)
    y = (0.4 * np.sin(2 * np.pi * (180 + vibrato) * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return Path(tmp.name)


class TestFeatureNames:
    def test_count_is_107(self) -> None:
        names = feature_names()
        assert len(names) == 111  # 88 openSMILE + 13 Praat + 10 prosody

    def test_no_duplicates(self) -> None:
        names = feature_names()
        assert len(names) == len(set(names))

    def test_groups_present(self) -> None:
        names = feature_names()
        # openSMILE
        assert any("F0semitone" in n for n in names)
        assert any("jitterLocal" in n for n in names)
        # Praat
        assert any("praat_" in n for n in names)
        # Prosody
        assert "speech_rate" in names
        assert "pause_ratio" in names


class TestBuildFeatureVector:
    def test_returns_104_features(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        assert len(features) == 113  # 111 classifier + 2 metadata

    def test_matches_feature_names(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        names = feature_names()
        from src.features.feature_vector import METADATA_KEYS
        assert set(features.keys()) == set(names) | set(METADATA_KEYS)

    def test_all_values_are_numeric(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        for name, val in features.items():
            assert isinstance(val, (int, float)), f"{name} is {type(val)}"

    def test_metadata_fields_present(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        assert "audio_duration_s" in features
        assert "snr_estimate_db" in features
        assert features["audio_duration_s"] > 0
        assert isinstance(features["snr_estimate_db"], float)

    def test_metadata_excluded_from_to_array(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        arr = to_array(features)
        assert arr.shape == (111,)  # metadata NOT included

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            build_feature_vector("/nonexistent.wav")


class TestToArray:
    def test_shape(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        arr = to_array(features)
        assert arr.shape == (111,)

    def test_dtype(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        arr = to_array(features)
        assert arr.dtype == np.float64

    def test_order_matches_names(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        arr = to_array(features)
        names = feature_names()
        for i, name in enumerate(names):
            assert arr[i] == float(features[name]), f"Mismatch at {name}"


class TestRealDatasets:
    def test_ravdess_full_vector(self) -> None:
        path = Path("data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
        if not path.exists():
            pytest.skip("RAVDESS not downloaded")
        features = build_feature_vector(path)
        assert len(features) == 113  # 111 classifier + 2 metadata
        arr = to_array(features)
        nan_count = int(np.sum(np.isnan(arr)))
        assert nan_count == 0, f"{nan_count} NaN values in feature vector"

    def test_crema_d_full_vector(self) -> None:
        import glob

        files = glob.glob("data/crema_d/audios/*.wav")
        if not files:
            pytest.skip("CREMA-D not downloaded")
        features = build_feature_vector(files[0])
        assert len(features) == 113  # 111 classifier + 2 metadata
        arr = to_array(features)
        nan_count = int(np.sum(np.isnan(arr)))
        assert nan_count == 0, f"{nan_count} NaN values in feature vector"


class TestValidateFeatureVector:
    def test_valid_vector_passes(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        warnings = validate_feature_vector(features)
        # May have range warnings on synthetic audio, but should not raise
        assert isinstance(warnings, list)

    def test_missing_feature_raises(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        del features["speech_rate"]
        with pytest.raises(RuntimeError, match="missing"):
            validate_feature_vector(features)

    def test_nan_feature_raises(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        features["speech_rate"] = float("nan")
        with pytest.raises(RuntimeError, match="NaN"):
            validate_feature_vector(features)

    def test_inf_feature_raises(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        features["speech_rate"] = float("inf")
        with pytest.raises(RuntimeError, match="Inf"):
            validate_feature_vector(features)

    def test_out_of_range_warns(self) -> None:
        wav = _make_speech_wav()
        features = build_feature_vector(wav)
        # Force an out-of-range value
        features["praat_f0_mean_hz"] = 900.0  # max is 800
        warnings = validate_feature_vector(features)
        assert any("praat_f0_mean_hz" in w for w in warnings)
