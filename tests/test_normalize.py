"""Tests for src.features.normalize."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.features.normalize import FeatureNormalizer


# ── helpers ──────────────────────────────────────────────────────────

def _make_data(
    n_samples: int = 50,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Create synthetic feature matrix with known properties."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) * np.arange(1, n_features + 1)
    names = [f"feat_{i}" for i in range(n_features)]
    return X, names


def _make_data_with_issues(
    n_samples: int = 50,
) -> tuple[np.ndarray, list[str]]:
    """Create data with zero-variance and correlated columns."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, 6)

    # feat_3: zero variance (constant)
    X[:, 3] = 5.0

    # feat_4: perfect copy of feat_0 (r=1.0)
    X[:, 4] = X[:, 0]

    # feat_5: near-perfect negative correlation with feat_1
    X[:, 5] = -X[:, 1] + rng.randn(n_samples) * 0.01

    names = [f"feat_{i}" for i in range(6)]
    return X, names


# ── fit / transform ─────────────────────────────────────────────────

class TestFitTransform:
    def test_output_is_standardized(self) -> None:
        X, names = _make_data()
        norm = FeatureNormalizer(var_threshold=1e-10, corr_threshold=1.0)
        X_out = norm.fit_transform(X, names)
        # Mean ~0, std ~1 per column
        assert np.allclose(np.mean(X_out, axis=0), 0.0, atol=1e-10)
        assert np.allclose(np.std(X_out, axis=0), 1.0, atol=1e-10)

    def test_removes_zero_variance(self) -> None:
        X, names = _make_data_with_issues()
        norm = FeatureNormalizer(var_threshold=1e-7, corr_threshold=1.0)
        norm.fit(X, names)
        assert "feat_3" not in norm.kept_names
        assert "feat_3" in norm._removed_low_var

    def test_removes_correlated(self) -> None:
        X, names = _make_data_with_issues()
        norm = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
        norm.fit(X, names)
        # feat_4 is a copy of feat_0 — one should be removed
        both_kept = "feat_0" in norm.kept_names and "feat_4" in norm.kept_names
        assert not both_kept, "Perfectly correlated pair should lose one"

    def test_n_features_reduced(self) -> None:
        X, names = _make_data_with_issues()
        norm = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
        X_out = norm.fit_transform(X, names)
        assert X_out.shape[1] < X.shape[1]

    def test_transform_shape(self) -> None:
        X, names = _make_data(n_samples=50)
        norm = FeatureNormalizer()
        X_train = norm.fit_transform(X[:40], names)
        X_test = norm.transform(X[40:])
        assert X_test.shape == (10, X_train.shape[1])

    def test_not_fitted_raises(self) -> None:
        norm = FeatureNormalizer()
        X, _ = _make_data()
        with pytest.raises(RuntimeError):
            norm.transform(X)

    def test_wrong_columns_raises(self) -> None:
        X, names = _make_data(n_features=10)
        norm = FeatureNormalizer()
        norm.fit(X, names)
        X_bad = np.zeros((5, 7))
        with pytest.raises(ValueError):
            norm.transform(X_bad)

    def test_mismatched_names_raises(self) -> None:
        X, _ = _make_data(n_features=5)
        norm = FeatureNormalizer()
        with pytest.raises(ValueError):
            norm.fit(X, ["a", "b", "c"])  # 3 names for 5 columns


# ── save / load ──────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self) -> None:
        X, names = _make_data_with_issues()
        norm = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
        X_out = norm.fit_transform(X, names)

        with tempfile.TemporaryDirectory() as tmp:
            norm.save(tmp)
            loaded = FeatureNormalizer.load(tmp)

        X_out_loaded = loaded.transform(X)
        np.testing.assert_array_almost_equal(X_out, X_out_loaded)
        assert loaded.kept_names == norm.kept_names
        assert loaded.n_features_in == norm.n_features_in
        assert loaded.n_features_out == norm.n_features_out

    def test_save_not_fitted_raises(self) -> None:
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError):
            norm.save("/tmp/should_not_exist")


# ── summary ──────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_content(self) -> None:
        X, names = _make_data_with_issues()
        norm = FeatureNormalizer(var_threshold=1e-7, corr_threshold=0.95)
        norm.fit(X, names)
        summary = norm.summary()
        assert "Input features:" in summary
        assert "Output features:" in summary
        assert "feat_3" in summary  # zero-var removal


# ── integration with real data ───────────────────────────────────────

class TestRealData:
    def test_on_ravdess_20(self) -> None:
        import glob

        from src.features.feature_vector import (
            build_feature_vector,
            feature_names,
            to_array,
        )

        files = sorted(glob.glob("data/ravdess/Actor_01/03-01-*.wav"))[:20]
        if not files:
            pytest.skip("RAVDESS not downloaded")

        names = feature_names()
        X = np.array([to_array(build_feature_vector(f)) for f in files])

        norm = FeatureNormalizer()
        X_out = norm.fit_transform(X, names)

        assert X_out.shape[0] == 20
        assert X_out.shape[1] <= 104
        assert X_out.shape[1] > 50  # shouldn't drop too many
        assert np.sum(np.isnan(X_out)) == 0
