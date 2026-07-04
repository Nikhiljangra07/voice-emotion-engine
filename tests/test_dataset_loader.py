"""Tests for src.utils.dataset_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.dataset_loader import (
    EKMAN6_LABELS,
    AudioSample,
    load_all,
    load_crema_d,
    load_meld,
    load_ravdess,
    to_dataframe,
)


# ── RAVDESS ──────────────────────────────────────────────────────────

class TestLoadRavdess:
    def test_loads_files(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        assert len(samples) > 0

    def test_only_ekman6_labels(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        labels = {s.label for s in samples}
        assert labels.issubset(set(EKMAN6_LABELS))

    def test_no_neutral_or_calm(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        labels = {s.label for s in samples}
        assert "neutral" not in labels
        assert "calm" not in labels

    def test_has_surprise(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        labels = {s.label for s in samples}
        assert "surprise" in labels

    def test_expected_count(self) -> None:
        """RAVDESS has 192 files per emotion * 6 Ekman emotions = 1152."""
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        # 6 emotions * 192 = 1152 (neutral=96 + calm=192 excluded)
        assert len(samples) == 1152

    def test_dataset_field(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        assert all(s.dataset == "ravdess" for s in samples)

    def test_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ravdess("/nonexistent/path")


# ── CREMA-D ──────────────────────────────────────────────────────────

class TestLoadCremaD:
    def test_loads_files(self) -> None:
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_crema_d()
        assert len(samples) > 0

    def test_only_ekman6_labels(self) -> None:
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_crema_d()
        labels = {s.label for s in samples}
        assert labels.issubset(set(EKMAN6_LABELS))

    def test_no_neutral(self) -> None:
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_crema_d()
        labels = {s.label for s in samples}
        assert "neutral" not in labels

    def test_no_surprise(self) -> None:
        """CREMA-D does not have surprise emotion."""
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_crema_d()
        labels = {s.label for s in samples}
        assert "surprise" not in labels

    def test_dataset_field(self) -> None:
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_crema_d()
        assert all(s.dataset == "crema_d" for s in samples)

    def test_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_crema_d("/nonexistent/path")


# ── MELD ──────────────────────────────────────────────────────────────

class TestLoadMeld:
    def test_loads_files(self) -> None:
        if not Path("data/meld/audio").exists():
            pytest.skip("MELD not downloaded")
        samples = load_meld()
        assert len(samples) > 0

    def test_only_ekman6_labels(self) -> None:
        if not Path("data/meld/audio").exists():
            pytest.skip("MELD not downloaded")
        samples = load_meld()
        labels = {s.label for s in samples}
        assert labels.issubset(set(EKMAN6_LABELS))

    def test_no_neutral(self) -> None:
        if not Path("data/meld/audio").exists():
            pytest.skip("MELD not downloaded")
        samples = load_meld()
        labels = {s.label for s in samples}
        assert "neutral" not in labels

    def test_has_all_ekman6(self) -> None:
        if not Path("data/meld/audio").exists():
            pytest.skip("MELD not downloaded")
        samples = load_meld()
        labels = {s.label for s in samples}
        assert labels == set(EKMAN6_LABELS)

    def test_dataset_field(self) -> None:
        if not Path("data/meld/audio").exists():
            pytest.skip("MELD not downloaded")
        samples = load_meld()
        assert all(s.dataset == "meld" for s in samples)

    def test_paths_exist(self) -> None:
        if not Path("data/meld/audio").exists():
            pytest.skip("MELD not downloaded")
        samples = load_meld()
        for s in samples[:50]:
            assert Path(s.path).exists(), f"Missing: {s.path}"

    def test_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_meld("/nonexistent/path")


# ── Combined ─────────────────────────────────────────────────────────

class TestLoadAll:
    def test_combines_both(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_all()
        datasets = {s.dataset for s in samples}
        assert "ravdess" in datasets
        assert "crema_d" in datasets

    def test_all_paths_exist(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        if not Path("data/crema_d/audios").exists():
            pytest.skip("CREMA-D not downloaded")
        samples = load_all()
        for s in samples[:50]:  # spot check first 50
            assert Path(s.path).exists(), f"Missing: {s.path}"


# ── DataFrame ────────────────────────────────────────────────────────

class TestToDataframe:
    def test_columns(self) -> None:
        samples = [
            AudioSample("a.wav", "anger", "ravdess", "actor_01", "normal"),
            AudioSample("b.wav", "joy", "crema_d", "actor_02", "high"),
        ]
        df = to_dataframe(samples)
        assert list(df.columns) == ["path", "label", "dataset", "actor", "intensity"]
        assert len(df) == 2

    def test_from_real_data(self) -> None:
        if not Path("data/ravdess").exists():
            pytest.skip("RAVDESS not downloaded")
        samples = load_ravdess()
        df = to_dataframe(samples)
        assert len(df) == len(samples)
        assert df["label"].nunique() == 6
