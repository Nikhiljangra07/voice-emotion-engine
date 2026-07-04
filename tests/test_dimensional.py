"""Tests for the dimensional (PAD trajectory) engine scaffold — Phase 2 P2.1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dimensional import (
    DIMENSIONS,
    CentroidNamer,
    DimensionalRegressor,
    ccc,
    dimensional_report,
    pearson,
    rmse,
    vad_from_categorical,
)
from src.dimensional.loader import load_cmu_mosei, load_iemocap
from src.features.feature_vector import feature_names


# ────────────────────────── metrics ──────────────────────────
class TestMetrics:
    def test_ccc_perfect(self) -> None:
        y = np.array([0.1, 0.5, -0.3, 0.8, -0.6])
        assert ccc(y, y) == pytest.approx(1.0)

    def test_ccc_shift_penalised(self) -> None:
        y = np.array([0.1, 0.5, -0.3, 0.8, -0.6])
        # Perfectly correlated but shifted -> CCC < Pearson(=1).
        assert ccc(y, y + 0.5) < 1.0
        assert pearson(y, y + 0.5) == pytest.approx(1.0)

    def test_ccc_constant_is_zero(self) -> None:
        y = np.array([0.2, 0.4, 0.6])
        const = np.array([0.5, 0.5, 0.5])
        assert ccc(const, y) == 0.0  # degenerate denominator handled

    def test_rmse(self) -> None:
        assert rmse(np.array([0.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(
            np.sqrt(2.0)
        )

    def test_dimensional_report_shape(self) -> None:
        rng = np.random.default_rng(0)
        Yt = rng.normal(size=(50, 3))
        Yp = Yt + rng.normal(scale=0.1, size=(50, 3))
        rep = dimensional_report(Yt, Yp)
        for dim in DIMENSIONS:
            assert "ccc" in rep[dim] and "rmse" in rep[dim]
        assert "mean" in rep and "ccc" in rep["mean"]
        assert rep["valence"]["ccc"] > 0.8  # near-perfect predictions


# ───────────────────── DimensionalRegressor ──────────────────────
def _synthetic_linear(n: int = 140, seed: int = 1):
    """X (n,111) and Y (n,3) where Y is a linear map of X (+ small noise)."""
    rng = np.random.default_rng(seed)
    n_feat = len(feature_names())
    X = rng.normal(size=(n, n_feat))
    W = rng.normal(scale=0.3, size=(n_feat, 3))
    Y = X @ W + rng.normal(scale=0.05, size=(n, 3))
    return X, Y


class TestDimensionalRegressor:
    def test_fit_predict_shape(self) -> None:
        X, Y = _synthetic_linear()
        reg = DimensionalRegressor(model="ridge").fit(X, Y)
        Yhat = reg.predict(X)
        assert Yhat.shape == (X.shape[0], 3)

    def test_ridge_learns_linear(self) -> None:
        X, Y = _synthetic_linear()
        split = 110
        reg = DimensionalRegressor(model="ridge").fit(X[:split], Y[:split])
        rep = reg.evaluate(X[split:], Y[split:])
        # Linear data -> ridge should recover all three dimensions well.
        assert rep["mean"]["ccc"] > 0.5

    def test_predict_point(self) -> None:
        X, Y = _synthetic_linear()
        reg = DimensionalRegressor(model="ridge").fit(X, Y)
        pt = reg.predict_point(X[0])
        assert set(pt.keys()) == set(DIMENSIONS)
        assert all(isinstance(v, float) for v in pt.values())

    @pytest.mark.parametrize("kind", ["svr", "rf", "ridge"])
    def test_all_model_kinds_fit(self, kind: str) -> None:
        X, Y = _synthetic_linear(n=60)
        reg = DimensionalRegressor(model=kind).fit(X, Y)
        assert reg.is_fitted
        assert reg.predict(X[:5]).shape == (5, 3)

    def test_bad_model_kind(self) -> None:
        with pytest.raises(ValueError):
            DimensionalRegressor(model="nn")

    def test_bad_target_shape(self) -> None:
        X, Y = _synthetic_linear(n=40)
        with pytest.raises(ValueError):
            DimensionalRegressor(model="ridge").fit(X, Y[:, :2])  # only 2 cols

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, Y = _synthetic_linear()
        reg = DimensionalRegressor(model="ridge").fit(X, Y)
        before = reg.predict(X[:10])
        reg.save(tmp_path / "dim")
        loaded = DimensionalRegressor.load(tmp_path / "dim")
        after = loaded.predict(X[:10])
        assert np.allclose(before, after)

    def test_calibration_matches_target_spread(self) -> None:
        # Calibrated train predictions should match the target mean/std closely
        # (that is exactly what the affine enforces).
        X, Y = _synthetic_linear()
        reg = DimensionalRegressor(model="ridge", calibrate=True).fit(X, Y)
        P = reg.predict(X)
        for i in range(3):
            assert P[:, i].std() == pytest.approx(Y[:, i].std(), rel=0.05)
            assert P[:, i].mean() == pytest.approx(Y[:, i].mean(), abs=0.05)

    def test_calibration_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, Y = _synthetic_linear()
        reg = DimensionalRegressor(model="rf", calibrate=True).fit(X, Y)
        before = reg.predict(X[:10])
        reg.save(tmp_path / "dimcal")
        loaded = DimensionalRegressor.load(tmp_path / "dimcal")
        assert loaded.calibrate is True
        assert np.allclose(before, loaded.predict(X[:10]))


# ───────────────────────── CentroidNamer ─────────────────────────
def _synthetic_clusters(seed: int = 2):
    rng = np.random.default_rng(seed)
    centres = {
        "anger": np.array([-0.6, 0.6, 0.0]),
        "joy": np.array([0.7, 0.6, 0.5]),
        "sadness": np.array([-0.6, 0.2, -0.3]),
    }
    points, labels = [], []
    for emo, c in centres.items():
        points.append(c + rng.normal(scale=0.05, size=(40, 3)))
        labels += [emo] * 40
    return np.vstack(points), labels, centres


class TestCentroidNamer:
    def test_nearest_cluster(self) -> None:
        pts, labels, centres = _synthetic_clusters()
        namer = CentroidNamer().fit(pts, labels)
        out = namer.predict(centres["joy"])
        assert out["emotion"] == "joy"

    def test_distribution_sums_to_one(self) -> None:
        pts, labels, _ = _synthetic_clusters()
        namer = CentroidNamer().fit(pts, labels)
        out = namer.predict(np.array([0.0, 0.4, 0.1]))
        dist = out["distribution"]
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert set(dist.keys()) == {"anger", "joy", "sadness"}

    def test_intensity_is_radius(self) -> None:
        pts, labels, _ = _synthetic_clusters()
        namer = CentroidNamer().fit(pts, labels)
        out = namer.predict(np.array([0.0, 0.0, 0.0]))
        assert out["intensity"] == pytest.approx(0.0)

    def test_ambiguous_midpoint(self) -> None:
        pts, labels, centres = _synthetic_clusters()
        namer = CentroidNamer().fit(pts, labels)
        # Midpoint between anger and joy, far from sadness -> ambiguous.
        mid = (centres["anger"] + centres["joy"]) / 2.0
        out = namer.predict(mid)
        assert out["ambiguous"] is True

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        pts, labels, centres = _synthetic_clusters()
        namer = CentroidNamer().fit(pts, labels)
        before = namer.predict(centres["anger"])["distribution"]
        namer.save(tmp_path / "namer")
        loaded = CentroidNamer.load(tmp_path / "namer")
        after = loaded.predict(centres["anger"])["distribution"]
        assert before == pytest.approx(after)


# ─────────────────────────── loader ──────────────────────────────
class TestLoader:
    def test_vad_from_categorical_values(self) -> None:
        from src.signal_mapper import _EKMAN_DIMENSIONS

        vad = vad_from_categorical(["anger", "joy"])
        assert vad.shape == (2, 3)
        assert vad[0, 0] == pytest.approx(_EKMAN_DIMENSIONS["anger"]["valence"])
        assert vad[1, 2] == pytest.approx(_EKMAN_DIMENSIONS["joy"]["dominance"])

    def test_vad_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError):
            vad_from_categorical(["ennui"])

    def test_stub_loaders_raise(self) -> None:
        with pytest.raises(NotImplementedError):
            load_iemocap("/nope")
        with pytest.raises(NotImplementedError):
            load_cmu_mosei("/nope")

    def test_msp_emotion_mapping(self) -> None:
        from src.dimensional import msp_emotion_to_ekman

        assert msp_emotion_to_ekman("H") == "joy"
        assert msp_emotion_to_ekman("Happiness") == "joy"
        assert msp_emotion_to_ekman("D") == "disgust"
        assert msp_emotion_to_ekman("A") == "anger"
        assert msp_emotion_to_ekman("C") is None  # contempt: not Ekman-6
        assert msp_emotion_to_ekman("Neutral") is None
        assert msp_emotion_to_ekman("X") is None  # no agreement

    def test_normalize_vad_msp(self) -> None:
        from src.dimensional import normalize_vad_msp

        # 1-7 SAM -> PAD plane. Midpoint 4 -> 0 for valence/dominance.
        assert normalize_vad_msp(4.0, 1.0, 4.0) == pytest.approx((0.0, 0.0, 0.0))
        assert normalize_vad_msp(7.0, 7.0, 7.0) == pytest.approx((1.0, 1.0, 1.0))
        assert normalize_vad_msp(1.0, 1.0, 1.0) == pytest.approx((-1.0, 0.0, -1.0))


class TestMSPLoader:
    def _write_corpus(self, root: Path) -> None:
        (root / "Labels").mkdir(parents=True)
        (root / "Audios").mkdir(parents=True)
        csv = root / "Labels" / "labels_consensus.csv"
        csv.write_text(
            "FileName,EmoClass,EmoAct,EmoVal,EmoDom,SpkrID,Gender,Split_Set\n"
            "MSP-PODCAST_0001_0001.wav,A,5.9,2.5,5.1,Spkr_1,Male,Train\n"
            "MSP-PODCAST_0001_0002.wav,H,5.5,7.0,7.0,Spkr_2,Female,Train\n"
            "MSP-PODCAST_0002_0001.wav,N,2.0,4.0,2.0,Spkr_3,Male,Development\n"
            "MSP-PODCAST_0002_0002.wav,C,3.0,3.0,4.0,Spkr_4,Female,Development\n"
            "MSP-PODCAST_0003_0001.wav,U,6.5,7.0,5.0,Spkr_5,Male,Test1\n"
            "MSP-PODCAST_xxxx_xxxx.wav,X,4.0,4.0,4.0,Unknown,Unknown,Test3\n"
        )

    def test_loads_and_skips_test3(self, tmp_path: Path) -> None:
        from src.dimensional import load_msp_podcast

        self._write_corpus(tmp_path)
        samples = load_msp_podcast(tmp_path)
        # 6 rows minus the Test3 row = 5
        assert len(samples) == 5
        assert all(s.dataset == "msp-podcast" for s in samples)
        # H mapped to joy, A to anger, N stays "neutral", C stays "contempt"
        emos = {s.path.split("/")[-1]: s.emotion for s in samples}
        assert emos["MSP-PODCAST_0001_0001.wav"] == "anger"
        assert emos["MSP-PODCAST_0001_0002.wav"] == "joy"
        assert emos["MSP-PODCAST_0002_0001.wav"] == "neutral"

    def test_split_filter(self, tmp_path: Path) -> None:
        from src.dimensional import load_msp_podcast

        self._write_corpus(tmp_path)
        dev = load_msp_podcast(tmp_path, split="dev")
        assert len(dev) == 2
        assert {s.split for s in dev} == {"Development"}

    def test_ekman6_only(self, tmp_path: Path) -> None:
        from src.dimensional import load_msp_podcast

        self._write_corpus(tmp_path)
        ek = load_msp_podcast(tmp_path, ekman6_only=True)
        # drops Neutral (N) and Contempt (C); keeps A, H, U
        kept = {s.emotion for s in ek}
        assert kept == {"anger", "joy", "surprise"}

    def test_vad_matrix_native_and_normalized(self, tmp_path: Path) -> None:
        from src.dimensional import load_msp_podcast, vad_matrix

        self._write_corpus(tmp_path)
        s = load_msp_podcast(tmp_path, split="train")
        native = vad_matrix(s, normalize=False)
        norm = vad_matrix(s, normalize=True)
        assert native.shape == (2, 3)
        assert norm.shape == (2, 3)
        # native values are 1-7 scale; normalized are within [-1, 1].
        assert native.max() > 1.0
        assert norm.min() >= -1.0 and norm.max() <= 1.0

    def test_missing_labels_raises(self, tmp_path: Path) -> None:
        from src.dimensional import load_msp_podcast

        with pytest.raises(FileNotFoundError):
            load_msp_podcast(tmp_path)  # no Labels/ folder


# ───────────── real-feature smoke test (skips if data absent) ──────────────
class TestSmokeOnRealFeatures:
    def test_functional_mapping_pipeline(self) -> None:
        feats = Path("out/features_ravdess.npy")
        labs = Path("out/labels_ravdess.npy")
        if not feats.exists() or not labs.exists():
            pytest.skip("Extracted RAVDESS features not available")

        X = np.load(feats)[:, : len(feature_names())]
        labels = list(np.load(labs, allow_pickle=True))
        # SMOKE ONLY: functional (circular) VAD targets from categorical labels.
        Y = vad_from_categorical(labels)

        split = int(0.8 * len(X))
        reg = DimensionalRegressor(model="ridge").fit(X[:split], Y[:split])
        rep = reg.evaluate(X[split:], Y[split:])
        # Wiring check only — just assert finite, in-range CCC. NOT a result.
        assert -1.0 <= rep["mean"]["ccc"] <= 1.0
