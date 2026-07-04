"""Speaker diarization front-end (Path A — classical, in-house).

"Who spoke when", from the speaker-discriminative subset of the features we
already extract — NO new heavy dependency, no deep learning. The insight
(confirmed by our own cross-dataset stability report): formant frequencies and
MFCC timbre means reflect the **vocal tract** (speaker identity) and are
emotion-stable, whereas pitch/jitter/loudness swing with emotion. So we cluster
windows on the identity features and deliberately exclude the emotion-volatile
ones — the opposite of the failed Phase-1 F0-threshold attempt.

This is a front-end LAYER: it does not touch the emotion engine. Each window's
speaker label is attached to that window's V/A/D, yielding one trajectory per
speaker. Quality is moderate (good when voices are acoustically distinct,
weaker for similar voices) — the honest "basic but real" step. Upgrade path:
swap this for pretrained neural speaker embeddings (pyannote) behind the same
interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Speaker-discriminative, emotion-STABLE features (vocal-tract identity + pitch
# range). Excludes jitter/shimmer/loudness/energy-envelope (emotion-volatile).
SPEAKER_FEATURE_NAMES: tuple[str, ...] = (
    "praat_f0_mean_hz", "praat_f0_std_hz",
    "praat_f1_mean_hz", "praat_f2_mean_hz", "praat_f3_mean_hz",
    "praat_f1_bandwidth_hz", "praat_f2_bandwidth_hz", "praat_f3_bandwidth_hz",
    "F1frequency_sma3nz_amean", "F2frequency_sma3nz_amean",
    "F3frequency_sma3nz_amean",
    "mfcc1V_sma3nz_amean", "mfcc2V_sma3nz_amean",
    "mfcc3V_sma3nz_amean", "mfcc4V_sma3nz_amean",
    "mfcc1_sma3_amean", "mfcc2_sma3_amean",
    "mfcc3_sma3_amean", "mfcc4_sma3_amean",
)


def speaker_feature_indices(feature_names: list[str]) -> list[int]:
    """Resolve the speaker-feature names to column indices in a feature vector."""
    idx = [feature_names.index(n) for n in SPEAKER_FEATURE_NAMES
           if n in feature_names]
    if len(idx) < 6:
        raise ValueError(
            f"Only {len(idx)} speaker features found; expected the formant/MFCC "
            "identity set. Feature schema may have changed."
        )
    return idx


class SpeakerDiarizer:
    """Cluster windows into speakers by their vocal-tract fingerprint.

    Args:
        n_speakers: fixed speaker count; if None, estimated by silhouette over
            [2, max_speakers].
        max_speakers: cap for auto-estimation.
        min_turn_s / hop_s: a speaker must hold the floor for at least
            ``min_turn_s`` — shorter label flips are smoothed away (people don't
            change every second). ``hop_s`` is the trajectory hop.
    """

    def __init__(
        self,
        n_speakers: int | None = None,
        max_speakers: int = 6,
        min_turn_s: float = 3.0,
        hop_s: float = 1.0,
    ) -> None:
        if n_speakers is not None and n_speakers < 1:
            raise ValueError("n_speakers must be >= 1.")
        self.n_speakers = n_speakers
        self.max_speakers = max_speakers
        self.min_turn_s = min_turn_s
        self.hop_s = hop_s
        self.estimated_k_: int | None = None

    def _estimate_k(self, Xs: np.ndarray) -> int:
        """Pick speaker count by best silhouette over [2, max_speakers]."""
        n = Xs.shape[0]
        upper = min(self.max_speakers, n - 1)
        if upper < 2:
            return 1
        best_k, best_score = 2, -1.0
        for k in range(2, upper + 1):
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
            try:
                score = silhouette_score(Xs, labels)
            except ValueError:
                continue
            if score > best_score:
                best_k, best_score = k, score
        # Weak separation → likely a single speaker.
        return best_k if best_score >= 0.15 else 1

    def _smooth(self, labels: np.ndarray) -> np.ndarray:
        """Enforce a minimum turn length: relabel runs shorter than min_turn
        with the previous run's speaker (people don't switch every second)."""
        min_win = max(1, int(round(self.min_turn_s / self.hop_s)))
        out = labels.copy()
        n = len(out)
        i = 0
        while i < n:
            j = i
            while j < n and out[j] == out[i]:
                j += 1
            run_len = j - i
            if run_len < min_win and i > 0:
                out[i:j] = out[i - 1]
            i = j
        return out

    def fit_predict(
        self, X: np.ndarray, feature_names: list[str]
    ) -> np.ndarray:
        """Assign a speaker label per window.

        Args:
            X: (n_windows, n_features) full feature matrix; NaN rows (failed
               windows) are excluded from clustering and labelled -1.
            feature_names: names matching X's columns.

        Returns:
            (n_windows,) int labels; -1 for windows that couldn't be analyzed.
        """
        cols = speaker_feature_indices(feature_names)
        Xsp = np.asarray(X, dtype=float)[:, cols]
        valid = ~np.any(np.isnan(Xsp), axis=1)
        labels = np.full(X.shape[0], -1, dtype=int)
        if valid.sum() < 2:
            labels[valid] = 0
            self.estimated_k_ = 1
            return labels

        Xs = StandardScaler().fit_transform(Xsp[valid])
        k = self.n_speakers if self.n_speakers is not None else self._estimate_k(Xs)
        self.estimated_k_ = k
        if k <= 1:
            labels[valid] = 0
        else:
            labels[valid] = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)

        # Smooth over the full timeline (valid windows in order).
        labels[valid] = self._smooth(labels[valid])
        return labels
