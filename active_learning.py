# active_learning.py
"""
Lightweight active learning / auto-propagation for time-series labels.

Strategy:
- Featureize fixed-width windows around indices (mean, std, slope, energy, FFT bands).
- Keep a small incremental dataset per label.
- Train a fast KNN/LogReg model when we have enough data.
- After a new label is added, scan the series with a sliding window:
    - Predict probabilities
    - Auto-accept if confidence >= high_threshold
    - Otherwise return as suggestions (low/highlight)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# sklearn pieces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _window_features(x: np.ndarray) -> np.ndarray:
    """
    Compute a compact feature vector for a 1D window.
    Features: mean, std, slope (linreg), energy, fft magnitudes (first 6 bins, excluding DC).
    """
    if x.ndim != 1:
        x = x.ravel()
    n = len(x)
    if n < 3:
        return np.zeros(16, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x)) + 1e-9
    # slope via simple linear regression against [0..n-1]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    slope = float(np.dot(t, x - mean) / (np.dot(t, t) + 1e-9))
    energy = float(np.mean(x**2))
    # FFT (magnitude), drop DC, take first few bins
    mag = np.abs(np.fft.rfft(x - mean))
    mag = mag[1:7] if mag.size > 7 else mag[1:]
    # pad to 6
    if mag.size < 6:
        mag = np.pad(mag, (0, 6 - mag.size), mode='constant')
    return np.array([mean, std, slope, energy, *mag[:6]], dtype=float)


def _sliding_windows(arr: np.ndarray, win: int, step: int) -> List[Tuple[int, int]]:
    spans = []
    n = len(arr)
    if n < win:
        return [(0, n)]
    for i in range(0, n - win + 1, step):
        spans.append((i, i + win))
    return spans


@dataclass
class ActiveLearner:
    window: int = 32
    step: int = 8
    high_threshold: float = 0.85
    low_threshold: float = 0.60
    min_samples_to_train: int = 6

    # internal buffers
    X: List[np.ndarray] = field(default_factory=list)
    y: List[str] = field(default_factory=list)
    labels_: List[str] = field(default_factory=list)
    model: Optional[Pipeline] = None

    def _ensure_label(self, label: str):
        if label not in self.labels_:
            self.labels_.append(label)

    def add_labeled_span(self, seq: np.ndarray, start: int, end: int, label: str):
        """Add one positive example using the window within [start, end]."""
        self._ensure_label(label)
        start = int(max(0, min(len(seq)-1, start)))
        end = int(max(0, min(len(seq), end)))
        if end <= start:
            return
        mid = (start + end) // 2
        i0 = max(0, mid - self.window // 2)
        i1 = min(len(seq), i0 + self.window)
        i0 = max(0, i1 - self.window)  # fix tail
        feat = _window_features(seq[i0:i1])
        self.X.append(feat)
        self.y.append(label)

    def _maybe_train(self):
        if len(self.X) < self.min_samples_to_train or len(set(self.y)) < 1:
            self.model = None
            return
        # if only one class so far, do 1-NN novelty via distance to centroid
        if len(set(self.y)) == 1:
            # still use KNN (k=1) to produce prob=1.0 for that class when close
            clf = KNeighborsClassifier(n_neighbors=1)
        else:
            # LogisticReg works well in low-D
            clf = LogisticRegression(max_iter=500)
        self.model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        X_arr = np.vstack(self.X)
        self.model.fit(X_arr, np.array(self.y))

    def propagate(self, seq: np.ndarray, target_label: str) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Scan the sequence, predict probabilities for target_label on sliding windows,
        and return a dict:
          {
            "auto": [(start, end, prob), ...],
            "suggest": [(start, end, prob), ...]
          }
        """
        out = {"auto": [], "suggest": []}
        if len(seq) < 3:
            return out

        self._maybe_train()
        spans = _sliding_windows(seq, self.window, self.step)
        if not spans:
            return out

        feats = np.vstack([_window_features(seq[a:b]) for a, b in spans])
        if self.model is None:
            # not trained enough: fallback similarity to the last example of target_label
            last_idx = None
            for i in range(len(self.y)-1, -1, -1):
                if self.y[i] == target_label:
                    last_idx = i; break
            if last_idx is None:
                return out
            ref = self.X[last_idx]  # feature vector
            # cosine similarity
            ref_norm = np.linalg.norm(ref) + 1e-9
            sims = feats @ ref / (ref_norm * (np.linalg.norm(feats, axis=1) + 1e-9))
            for (a, b), s in zip(spans, sims):
                if s >= self.high_threshold:
                    out["auto"].append((a, b, float(s)))
                elif s >= self.low_threshold:
                    out["suggest"].append((a, b, float(s)))
            return out

        # trained model: probability for target_label (or decision function if binary)
        clf = self.model.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            # Map label -> column index
            classes = list(self.model.predict_proba(feats[:1])[0]*0 + 0)  # dummy just to touch shape
            # Above trick is to ensure predict_proba is called later without try/except
            proba = self.model.predict_proba(feats)
            # find index for target_label
            target_idx = list(self.model.classes_).index(target_label) if target_label in self.model.classes_ else None
            if target_idx is None:
                return out
            ps = proba[:, target_idx]
        else:
            # decision_function → squash to pseudo-prob via sigmoid
            from scipy.special import expit
            df = self.model.decision_function(feats)
            ps = expit(df) if df.ndim == 1 else expit(df[:, 0])

        for (a, b), p in zip(spans, ps):
            p = float(p)
            if p >= self.high_threshold:
                out["auto"].append((a, b, p))
            elif p >= self.low_threshold:
                out["suggest"].append((a, b, p))
        return out
