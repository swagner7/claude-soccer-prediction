"""Probability calibration for multiclass models.

Provides three calibration methods plus a pass-through (identity) option:
1. Platt scaling (logistic sigmoid per class) — 2 params per class, low overfit risk
2. Temperature scaling — single parameter, lowest overfit risk
3. Isotonic regression — non-parametric, highest flexibility but highest overfit risk

The recommended workflow is to try all methods and pick the one with the lowest
log loss on a held-out validation fold of the calibration set.
"""

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration strategies
# ---------------------------------------------------------------------------

class IdentityCalibrator:
    """Pass-through: returns raw probabilities unchanged."""

    def __init__(self, booster, n_classes=3):
        self.booster = booster
        self.n_classes = n_classes

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raw = self.booster.predict(X)
        if raw.ndim == 1:
            raw = raw.reshape(-1, self.n_classes)
        return raw

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class TemperatureScaledModel(BaseEstimator):
    """Temperature scaling: divide logits by a single learned temperature T.

    T > 1 softens probabilities (reduces overconfidence).
    T < 1 sharpens probabilities.
    This is the simplest calibration method with only 1 free parameter,
    making it very unlikely to overfit.
    """

    def __init__(self, booster, n_classes=3):
        self.booster = booster
        self.n_classes = n_classes
        self.temperature = 1.0

    def fit(self, X, y):
        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = raw_probs.reshape(-1, self.n_classes)

        # Clamp to avoid log(0)
        raw_probs = np.clip(raw_probs, 1e-15, 1.0)
        logits = np.log(raw_probs)

        def nll(T):
            scaled = logits / T
            # Softmax
            exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0)
            return log_loss(y, probs, labels=list(range(self.n_classes)))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        return self

    def predict_proba(self, X):
        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = raw_probs.reshape(-1, self.n_classes)

        raw_probs = np.clip(raw_probs, 1e-15, 1.0)
        logits = np.log(raw_probs)
        scaled = logits / self.temperature

        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class PlattScaledModel(BaseEstimator):
    """Platt scaling: fit a logistic regression per class on raw probabilities.

    Uses 2 parameters per class (slope + intercept), giving 6 total for
    3-class problems. Much less prone to overfit than isotonic regression.
    """

    def __init__(self, booster, n_classes=3):
        self.booster = booster
        self.n_classes = n_classes
        self.calibrators = []

    def fit(self, X, y):
        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = raw_probs.reshape(-1, self.n_classes)

        self.calibrators = []
        for i in range(self.n_classes):
            y_binary = (y == i).astype(float)
            lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
            lr.fit(raw_probs[:, i].reshape(-1, 1), y_binary)
            self.calibrators.append(lr)

        return self

    def predict_proba(self, X):
        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = raw_probs.reshape(-1, self.n_classes)

        calibrated = np.zeros_like(raw_probs)
        for i, lr in enumerate(self.calibrators):
            calibrated[:, i] = lr.predict_proba(raw_probs[:, i].reshape(-1, 1))[:, 1]

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class IsotonicCalibratedModel(BaseEstimator):
    """Isotonic regression per class — original method, kept for comparison."""

    def __init__(self, booster, n_classes=3):
        self.booster = booster
        self.n_classes = n_classes
        self.calibrators = []

    def fit(self, X, y):
        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = raw_probs.reshape(-1, self.n_classes)

        self.calibrators = []
        for i in range(self.n_classes):
            y_binary = (y == i).astype(float)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_probs[:, i], y_binary)
            self.calibrators.append(iso)

        return self

    def predict_proba(self, X):
        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = raw_probs.reshape(-1, self.n_classes)

        calibrated = np.zeros_like(raw_probs)
        for i, iso in enumerate(self.calibrators):
            calibrated[:, i] = iso.predict(raw_probs[:, i])

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# For backward compatibility
CalibratedMulticlassModel = IsotonicCalibratedModel


# ---------------------------------------------------------------------------
# Auto-selection: pick the best calibration method via cross-validation
# ---------------------------------------------------------------------------

CALIBRATION_METHODS = {
    "none": IdentityCalibrator,
    "temperature": TemperatureScaledModel,
    "platt": PlattScaledModel,
    "isotonic": IsotonicCalibratedModel,
}


def select_best_calibration(
    booster,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    n_classes: int = 3,
    n_folds: int = 3,
) -> tuple:
    """Select the best calibration method using cross-validation on the calibration set.

    Returns (calibrated_model, method_name, cv_scores_dict).

    For each method, runs n_folds CV on the calibration set to estimate
    out-of-sample log loss. Picks the method with the lowest CV log loss,
    then refits on the full calibration set.
    """
    cv_scores = {}

    for method_name, CalClass in CALIBRATION_METHODS.items():
        fold_scores = []
        kf = KFold(n_splits=n_folds, shuffle=False)  # No shuffle: preserve time ordering

        for train_idx, val_idx in kf.split(X_cal):
            cal_model = CalClass(booster, n_classes=n_classes)
            cal_model.fit(X_cal[train_idx], y_cal[train_idx])
            preds = cal_model.predict_proba(X_cal[val_idx])
            ll = log_loss(y_cal[val_idx], preds, labels=list(range(n_classes)))
            fold_scores.append(ll)

        mean_cv = np.mean(fold_scores)
        cv_scores[method_name] = mean_cv

    best_method = min(cv_scores, key=cv_scores.get)

    # Refit best method on full calibration set
    BestClass = CALIBRATION_METHODS[best_method]
    best_model = BestClass(booster, n_classes=n_classes)
    best_model.fit(X_cal, y_cal)

    return best_model, best_method, cv_scores
