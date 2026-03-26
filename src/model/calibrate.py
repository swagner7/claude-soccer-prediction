"""Probability calibration for the model."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator


class CalibratedMulticlassModel(BaseEstimator):
    """Calibrate multiclass probabilities using per-class isotonic regression."""

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


def calibrate_model(
    booster,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> CalibratedMulticlassModel:
    """Calibrate LightGBM probabilities using isotonic regression."""
    calibrator = CalibratedMulticlassModel(booster)
    calibrator.fit(X_cal, y_cal)
    return calibrator
