"""Logistic regression baseline model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_baseline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    """Train multinomial logistic regression baseline."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def predict_baseline(
    model: LogisticRegression, scaler: StandardScaler, X: pd.DataFrame
) -> np.ndarray:
    """Return calibrated probabilities from logistic regression."""
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)
