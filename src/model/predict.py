"""Generate predictions from trained model."""

import numpy as np
import pandas as pd


def predict_match(
    calibrator, X: np.ndarray, match_info: pd.DataFrame
) -> pd.DataFrame:
    """Generate calibrated match predictions."""
    probs = calibrator.predict_proba(X)
    predictions = match_info[["date", "home_team", "away_team"]].copy()
    predictions["prob_home"] = probs[:, 0]
    predictions["prob_draw"] = probs[:, 1]
    predictions["prob_away"] = probs[:, 2]
    predictions["pred_result"] = np.argmax(probs, axis=1)
    predictions["pred_result"] = predictions["pred_result"].map(
        {0: "H", 1: "D", 2: "A"}
    )
    return predictions
