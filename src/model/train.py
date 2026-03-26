"""Train logistic regression model for match prediction."""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.config import (
    CALIBRATION_FRACTION,
    FEATURES_DIR,
    MODELS_DIR,
    TEST_FRACTION,
)
from src.model.baseline import predict_baseline, train_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Feature columns to exclude from training
META_COLS = [
    "date", "home_team", "away_team", "season", "league", "division",
    "result", "result_code", "home_goals", "away_goals",
]
ODDS_PREFIX = "odds_"
MARKET_PREFIX = "market_prob_"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns for training."""
    return [
        c for c in df.columns
        if c not in META_COLS
        and not c.startswith(ODDS_PREFIX)
        and not c.startswith(MARKET_PREFIX)
    ]


def temporal_train_test_split(
    df: pd.DataFrame,
    test_frac: float = TEST_FRACTION,
    cal_frac: float = CALIBRATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train | calibration | test."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - test_frac - cal_frac))
    cal_end = int(n * (1 - test_frac))

    train = df.iloc[:train_end]
    cal = df.iloc[train_end:cal_end]
    test = df.iloc[cal_end:]

    return train, cal, test


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features...")
    df = pd.read_parquet(FEATURES_DIR / "features.parquet")

    # Drop rows with NaN target
    df = df.dropna(subset=["result_code"])

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")

    # Temporal split
    train_df, cal_df, test_df = temporal_train_test_split(df)
    logger.info(
        f"Split: train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)}"
    )
    logger.info(
        f"Train dates: {train_df['date'].min()} to {train_df['date'].max()}"
    )
    logger.info(
        f"Test dates: {test_df['date'].min()} to {test_df['date'].max()}"
    )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["result_code"].values.astype(int)
    X_cal = cal_df[feature_cols].values.astype(np.float32)
    y_cal = cal_df["result_code"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["result_code"].values.astype(int)

    # Handle NaN in features
    nan_mask = np.isnan(X_train)
    col_means = np.nanmean(X_train, axis=0) if nan_mask.any() else None
    if nan_mask.any():
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_means[j]
            X_cal[np.isnan(X_cal[:, j]), j] = col_means[j]
            X_test[np.isnan(X_test[:, j]), j] = col_means[j]
        logger.info("Imputed NaN values with column means")

    # Save column means and feature list for prediction time
    joblib.dump(col_means, MODELS_DIR / "col_means.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    # --- Train logistic regression ---
    logger.info("Training logistic regression...")
    model, scaler = train_baseline(
        pd.DataFrame(X_train, columns=feature_cols),
        pd.Series(y_train),
    )
    joblib.dump((model, scaler), MODELS_DIR / "logreg_v1.joblib")

    train_preds = predict_baseline(
        model, scaler, pd.DataFrame(X_train, columns=feature_cols)
    )
    train_logloss = log_loss(y_train, train_preds, labels=[0, 1, 2])
    logger.info(f"Train log loss: {train_logloss:.4f}")

    test_preds = predict_baseline(
        model, scaler, pd.DataFrame(X_test, columns=feature_cols)
    )
    test_logloss = log_loss(y_test, test_preds, labels=[0, 1, 2])
    logger.info(f"Test log loss:  {test_logloss:.4f}")

    test_accuracy = (np.argmax(test_preds, axis=1) == y_test).mean()
    logger.info(f"Test accuracy:  {test_accuracy:.4f}")

    # --- Market implied probabilities as benchmark ---
    market_cols = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    if all(c in test_df.columns for c in market_cols):
        market_probs = test_df[market_cols].values
        valid_market = ~np.isnan(market_probs).any(axis=1)
        if valid_market.sum() > 0:
            market_logloss = log_loss(
                y_test[valid_market],
                market_probs[valid_market],
                labels=[0, 1, 2],
            )
            logger.info(f"Market implied log loss: {market_logloss:.4f}")

    # --- Save test predictions for evaluation ---
    test_results = test_df[
        ["date", "home_team", "away_team", "result", "result_code",
         "home_goals", "away_goals", "league", "season"]
        + [c for c in test_df.columns if c.startswith("odds_")]
        + [c for c in test_df.columns if c.startswith("market_prob_")]
    ].copy()
    test_results["prob_home"] = test_preds[:, 0]
    test_results["prob_draw"] = test_preds[:, 1]
    test_results["prob_away"] = test_preds[:, 2]
    test_results.to_parquet(MODELS_DIR / "test_predictions.parquet", index=False)

    logger.info(f"\nModels saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
