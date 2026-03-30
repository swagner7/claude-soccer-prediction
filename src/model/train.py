"""Train multiple models, select best, tune it with Optuna, calibrate, and save."""

import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    CALIBRATION_FRACTION,
    FEATURES_DIR,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    RANDOM_STATE,
    TEST_FRACTION,
)
from src.model.calibrate import CalibratedMulticlassModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

META_COLS = [
    "date", "home_team", "away_team", "season", "league", "division",
    "result", "result_code", "home_goals", "away_goals",
]
ODDS_PREFIX = "odds_"
MARKET_PREFIX = "market_prob_"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in META_COLS
        and not c.startswith(ODDS_PREFIX)
        and not c.startswith(MARKET_PREFIX)
    ]


def temporal_train_test_split(df, test_frac=TEST_FRACTION, cal_frac=CALIBRATION_FRACTION):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - test_frac - cal_frac))
    cal_end = int(n * (1 - test_frac))
    return df.iloc[:train_end], df.iloc[train_end:cal_end], df.iloc[cal_end:]


def time_series_cv_splits(n_samples, n_splits=5):
    min_train = n_samples // (n_splits + 1)
    fold_size = (n_samples - min_train) // n_splits
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_end = min(train_end + fold_size, n_samples)
        splits.append((np.arange(train_end), np.arange(train_end, val_end)))
    return splits


# ---------------------------------------------------------------------------
# Unified predictor wrappers
# ---------------------------------------------------------------------------

class SklearnPredictor:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        X_in = self.scaler.transform(X) if self.scaler is not None else X
        return self.model.predict_proba(X_in)


class LGBMPredictor:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(X)


class XGBPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        import xgboost as xgb
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)


# ---------------------------------------------------------------------------
# Model trainers (default hyperparams for screening)
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, **kw):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=kw.get("C", 1.0),
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled, y_train)
    return SklearnPredictor(model, scaler), model


def train_random_forest(X_train, y_train, **kw):
    model = RandomForestClassifier(
        n_estimators=kw.get("n_estimators", 300),
        max_depth=kw.get("max_depth", 12),
        min_samples_leaf=kw.get("min_samples_leaf", 20),
        max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return SklearnPredictor(model), model


def train_extra_trees(X_train, y_train, **kw):
    model = ExtraTreesClassifier(
        n_estimators=kw.get("n_estimators", 300),
        max_depth=kw.get("max_depth", 14),
        min_samples_leaf=kw.get("min_samples_leaf", 15),
        max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return SklearnPredictor(model), model


def train_gradient_boosting(X_train, y_train, **kw):
    model = GradientBoostingClassifier(
        n_estimators=kw.get("n_estimators", 200),
        max_depth=kw.get("max_depth", 5),
        learning_rate=kw.get("learning_rate", 0.05),
        subsample=kw.get("subsample", 0.8),
        min_samples_leaf=kw.get("min_samples_leaf", 20),
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return SklearnPredictor(model), model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, **kw):
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "seed": RANDOM_STATE,
        "verbosity": 0,
        "max_depth": kw.get("max_depth", 6),
        "learning_rate": kw.get("learning_rate", 0.05),
        "subsample": kw.get("subsample", 0.8),
        "colsample_bytree": kw.get("colsample_bytree", 0.8),
        "reg_alpha": kw.get("reg_alpha", 0.1),
        "reg_lambda": kw.get("reg_lambda", 1.0),
        "min_child_weight": kw.get("min_child_weight", 5),
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    if X_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))
    booster = xgb.train(
        params, dtrain,
        num_boost_round=kw.get("n_rounds", 500),
        evals=evals,
        early_stopping_rounds=30 if X_val is not None else None,
        verbose_eval=0,
    )
    return XGBPredictor(booster), booster


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, **kw):
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "num_leaves": kw.get("num_leaves", 50),
        "learning_rate": kw.get("learning_rate", 0.05),
        "min_child_samples": kw.get("min_child_samples", 30),
        "feature_fraction": kw.get("feature_fraction", 0.8),
        "bagging_fraction": kw.get("bagging_fraction", 0.8),
        "bagging_freq": 5,
        "reg_alpha": kw.get("reg_alpha", 0.1),
        "reg_lambda": kw.get("reg_lambda", 1.0),
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    valid_sets = [dtrain]
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)
    booster = lgb.train(
        params, dtrain,
        num_boost_round=kw.get("n_rounds", 500),
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return LGBMPredictor(booster), booster


def train_mlp(X_train, y_train, **kw):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=kw.get("hidden_layer_sizes", (128, 64)),
        activation="relu",
        max_iter=kw.get("max_iter", 300),
        learning_rate_init=kw.get("learning_rate_init", 0.001),
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled, y_train)
    return SklearnPredictor(model, scaler), model


# ---------------------------------------------------------------------------
# Optuna tuning per model type
# ---------------------------------------------------------------------------

TUNERS = {}


def _register_tuner(name):
    def decorator(fn):
        TUNERS[name] = fn
        return fn
    return decorator


@_register_tuner("lightgbm")
def _tune_lightgbm(trial, X_train, y_train, cv_splits):
    kw = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "n_rounds": trial.suggest_int("n_rounds", 100, 600),
    }
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_lightgbm(
            X_train[tr_idx], y_train[tr_idx],
            X_train[va_idx], y_train[va_idx], **kw,
        )
        preds = pred.predict(X_train[va_idx])
        scores.append(log_loss(y_train[va_idx], preds, labels=[0, 1, 2]))
    return np.mean(scores)


@_register_tuner("xgboost")
def _tune_xgboost(trial, X_train, y_train, cv_splits):
    kw = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "n_rounds": trial.suggest_int("n_rounds", 100, 600),
    }
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_xgboost(
            X_train[tr_idx], y_train[tr_idx],
            X_train[va_idx], y_train[va_idx], **kw,
        )
        preds = pred.predict(X_train[va_idx])
        scores.append(log_loss(y_train[va_idx], preds, labels=[0, 1, 2]))
    return np.mean(scores)


@_register_tuner("random_forest")
def _tune_rf(trial, X_train, y_train, cv_splits):
    kw = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 6, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_random_forest(X_train[tr_idx], y_train[tr_idx], **kw)
        preds = pred.predict(X_train[va_idx])
        scores.append(log_loss(y_train[va_idx], preds, labels=[0, 1, 2]))
    return np.mean(scores)


@_register_tuner("gradient_boosting")
def _tune_gb(trial, X_train, y_train, cv_splits):
    kw = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
    }
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_gradient_boosting(X_train[tr_idx], y_train[tr_idx], **kw)
        preds = pred.predict(X_train[va_idx])
        scores.append(log_loss(y_train[va_idx], preds, labels=[0, 1, 2]))
    return np.mean(scores)


@_register_tuner("extra_trees")
def _tune_et(trial, X_train, y_train, cv_splits):
    kw = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 6, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_extra_trees(X_train[tr_idx], y_train[tr_idx], **kw)
        preds = pred.predict(X_train[va_idx])
        scores.append(log_loss(y_train[va_idx], preds, labels=[0, 1, 2]))
    return np.mean(scores)


@_register_tuner("mlp")
def _tune_mlp(trial, X_train, y_train, cv_splits):
    h1 = trial.suggest_int("h1", 64, 256)
    h2 = trial.suggest_int("h2", 32, 128)
    kw = {
        "hidden_layer_sizes": (h1, h2),
        "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "max_iter": 300,
    }
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_mlp(X_train[tr_idx], y_train[tr_idx], **kw)
        preds = pred.predict(X_train[va_idx])
        scores.append(log_loss(y_train[va_idx], preds, labels=[0, 1, 2]))
    return np.mean(scores)


# Map model names to trainers
MODEL_TRAINERS = {
    "logreg": train_logistic_regression,
    "random_forest": train_random_forest,
    "extra_trees": train_extra_trees,
    "gradient_boosting": train_gradient_boosting,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "mlp": train_mlp,
}

# Models that need val set for early stopping
NEEDS_VAL = {"lightgbm", "xgboost"}


def retrain_best(best_name, best_params, X_train, y_train, X_val, y_val):
    """Retrain the best model with tuned hyperparameters."""
    trainer = MODEL_TRAINERS[best_name]
    if best_name in NEEDS_VAL:
        return trainer(X_train, y_train, X_val, y_val, **best_params)
    return trainer(X_train, y_train, **best_params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features...")
    df = pd.read_parquet(FEATURES_DIR / "features.parquet")
    df = df.dropna(subset=["result_code"])

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")

    train_df, cal_df, test_df = temporal_train_test_split(df)
    logger.info(f"Split: train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)}")
    logger.info(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Cal:   {cal_df['date'].min()} to {cal_df['date'].max()}")
    logger.info(f"Test:  {test_df['date'].min()} to {test_df['date'].max()}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["result_code"].values.astype(int)
    X_cal = cal_df[feature_cols].values.astype(np.float32)
    y_cal = cal_df["result_code"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["result_code"].values.astype(int)

    # Impute NaN
    col_means = np.nanmean(X_train, axis=0)
    for X in [X_train, X_cal, X_test]:
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_means[j]
    logger.info("Imputed NaN values with column means")

    joblib.dump(col_means, MODELS_DIR / "col_means.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    # ===================================================================
    # Phase 1: Screen all models with default hyperparameters
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Screening all models (default hyperparameters)")
    logger.info("=" * 60)

    all_models = {}   # name -> (predictor, raw_model)
    raw_results = {}  # name -> {log_loss, accuracy, preds}

    for name, trainer in MODEL_TRAINERS.items():
        logger.info(f"\n  Training {name}...")
        try:
            if name in NEEDS_VAL:
                pred, raw_model = trainer(X_train, y_train, X_cal, y_cal)
            else:
                pred, raw_model = trainer(X_train, y_train)
            preds = pred.predict(X_test)
            ll = log_loss(y_test, preds, labels=[0, 1, 2])
            acc = accuracy_score(y_test, np.argmax(preds, axis=1))
            all_models[name] = (pred, raw_model)
            raw_results[name] = {"log_loss": ll, "accuracy": acc, "preds": preds}
            logger.info(f"    {name:25s}  log_loss={ll:.4f}  accuracy={acc:.4f}")
        except Exception as e:
            logger.warning(f"    {name} failed: {e}")

    # Market benchmark
    market_cols = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    market_ll = np.nan
    if all(c in test_df.columns for c in market_cols):
        market_probs = test_df[market_cols].values
        valid = ~np.isnan(market_probs).any(axis=1)
        if valid.sum() > 0:
            market_ll = log_loss(y_test[valid], market_probs[valid], labels=[0, 1, 2])
            logger.info(f"\n  {'market':25s}  log_loss={market_ll:.4f}")

    # Rank by raw log loss and pick best
    ranked = sorted(raw_results.items(), key=lambda x: x[1]["log_loss"])
    screening_best = ranked[0][0]
    logger.info(f"\n  Screening winner: {screening_best} "
                f"(log_loss={raw_results[screening_best]['log_loss']:.4f})")

    # ===================================================================
    # Phase 2: Tune the best model with Optuna
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Tuning {screening_best} with Optuna ({OPTUNA_N_TRIALS} trials)")
    logger.info("=" * 60)

    cv_splits = time_series_cv_splits(len(X_train), n_splits=5)

    if screening_best in TUNERS:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: TUNERS[screening_best](trial, X_train, y_train, cv_splits),
            n_trials=OPTUNA_N_TRIALS,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info(f"  Best CV log loss: {study.best_value:.4f}")
        logger.info(f"  Best params: {best_params}")

        # Retrain with tuned params on full train set
        logger.info(f"\n  Retraining {screening_best} with tuned params...")
        tuned_pred, tuned_model = retrain_best(
            screening_best, best_params, X_train, y_train, X_cal, y_cal
        )
        tuned_preds = tuned_pred.predict(X_test)
        tuned_ll = log_loss(y_test, tuned_preds, labels=[0, 1, 2])
        tuned_acc = accuracy_score(y_test, np.argmax(tuned_preds, axis=1))
        logger.info(f"  Tuned {screening_best}: log_loss={tuned_ll:.4f}  accuracy={tuned_acc:.4f}")

        # Replace the default model with the tuned one
        all_models[screening_best] = (tuned_pred, tuned_model)
        raw_results[screening_best] = {
            "log_loss": tuned_ll, "accuracy": tuned_acc, "preds": tuned_preds,
        }
    else:
        best_params = {}
        logger.info(f"  No tuner registered for {screening_best}, skipping Optuna.")

    # ===================================================================
    # Phase 3: Calibrate all models, pick final best
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Calibrating all models (isotonic regression)")
    logger.info("=" * 60)

    calibrated_models = {}
    cal_results = {}

    for name, (predictor, _) in all_models.items():
        calibrator = CalibratedMulticlassModel(predictor)
        calibrator.fit(X_cal, y_cal)
        calibrated_models[name] = calibrator

        cal_preds = calibrator.predict_proba(X_test)
        ll = log_loss(y_test, cal_preds, labels=[0, 1, 2])
        acc = accuracy_score(y_test, np.argmax(cal_preds, axis=1))
        cal_results[name] = {"log_loss": ll, "accuracy": acc, "preds": cal_preds}
        logger.info(f"  {name:25s}  cal_log_loss={ll:.4f}  accuracy={acc:.4f}")

    best_name = min(cal_results, key=lambda k: cal_results[k]["log_loss"])
    logger.info(f"\n  FINAL BEST: {best_name} "
                f"(calibrated log_loss={cal_results[best_name]['log_loss']:.4f})")

    # ===================================================================
    # Save everything
    # ===================================================================
    for name, (_, raw_model) in all_models.items():
        joblib.dump(raw_model, MODELS_DIR / f"{name}_raw.joblib")
    for name, calibrator in calibrated_models.items():
        joblib.dump(calibrator, MODELS_DIR / f"{name}_calibrated.joblib")
    joblib.dump(best_name, MODELS_DIR / "best_model_name.joblib")

    # Test predictions
    test_results = test_df[
        ["date", "home_team", "away_team", "result", "result_code",
         "home_goals", "away_goals", "league", "season"]
        + [c for c in test_df.columns if c.startswith("odds_")]
        + [c for c in test_df.columns if c.startswith("market_prob_")]
    ].copy()

    best_preds = cal_results[best_name]["preds"]
    test_results["prob_home"] = best_preds[:, 0]
    test_results["prob_draw"] = best_preds[:, 1]
    test_results["prob_away"] = best_preds[:, 2]

    for name in all_models:
        for i, outcome in enumerate(["home", "draw", "away"]):
            test_results[f"raw_{name}_prob_{outcome}"] = raw_results[name]["preds"][:, i]
            test_results[f"cal_{name}_prob_{outcome}"] = cal_results[name]["preds"][:, i]

    test_results.to_parquet(MODELS_DIR / "test_predictions.parquet", index=False)

    # Model comparison JSON
    comparison = {
        "models": {},
        "best_model": best_name,
        "tuned_model": screening_best,
        "tuned_params": best_params,
        "market_log_loss": float(market_ll) if pd.notna(market_ll) else None,
    }
    for name in all_models:
        comparison["models"][name] = {
            "raw_log_loss": round(raw_results[name]["log_loss"], 4),
            "raw_accuracy": round(raw_results[name]["accuracy"], 4),
            "cal_log_loss": round(cal_results[name]["log_loss"], 4),
            "cal_accuracy": round(cal_results[name]["accuracy"], 4),
        }

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nAll models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
