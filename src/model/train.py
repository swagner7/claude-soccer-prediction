"""Train separate model groups for Big 5 (with xG) and Others (without xG)."""

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
    BIG5_LEAGUES,
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
XG_PREFIX = "xg_"


def get_feature_columns(df: pd.DataFrame, include_xg: bool = True) -> list[str]:
    cols = [
        c for c in df.columns
        if c not in META_COLS
        and not c.startswith(ODDS_PREFIX)
        and not c.startswith(MARKET_PREFIX)
    ]
    if not include_xg:
        cols = [c for c in cols if not c.startswith(XG_PREFIX)]
    return cols


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


@_register_tuner("logreg")
def _tune_logreg(trial, X_train, y_train, cv_splits):
    kw = {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    scores = []
    for tr_idx, va_idx in cv_splits:
        pred, _ = train_logistic_regression(X_train[tr_idx], y_train[tr_idx], **kw)
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
# Train a single model group through all 3 phases
# ---------------------------------------------------------------------------

def train_group(
    group_name: str,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    save_dir,
) -> dict:
    """Run the full 3-phase pipeline for one league group.

    Returns dict with keys: best_name, all_models, raw_results, cal_results,
    feature_cols, col_means.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Split: train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)}")
    logger.info(f"  Train: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"  Cal:   {cal_df['date'].min()} to {cal_df['date'].max()}")
    logger.info(f"  Test:  {test_df['date'].min()} to {test_df['date'].max()}")

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

    joblib.dump(col_means, save_dir / "col_means.joblib")
    joblib.dump(feature_cols, save_dir / "feature_cols.joblib")

    # ===================================================================
    # Phase 1: Screen all models
    # ===================================================================
    logger.info(f"\n  --- Phase 1: Screening ({group_name}) ---")

    all_models = {}
    raw_results = {}

    for name, trainer in MODEL_TRAINERS.items():
        logger.info(f"    Training {name}...")
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
            logger.info(f"      {name:25s}  log_loss={ll:.4f}  accuracy={acc:.4f}")
        except Exception as e:
            logger.warning(f"      {name} failed: {e}")

    # Market benchmark
    market_cols = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    market_ll = np.nan
    if all(c in test_df.columns for c in market_cols):
        market_probs = test_df[market_cols].values
        valid = ~np.isnan(market_probs).any(axis=1)
        if valid.sum() > 0:
            market_ll = log_loss(y_test[valid], market_probs[valid], labels=[0, 1, 2])
            logger.info(f"      {'market':25s}  log_loss={market_ll:.4f}")

    ranked = sorted(raw_results.items(), key=lambda x: x[1]["log_loss"])
    screening_best = ranked[0][0]
    logger.info(f"    Screening winner: {screening_best} "
                f"(log_loss={raw_results[screening_best]['log_loss']:.4f})")

    # ===================================================================
    # Phase 2: Tune with Optuna
    # ===================================================================
    logger.info(f"\n  --- Phase 2: Tuning {screening_best} ({group_name}, "
                f"{OPTUNA_N_TRIALS} trials) ---")

    cv_splits = time_series_cv_splits(len(X_train), n_splits=3)

    if screening_best in TUNERS:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: TUNERS[screening_best](trial, X_train, y_train, cv_splits),
            n_trials=OPTUNA_N_TRIALS,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info(f"    Best CV log loss: {study.best_value:.4f}")
        logger.info(f"    Best params: {best_params}")

        tuned_pred, tuned_model = retrain_best(
            screening_best, best_params, X_train, y_train, X_cal, y_cal
        )
        tuned_preds = tuned_pred.predict(X_test)
        tuned_ll = log_loss(y_test, tuned_preds, labels=[0, 1, 2])
        tuned_acc = accuracy_score(y_test, np.argmax(tuned_preds, axis=1))
        logger.info(f"    Tuned {screening_best}: log_loss={tuned_ll:.4f}  "
                    f"accuracy={tuned_acc:.4f}")

        all_models[screening_best] = (tuned_pred, tuned_model)
        raw_results[screening_best] = {
            "log_loss": tuned_ll, "accuracy": tuned_acc, "preds": tuned_preds,
        }
    else:
        best_params = {}

    # ===================================================================
    # Phase 3: Calibrate
    # ===================================================================
    logger.info(f"\n  --- Phase 3: Calibrating ({group_name}) ---")

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
        logger.info(f"    {name:25s}  cal_log_loss={ll:.4f}  accuracy={acc:.4f}")

    best_name = min(cal_results, key=lambda k: cal_results[k]["log_loss"])
    logger.info(f"    BEST for {group_name}: {best_name} "
                f"(cal_log_loss={cal_results[best_name]['log_loss']:.4f})")

    # Save models
    for name, (_, raw_model) in all_models.items():
        joblib.dump(raw_model, save_dir / f"{name}_raw.joblib")
    for name, calibrator in calibrated_models.items():
        joblib.dump(calibrator, save_dir / f"{name}_calibrated.joblib")
    joblib.dump(best_name, save_dir / "best_model_name.joblib")

    # Model comparison JSON
    comparison = {
        "group": group_name,
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
    with open(save_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return {
        "best_name": best_name,
        "all_models": all_models,
        "raw_results": raw_results,
        "cal_results": cal_results,
        "feature_cols": feature_cols,
        "col_means": col_means,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features...")
    df = pd.read_parquet(FEATURES_DIR / "features.parquet")
    df = df.dropna(subset=["result_code"])

    # Split into Big 5 and Others
    is_big5 = df["league"].isin(BIG5_LEAGUES)
    big5_df = df[is_big5].copy()
    others_df = df[~is_big5].copy()

    logger.info(f"Big 5:   {len(big5_df)} matches ({sorted(big5_df['league'].unique())})")
    logger.info(f"Others:  {len(others_df)} matches ({sorted(others_df['league'].unique())})")

    # Feature columns: Big 5 gets xG features, Others does not
    big5_features = get_feature_columns(df, include_xg=True)
    others_features = get_feature_columns(df, include_xg=False)
    logger.info(f"Big 5 features: {len(big5_features)}  |  Others features: {len(others_features)}")

    # Temporal splits per group
    big5_train, big5_cal, big5_test = temporal_train_test_split(big5_df)
    others_train, others_cal, others_test = temporal_train_test_split(others_df)

    # ===================================================================
    # Train Big 5 group
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GROUP: Big 5 (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)")
    logger.info("=" * 70)

    big5_result = train_group(
        "big5", big5_train, big5_cal, big5_test, big5_features,
        MODELS_DIR / "big5",
    )

    # ===================================================================
    # Train Others group
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GROUP: Others (Championship, Eredivisie, Belgian Pro, Primeira, "
                "Super Lig, Greece)")
    logger.info("=" * 70)

    others_result = train_group(
        "others", others_train, others_cal, others_test, others_features,
        MODELS_DIR / "others",
    )

    # ===================================================================
    # Build unified test predictions
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Merging test predictions from both groups")
    logger.info("=" * 70)

    info_cols = (
        ["date", "home_team", "away_team", "result", "result_code",
         "home_goals", "away_goals", "league", "season"]
        + [c for c in df.columns if c.startswith("odds_")]
        + [c for c in df.columns if c.startswith("market_prob_")]
    )

    def build_test_results(test_df, result_dict, group_name):
        out = test_df[[c for c in info_cols if c in test_df.columns]].copy()
        out["model_group"] = group_name
        best = result_dict["best_name"]
        best_preds = result_dict["cal_results"][best]["preds"]
        out["prob_home"] = best_preds[:, 0]
        out["prob_draw"] = best_preds[:, 1]
        out["prob_away"] = best_preds[:, 2]

        for name in result_dict["all_models"]:
            for i, outcome in enumerate(["home", "draw", "away"]):
                out[f"raw_{name}_prob_{outcome}"] = result_dict["raw_results"][name]["preds"][:, i]
                out[f"cal_{name}_prob_{outcome}"] = result_dict["cal_results"][name]["preds"][:, i]
        return out

    big5_out = build_test_results(big5_test, big5_result, "big5")
    others_out = build_test_results(others_test, others_result, "others")

    # Unified test predictions
    combined = pd.concat([big5_out, others_out], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(MODELS_DIR / "test_predictions.parquet", index=False)

    logger.info(f"  Big 5 test: {len(big5_out)} matches, "
                f"best={big5_result['best_name']} "
                f"(cal_ll={big5_result['cal_results'][big5_result['best_name']]['log_loss']:.4f})")
    logger.info(f"  Others test: {len(others_out)} matches, "
                f"best={others_result['best_name']} "
                f"(cal_ll={others_result['cal_results'][others_result['best_name']]['log_loss']:.4f})")
    logger.info(f"  Combined: {len(combined)} test predictions saved")

    # Combined model comparison for evaluation plots
    combined_comparison = {
        "groups": {
            "big5": json.loads((MODELS_DIR / "big5" / "model_comparison.json").read_text()),
            "others": json.loads((MODELS_DIR / "others" / "model_comparison.json").read_text()),
        },
        "models": {},
    }
    # Merge per-model metrics across both groups (weighted average)
    all_model_names = set(big5_result["cal_results"].keys()) | set(others_result["cal_results"].keys())
    n_big5 = len(big5_test)
    n_others = len(others_test)
    n_total = n_big5 + n_others
    for name in all_model_names:
        entry = {}
        for metric_key in ["cal_log_loss", "cal_accuracy", "raw_log_loss", "raw_accuracy"]:
            vals = []
            if name in big5_result["cal_results"]:
                src = big5_result["cal_results"] if "cal" in metric_key else big5_result["raw_results"]
                k = metric_key.replace("cal_", "").replace("raw_", "")
                vals.append((src[name][k if "cal" not in metric_key else metric_key.replace("cal_", "")], n_big5))
            if name in others_result["cal_results"]:
                src = others_result["cal_results"] if "cal" in metric_key else others_result["raw_results"]
                k = metric_key.replace("cal_", "").replace("raw_", "")
                vals.append((src[name][k if "cal" not in metric_key else metric_key.replace("cal_", "")], n_others))
            if vals:
                entry[metric_key] = round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 4)
        combined_comparison["models"][name] = entry

    # Market benchmark (weighted)
    big5_mll = combined_comparison["groups"]["big5"].get("market_log_loss")
    others_mll = combined_comparison["groups"]["others"].get("market_log_loss")
    if big5_mll and others_mll:
        combined_comparison["market_log_loss"] = round(
            (big5_mll * n_big5 + others_mll * n_others) / n_total, 4
        )
    elif big5_mll:
        combined_comparison["market_log_loss"] = big5_mll
    elif others_mll:
        combined_comparison["market_log_loss"] = others_mll

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(combined_comparison, f, indent=2)

    logger.info(f"\nAll models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
