"""Orchestrate feature engineering pipeline."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURES_DIR, PROCESSED_DIR
from src.features.elo import compute_elo_ratings
from src.features.head_to_head import compute_h2h
from src.features.match_stats import compute_rolling_stats
from src.features.team_form import compute_team_form

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute days since last match for each team."""
    last_match: dict[str, pd.Timestamp] = {}
    results = []

    for _, row in matches.iterrows():
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        home_rest = (date - last_match[home]).days if home in last_match else np.nan
        away_rest = (date - last_match[away]).days if away in last_match else np.nan

        results.append({
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
        })

        last_match[home] = date
        last_match[away] = date

    return pd.DataFrame(results, index=matches.index)


def compute_market_implied_probs(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute implied probabilities from average market odds (overround removed)."""
    results = []
    for _, row in matches.iterrows():
        odds_h = row.get("odds_avg_home", np.nan)
        odds_d = row.get("odds_avg_draw", np.nan)
        odds_a = row.get("odds_avg_away", np.nan)

        if pd.notna(odds_h) and pd.notna(odds_d) and pd.notna(odds_a):
            imp_h = 1.0 / odds_h
            imp_d = 1.0 / odds_d
            imp_a = 1.0 / odds_a
            overround = imp_h + imp_d + imp_a

            results.append({
                "market_prob_home": imp_h / overround,
                "market_prob_draw": imp_d / overround,
                "market_prob_away": imp_a / overround,
            })
        else:
            results.append({
                "market_prob_home": np.nan,
                "market_prob_draw": np.nan,
                "market_prob_away": np.nan,
            })

    return pd.DataFrame(results, index=matches.index)


def build_feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """Build complete feature matrix. All features use only pre-match data."""
    # Matches must be sorted by date
    matches = matches.sort_values("date").reset_index(drop=True)

    logger.info("Computing Elo ratings...")
    elo = compute_elo_ratings(matches)

    logger.info("Computing team form...")
    form = compute_team_form(matches)

    logger.info("Computing head-to-head...")
    h2h = compute_h2h(matches)

    logger.info("Computing rolling match stats...")
    stats = compute_rolling_stats(matches)

    logger.info("Computing rest days...")
    rest = compute_rest_days(matches)

    logger.info("Computing market implied probabilities...")
    market = compute_market_implied_probs(matches)

    # Encode league as numeric
    le = LabelEncoder()
    league_encoded = le.fit_transform(matches["division"])

    # Combine all features
    features = pd.concat([elo, form, h2h, stats, rest, market], axis=1)
    features["league_code"] = league_encoded

    # Add identifiers and target
    features["date"] = matches["date"].values
    features["home_team"] = matches["home_team"].values
    features["away_team"] = matches["away_team"].values
    features["season"] = matches["season"].values
    features["league"] = matches["league"].values
    features["division"] = matches["division"].values
    features["result"] = matches["result"].values
    features["result_code"] = matches["result_code"].values
    features["home_goals"] = matches["home_goals"].values
    features["away_goals"] = matches["away_goals"].values

    # Carry over odds for betting evaluation
    odds_cols = [c for c in matches.columns if c.startswith("odds_")]
    for col in odds_cols:
        features[col] = matches[col].values

    logger.info(f"Feature matrix: {features.shape[0]} rows, {features.shape[1]} columns")
    return features


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed matches...")
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    logger.info(f"Loaded {len(matches)} matches")

    features = build_feature_matrix(matches)

    out_path = FEATURES_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info(f"Saved features to {out_path}")

    # Summary
    feature_cols = [
        c for c in features.columns
        if c not in [
            "date", "home_team", "away_team", "season", "league", "division",
            "result", "result_code", "home_goals", "away_goals",
        ] and not c.startswith("odds_") and not c.startswith("market_prob_")
    ]
    logger.info(f"Feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        non_null = features[c].notna().sum()
        logger.info(f"  {c}: {non_null}/{len(features)} non-null")


if __name__ == "__main__":
    main()
