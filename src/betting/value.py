"""Identify value bets with market-aware filtering."""

import numpy as np
import pandas as pd

from src.config import (
    EXCLUDED_LEAGUES,
    MARKET_SHRINKAGE,
    MAX_ADJ_PROB,
    MAX_MARKET_DIVERGENCE,
    MAX_ODDS,
    MIN_ADJ_PROB,
    MIN_EDGE,
    MIN_PROB,
    PREFERRED_ODDS_TYPE,
)


def _get_market_prob(row, outcome: str) -> float:
    """Get overround-removed market probability for an outcome."""
    cols = {
        "H": ("market_prob_home", "odds_avg_home"),
        "D": ("market_prob_draw", "odds_avg_draw"),
        "A": ("market_prob_away", "odds_avg_away"),
    }
    market_col, odds_col = cols[outcome]

    # Prefer pre-computed market prob (overround-removed)
    if market_col in row.index and pd.notna(row[market_col]):
        return row[market_col]

    # Fall back to raw implied from average odds
    if odds_col in row.index and pd.notna(row[odds_col]) and row[odds_col] > 1:
        return 1.0 / row[odds_col]

    return np.nan


def _get_best_odds(row, outcome: str, preferred: str = PREFERRED_ODDS_TYPE) -> float:
    """Get the best available odds for placing the bet.

    Uses Pinnacle if available (sharpest line, most likely to accept the bet),
    otherwise falls back to average market odds.
    """
    col_map = {
        "H": {"pin": "odds_pin_home", "avg": "odds_avg_home", "b365": "odds_b365_home", "max": "odds_max_home"},
        "D": {"pin": "odds_pin_draw", "avg": "odds_avg_draw", "b365": "odds_b365_draw", "max": "odds_max_draw"},
        "A": {"pin": "odds_pin_away", "avg": "odds_avg_away", "b365": "odds_b365_away", "max": "odds_max_away"},
    }

    # Try preferred first, then fallback chain
    for odds_type in [preferred, "avg", "b365"]:
        col = col_map[outcome].get(odds_type)
        if col and col in row.index:
            val = row[col]
            if pd.notna(val) and val > 1.0:
                return val
    return np.nan


def shrink_toward_market(model_prob: float, market_prob: float, shrinkage: float) -> float:
    """Blend model probability toward market probability to reduce overconfidence."""
    if np.isnan(market_prob):
        return model_prob
    return model_prob * (1 - shrinkage) + market_prob * shrinkage


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE,
    min_prob: float = MIN_PROB,
    max_odds: float = MAX_ODDS,
    shrinkage: float = MARKET_SHRINKAGE,
    max_divergence: float = MAX_MARKET_DIVERGENCE,
    min_adj_prob: float = MIN_ADJ_PROB,
    max_adj_prob: float = MAX_ADJ_PROB,
    excluded_leagues: set = EXCLUDED_LEAGUES,
    odds_type: str = PREFERRED_ODDS_TYPE,
) -> pd.DataFrame:
    """
    Find value bets with market-aware filtering.

    Key improvements over naive edge calculation:
    1. Shrink model probs toward market to reduce systematic overconfidence
    2. Use Pinnacle (sharp) odds as benchmark — value vs sharp line means more
    3. Filter out bets where model diverges too far from market (overconfidence trap)
    4. Lower max odds to avoid longshot bias
    5. Adjusted probability range filter to avoid longshots and noisy favorites
    6. Exclude leagues with poor data quality / illiquid markets
    """
    prob_cols = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}

    bets = []
    for _, row in predictions.iterrows():
        # Skip excluded leagues
        league = row.get("league", "")
        if league in excluded_leagues:
            continue

        for outcome in ["H", "D", "A"]:
            raw_model_prob = row.get(prob_cols[outcome])
            if pd.isna(raw_model_prob):
                continue

            # Get market-implied probability
            market_prob = _get_market_prob(row, outcome)

            # Filter: skip if model diverges too far from market
            if pd.notna(market_prob) and abs(raw_model_prob - market_prob) > max_divergence:
                continue

            # Shrink model prob toward market
            adj_prob = shrink_toward_market(raw_model_prob, market_prob, shrinkage)

            if adj_prob < min_prob:
                continue

            # Filter: adjusted probability must be in sweet spot
            if adj_prob < min_adj_prob or adj_prob > max_adj_prob:
                continue

            # Get odds for edge calculation and bet placement
            odds = _get_best_odds(row, outcome, preferred=odds_type)
            if pd.isna(odds) or odds <= 1.0 or odds > max_odds:
                continue

            # Compute edge using adjusted (shrunk) probability
            edge = adj_prob * odds - 1.0

            if edge > min_edge:
                bets.append({
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_outcome": outcome,
                    "model_prob": raw_model_prob,
                    "adj_prob": adj_prob,
                    "market_prob": market_prob if pd.notna(market_prob) else None,
                    "odds": odds,
                    "implied_prob": 1.0 / odds,
                    "edge": edge,
                    "actual_result": row["result"],
                    "won": row["result"] == outcome,
                    "league": league,
                })

    return pd.DataFrame(bets)
