"""Identify value bets based on model predictions vs market odds."""

import numpy as np
import pandas as pd

from src.config import MAX_ODDS, MIN_EDGE, MIN_PROB


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE,
    min_prob: float = MIN_PROB,
    max_odds: float = MAX_ODDS,
    odds_type: str = "avg",  # "avg", "max", "b365", "pin"
) -> pd.DataFrame:
    """
    Find value bets where model_prob * odds > 1 + min_edge.

    Returns one row per value bet (a match can have 0-3 value bets).
    """
    odds_cols = {
        "H": f"odds_{odds_type}_home",
        "D": f"odds_{odds_type}_draw",
        "A": f"odds_{odds_type}_away",
    }
    prob_cols = {
        "H": "prob_home",
        "D": "prob_draw",
        "A": "prob_away",
    }

    bets = []
    for _, row in predictions.iterrows():
        for outcome in ["H", "D", "A"]:
            odds_col = odds_cols[outcome]
            prob_col = prob_cols[outcome]

            if odds_col not in row.index:
                continue

            odds = row[odds_col]
            model_prob = row[prob_col]

            if pd.isna(odds) or pd.isna(model_prob):
                continue
            if odds <= 1.0 or odds > max_odds:
                continue
            if model_prob < min_prob:
                continue

            implied_prob = 1.0 / odds
            edge = model_prob * odds - 1.0

            if edge > min_edge:
                bets.append({
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_outcome": outcome,
                    "model_prob": model_prob,
                    "odds": odds,
                    "implied_prob": implied_prob,
                    "edge": edge,
                    "actual_result": row["result"],
                    "won": row["result"] == outcome,
                    "league": row.get("league", ""),
                })

    return pd.DataFrame(bets)
