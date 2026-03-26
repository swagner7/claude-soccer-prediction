"""Elo rating system for teams."""

import numpy as np
import pandas as pd

from src.config import ELO_HOME_ADVANTAGE, ELO_INITIAL_RATING, ELO_K_FACTOR


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    matches: pd.DataFrame,
    k_factor: float = ELO_K_FACTOR,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    initial_rating: float = ELO_INITIAL_RATING,
) -> pd.DataFrame:
    """
    Compute Elo ratings for each team, iterating chronologically.
    Returns DataFrame with home_elo, away_elo, elo_diff for each match.
    """
    ratings: dict[str, float] = {}
    results = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_elo = ratings.get(home, initial_rating)
        away_elo = ratings.get(away, initial_rating)

        # Record pre-match ratings as features
        results.append({
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
        })

        # Compute expected scores (with home advantage)
        exp_home = expected_score(home_elo + home_advantage, away_elo)
        exp_away = 1.0 - exp_home

        # Actual scores
        result = row["result"]
        if result == "H":
            actual_home, actual_away = 1.0, 0.0
        elif result == "A":
            actual_home, actual_away = 0.0, 1.0
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5

        # Update ratings
        ratings[home] = home_elo + k_factor * (actual_home - exp_home)
        ratings[away] = away_elo + k_factor * (actual_away - exp_away)

    return pd.DataFrame(results, index=matches.index)
