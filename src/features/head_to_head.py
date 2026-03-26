"""Head-to-head features between teams."""

import numpy as np
import pandas as pd

from src.config import H2H_MAX_LOOKBACK


def compute_h2h(
    matches: pd.DataFrame, max_lookback: int = H2H_MAX_LOOKBACK
) -> pd.DataFrame:
    """
    Compute head-to-head record between home and away team.
    Uses only matches completed before the current match date.
    """
    # Build H2H lookup: (team_a, team_b) -> list of past meetings
    # Stored as (date, home_goals, away_goals, was_team_a_home)
    h2h_history: dict[tuple[str, str], list[dict]] = {}

    results = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Key is always (sorted alphabetically) to find all meetings
        key = (min(home, away), max(home, away))

        past = [m for m in h2h_history.get(key, []) if m["date"] < date]
        past = past[-max_lookback:]

        if len(past) == 0:
            results.append({
                "h2h_matches": 0,
                "h2h_home_win_pct": np.nan,
                "h2h_draw_pct": np.nan,
                "h2h_away_win_pct": np.nan,
                "h2h_home_goals_avg": np.nan,
                "h2h_away_goals_avg": np.nan,
            })
        else:
            # Count results from perspective of current home team
            home_wins = 0
            draws = 0
            away_wins = 0
            home_goals_total = 0
            away_goals_total = 0

            for m in past:
                if m["home_team"] == home:
                    hg, ag = m["home_goals"], m["away_goals"]
                else:
                    hg, ag = m["away_goals"], m["home_goals"]

                home_goals_total += hg
                away_goals_total += ag

                if hg > ag:
                    home_wins += 1
                elif hg < ag:
                    away_wins += 1
                else:
                    draws += 1

            n = len(past)
            results.append({
                "h2h_matches": n,
                "h2h_home_win_pct": home_wins / n,
                "h2h_draw_pct": draws / n,
                "h2h_away_win_pct": away_wins / n,
                "h2h_home_goals_avg": home_goals_total / n,
                "h2h_away_goals_avg": away_goals_total / n,
            })

        # Add this match to history
        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append({
            "date": date,
            "home_team": home,
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
        })

    return pd.DataFrame(results, index=matches.index)
