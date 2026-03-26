"""Rolling match statistics features (shots, corners, fouls, cards)."""

import numpy as np
import pandas as pd

from src.config import FORM_WINDOWS

STAT_COLUMNS = [
    ("shots", "home_shots", "away_shots"),
    ("shots_on_target", "home_shots_on_target", "away_shots_on_target"),
    ("corners", "home_corners", "away_corners"),
    ("fouls", "home_fouls", "away_fouls"),
    ("yellows", "home_yellows", "away_yellows"),
]


def compute_rolling_stats(
    matches: pd.DataFrame, windows: list[int] = FORM_WINDOWS
) -> pd.DataFrame:
    """Compute rolling averages of match statistics per team."""
    # Build per-team stat history
    team_stats: dict[str, list[dict]] = {}

    for _, row in matches.iterrows():
        date = row["date"]
        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            opp_side = "away" if side == "home" else "home"
            record = {"date": date}
            for stat_name, home_col, away_col in STAT_COLUMNS:
                for_col = home_col if side == "home" else away_col
                against_col = away_col if side == "home" else home_col
                record[f"{stat_name}_for"] = row.get(for_col, np.nan)
                record[f"{stat_name}_against"] = row.get(against_col, np.nan)

            if team not in team_stats:
                team_stats[team] = []
            team_stats[team].append(record)

    # Compute rolling averages for each match
    features_list = []

    for _, row in matches.iterrows():
        date = row["date"]
        features = {}

        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            history = team_stats.get(team, [])
            past = [h for h in history if h["date"] < date]

            for w in windows:
                recent = past[-w:] if past else []
                n = len(recent)

                for stat_name, _, _ in STAT_COLUMNS:
                    for direction in ["for", "against"]:
                        col = f"{side}_{stat_name}_{direction}_avg_{w}"
                        if n == 0:
                            features[col] = np.nan
                        else:
                            vals = [r[f"{stat_name}_{direction}"] for r in recent]
                            vals = [v for v in vals if not np.isnan(v)]
                            features[col] = np.mean(vals) if vals else np.nan

        features_list.append(features)

    return pd.DataFrame(features_list, index=matches.index)
