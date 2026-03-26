"""Rolling team form features."""

import numpy as np
import pandas as pd

from src.config import FORM_WINDOWS


def _points(result: str, is_home: bool) -> int:
    if result == "H":
        return 3 if is_home else 0
    elif result == "A":
        return 0 if is_home else 3
    return 1


def compute_team_form(
    matches: pd.DataFrame, windows: list[int] = FORM_WINDOWS
) -> pd.DataFrame:
    """
    Compute rolling form features for each team.
    For each match, uses only historically completed matches (strict temporal ordering).
    """
    # Build per-team match history
    records: list[dict] = []
    for _, row in matches.iterrows():
        date = row["date"]
        for side in ["home", "away"]:
            is_home = side == "home"
            team = row[f"{side}_team"]
            goals_for = row[f"{side}_goals"]
            goals_against = row[f"{'away' if is_home else 'home'}_goals"]
            records.append({
                "date": date,
                "team": team,
                "is_home": is_home,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "points": _points(row["result"], is_home),
                "clean_sheet": 1 if goals_against == 0 else 0,
            })

    team_df = pd.DataFrame(records).sort_values("date")

    # Compute rolling stats per team
    form_data: dict[str, dict[int, dict[str, list]]] = {}
    # form_data[team][window] = {metric: [values by date]}

    team_histories: dict[str, list[dict]] = {}
    for _, row in team_df.iterrows():
        team = row["team"]
        if team not in team_histories:
            team_histories[team] = []
        team_histories[team].append(row.to_dict())

    # Now iterate through original matches and look up form
    home_features = []
    away_features = []

    for _, match_row in matches.iterrows():
        date = match_row["date"]
        features_home = {}
        features_away = {}

        for side, features in [("home", features_home), ("away", features_away)]:
            team = match_row[f"{side}_team"]
            is_home = side == "home"

            # Get all matches for this team BEFORE this date
            history = team_histories.get(team, [])
            past = [h for h in history if h["date"] < date]

            for w in windows:
                recent = past[-w:] if len(past) >= w else past
                n = len(recent)
                suffix = f"_{w}"

                if n == 0:
                    features[f"{side}_form_points{suffix}"] = np.nan
                    features[f"{side}_form_gd{suffix}"] = np.nan
                    features[f"{side}_form_goals_scored{suffix}"] = np.nan
                    features[f"{side}_form_goals_conceded{suffix}"] = np.nan
                    features[f"{side}_form_clean_sheets{suffix}"] = np.nan
                    features[f"{side}_home_form_points{suffix}"] = np.nan
                    features[f"{side}_away_form_points{suffix}"] = np.nan
                else:
                    pts = [r["points"] for r in recent]
                    gf = [r["goals_for"] for r in recent]
                    ga = [r["goals_against"] for r in recent]
                    cs = [r["clean_sheet"] for r in recent]

                    features[f"{side}_form_points{suffix}"] = sum(pts) / n
                    features[f"{side}_form_gd{suffix}"] = (sum(gf) - sum(ga)) / n
                    features[f"{side}_form_goals_scored{suffix}"] = sum(gf) / n
                    features[f"{side}_form_goals_conceded{suffix}"] = sum(ga) / n
                    features[f"{side}_form_clean_sheets{suffix}"] = sum(cs) / n

                    # Venue-specific form
                    home_past = [r for r in past if r["is_home"]][-w:]
                    away_past = [r for r in past if not r["is_home"]][-w:]

                    if home_past:
                        features[f"{side}_home_form_points{suffix}"] = (
                            sum(r["points"] for r in home_past) / len(home_past)
                        )
                    else:
                        features[f"{side}_home_form_points{suffix}"] = np.nan

                    if away_past:
                        features[f"{side}_away_form_points{suffix}"] = (
                            sum(r["points"] for r in away_past) / len(away_past)
                        )
                    else:
                        features[f"{side}_away_form_points{suffix}"] = np.nan

        home_features.append(features_home)
        away_features.append(features_away)

    home_df = pd.DataFrame(home_features, index=matches.index)
    away_df = pd.DataFrame(away_features, index=matches.index)
    return pd.concat([home_df, away_df], axis=1)
