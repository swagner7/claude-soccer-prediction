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
            "rest_diff": (home_rest - away_rest) if (
                pd.notna(home_rest) and pd.notna(away_rest)
            ) else np.nan,
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


def compute_scoring_patterns(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute scoring pattern features: streaks, BTTS, over/under tendencies."""
    team_histories: dict[str, list[dict]] = {}

    for _, row in matches.iterrows():
        date = row["date"]
        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            opp_side = "away" if side == "home" else "home"
            gf = row[f"{side}_goals"]
            ga = row[f"{opp_side}_goals"]
            if team not in team_histories:
                team_histories[team] = []
            team_histories[team].append({
                "date": date,
                "goals_for": gf,
                "goals_against": ga,
                "total_goals": gf + ga,
                "btts": 1 if (gf > 0 and ga > 0) else 0,
                "over25": 1 if (gf + ga) > 2.5 else 0,
                "scored": 1 if gf > 0 else 0,
                "conceded": 1 if ga > 0 else 0,
                "won": 1 if gf > ga else 0,
            })

    results = []
    for _, row in matches.iterrows():
        date = row["date"]
        feats = {}

        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            history = team_histories.get(team, [])
            past = [h for h in history if h["date"] < date]
            recent = past[-5:] if past else []
            n = len(recent)

            if n == 0:
                feats[f"{side}_scored_pct_5"] = np.nan
                feats[f"{side}_btts_pct_5"] = np.nan
                feats[f"{side}_over25_pct_5"] = np.nan
                feats[f"{side}_avg_total_goals_5"] = np.nan
                feats[f"{side}_win_streak"] = 0
                feats[f"{side}_unbeaten_streak"] = 0
                feats[f"{side}_scoring_streak"] = 0
            else:
                feats[f"{side}_scored_pct_5"] = sum(r["scored"] for r in recent) / n
                feats[f"{side}_btts_pct_5"] = sum(r["btts"] for r in recent) / n
                feats[f"{side}_over25_pct_5"] = sum(r["over25"] for r in recent) / n
                feats[f"{side}_avg_total_goals_5"] = sum(
                    r["total_goals"] for r in recent
                ) / n

                # Streaks (from most recent backwards)
                win_streak = 0
                for r in reversed(past):
                    if r["won"]:
                        win_streak += 1
                    else:
                        break
                feats[f"{side}_win_streak"] = win_streak

                unbeaten = 0
                for r in reversed(past):
                    if r["goals_for"] >= r["goals_against"]:
                        unbeaten += 1
                    else:
                        break
                feats[f"{side}_unbeaten_streak"] = unbeaten

                scoring = 0
                for r in reversed(past):
                    if r["scored"]:
                        scoring += 1
                    else:
                        break
                feats[f"{side}_scoring_streak"] = scoring

        results.append(feats)

    return pd.DataFrame(results, index=matches.index)


def compute_league_position_proxy(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute a league position proxy via cumulative points within each season+division."""
    # Build cumulative points per team per season-division
    team_season_pts: dict[tuple[str, str, str], int] = {}  # (team, season, div) -> pts
    team_season_gd: dict[tuple[str, str, str], int] = {}
    team_season_played: dict[tuple[str, str, str], int] = {}

    results = []
    for _, row in matches.iterrows():
        season = row["season"]
        div = row["division"]
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]

        hk = (home, season, div)
        ak = (away, season, div)

        # Record pre-match standing
        h_pts = team_season_pts.get(hk, 0)
        a_pts = team_season_pts.get(ak, 0)
        h_gd = team_season_gd.get(hk, 0)
        a_gd = team_season_gd.get(ak, 0)
        h_played = team_season_played.get(hk, 0)
        a_played = team_season_played.get(ak, 0)

        h_ppg = h_pts / h_played if h_played > 0 else np.nan
        a_ppg = a_pts / a_played if a_played > 0 else np.nan

        results.append({
            "home_season_pts": h_pts,
            "away_season_pts": a_pts,
            "home_season_gd": h_gd,
            "away_season_gd": a_gd,
            "home_ppg": h_ppg,
            "away_ppg": a_ppg,
            "ppg_diff": (h_ppg - a_ppg) if (
                pd.notna(h_ppg) and pd.notna(a_ppg)
            ) else np.nan,
        })

        # Update standings
        hg = row["home_goals"]
        ag = row["away_goals"]
        if result == "H":
            team_season_pts[hk] = h_pts + 3
            team_season_pts[ak] = a_pts
        elif result == "A":
            team_season_pts[hk] = h_pts
            team_season_pts[ak] = a_pts + 3
        else:
            team_season_pts[hk] = h_pts + 1
            team_season_pts[ak] = a_pts + 1

        team_season_gd[hk] = h_gd + (hg - ag)
        team_season_gd[ak] = a_gd + (ag - hg)
        team_season_played[hk] = h_played + 1
        team_season_played[ak] = a_played + 1

    return pd.DataFrame(results, index=matches.index)


def compute_goal_supremacy(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling average goal supremacy (goal diff per match)."""
    team_histories: dict[str, list[dict]] = {}

    for _, row in matches.iterrows():
        date = row["date"]
        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            opp_side = "away" if side == "home" else "home"
            gf = row[f"{side}_goals"]
            ga = row[f"{opp_side}_goals"]
            if team not in team_histories:
                team_histories[team] = []
            team_histories[team].append({
                "date": date,
                "supremacy": gf - ga,
            })

    results = []
    for _, row in matches.iterrows():
        date = row["date"]
        feats = {}
        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            history = team_histories.get(team, [])
            past = [h for h in history if h["date"] < date]

            for w in [3, 5, 10]:
                recent = past[-w:] if past else []
                if recent:
                    feats[f"{side}_supremacy_avg_{w}"] = np.mean(
                        [r["supremacy"] for r in recent]
                    )
                else:
                    feats[f"{side}_supremacy_avg_{w}"] = np.nan

        results.append(feats)

    return pd.DataFrame(results, index=matches.index)


def compute_xg_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling xG-based features using Understat data.

    Only uses pre-match xG data (from prior matches) to avoid leakage.
    For leagues without xG data, all features will be NaN.
    """
    has_xg = "home_xg" in matches.columns
    if not has_xg:
        logger.info("No xG columns found in matches, skipping xG features")
        return pd.DataFrame(index=matches.index)

    team_xg_history: dict[str, list[dict]] = {}

    for _, row in matches.iterrows():
        date = row["date"]
        home_xg = row.get("home_xg")
        away_xg = row.get("away_xg")

        if pd.isna(home_xg) or pd.isna(away_xg):
            continue

        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_goals"]
        ag = row["away_goals"]

        for team, side in [(home, "home"), (away, "away")]:
            opp_side = "away" if side == "home" else "home"
            xg_for = row[f"{side}_xg"]
            xg_against = row[f"{opp_side}_xg"]
            goals_for = row[f"{side}_goals"]
            goals_against = row[f"{opp_side}_goals"]

            if team not in team_xg_history:
                team_xg_history[team] = []
            team_xg_history[team].append({
                "date": date,
                "xg_for": xg_for,
                "xg_against": xg_against,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "xg_diff": xg_for - xg_against,
                "xg_overperformance": goals_for - xg_for,
            })

    results = []
    for _, row in matches.iterrows():
        date = row["date"]
        feats = {}

        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            history = team_xg_history.get(team, [])
            past = [h for h in history if h["date"] < date]

            for window in [5, 10]:
                recent = past[-window:] if past else []
                n = len(recent)
                suffix = f"{side}_{window}"

                if n == 0:
                    feats[f"xg_for_avg_{suffix}"] = np.nan
                    feats[f"xg_against_avg_{suffix}"] = np.nan
                    feats[f"xg_diff_avg_{suffix}"] = np.nan
                    feats[f"xg_overperf_avg_{suffix}"] = np.nan
                else:
                    feats[f"xg_for_avg_{suffix}"] = np.mean([r["xg_for"] for r in recent])
                    feats[f"xg_against_avg_{suffix}"] = np.mean([r["xg_against"] for r in recent])
                    feats[f"xg_diff_avg_{suffix}"] = np.mean([r["xg_diff"] for r in recent])
                    feats[f"xg_overperf_avg_{suffix}"] = np.mean(
                        [r["xg_overperformance"] for r in recent]
                    )

        # xG differentials between teams
        for window in [5, 10]:
            h_diff = feats.get(f"xg_diff_avg_home_{window}")
            a_diff = feats.get(f"xg_diff_avg_away_{window}")
            if pd.notna(h_diff) and pd.notna(a_diff):
                feats[f"xg_diff_spread_{window}"] = h_diff - a_diff
            else:
                feats[f"xg_diff_spread_{window}"] = np.nan

        results.append(feats)

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

    logger.info("Computing scoring patterns...")
    scoring = compute_scoring_patterns(matches)

    logger.info("Computing league position proxy...")
    league_pos = compute_league_position_proxy(matches)

    logger.info("Computing goal supremacy...")
    supremacy = compute_goal_supremacy(matches)

    logger.info("Computing market implied probabilities...")
    market = compute_market_implied_probs(matches)

    logger.info("Computing xG features...")
    xg = compute_xg_features(matches)

    # Encode league as numeric
    le = LabelEncoder()
    league_encoded = le.fit_transform(matches["division"])

    # Combine all features
    feature_dfs = [elo, form, h2h, stats, rest, scoring, league_pos, supremacy, market]
    if not xg.empty:
        feature_dfs.append(xg)
    features = pd.concat(feature_dfs, axis=1)
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
