"""Kelly criterion bet sizing."""

import pandas as pd

from src.config import KELLY_FRACTION, MAX_SINGLE_BET_PCT, MAX_TOTAL_EXPOSURE_PCT


def kelly_fraction(prob: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Compute fractional Kelly stake as fraction of bankroll.
    Full Kelly: f* = (p * b - q) / b where b = odds - 1, q = 1 - p.
    """
    b = odds - 1.0
    q = 1.0 - prob
    full_kelly = (prob * b - q) / b

    if full_kelly <= 0:
        return 0.0

    return min(full_kelly * fraction, MAX_SINGLE_BET_PCT)


def size_bets(
    value_bets: pd.DataFrame,
    bankroll: float,
    max_single_pct: float = MAX_SINGLE_BET_PCT,
    max_total_pct: float = MAX_TOTAL_EXPOSURE_PCT,
) -> pd.DataFrame:
    """Apply Kelly sizing with exposure constraints.

    Uses adj_prob (market-shrunk probability) for sizing if available,
    otherwise falls back to model_prob.
    """
    if value_bets.empty:
        return value_bets

    bets = value_bets.copy()

    # Use adjusted prob for Kelly sizing (more conservative, accounts for overconfidence)
    prob_col = "adj_prob" if "adj_prob" in bets.columns else "model_prob"
    bets["kelly_frac"] = bets.apply(
        lambda r: kelly_fraction(r[prob_col], r["odds"]), axis=1
    )
    bets["kelly_frac"] = bets["kelly_frac"].clip(upper=max_single_pct)

    # Scale down if total exposure exceeds max
    total_exposure = bets["kelly_frac"].sum()
    if total_exposure > max_total_pct:
        scale = max_total_pct / total_exposure
        bets["kelly_frac"] *= scale

    bets["stake"] = (bets["kelly_frac"] * bankroll).round(2)
    bets = bets[bets["stake"] > 0]

    return bets
