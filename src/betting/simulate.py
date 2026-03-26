"""Bankroll simulation for betting strategy."""

import logging
from pathlib import Path

import pandas as pd

from src.betting.kelly import size_bets
from src.betting.value import find_value_bets
from src.config import BETS_DIR, INITIAL_BANKROLL, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def simulate_bankroll(
    value_bets: pd.DataFrame,
    initial_bankroll: float = INITIAL_BANKROLL,
) -> pd.DataFrame:
    """Simulate bankroll evolution over time with Kelly sizing."""
    if value_bets.empty:
        return pd.DataFrame()

    bets = value_bets.sort_values("date").copy()
    bankroll = initial_bankroll
    peak = initial_bankroll
    records = []

    # Group bets by date (matchday)
    for date, day_bets in bets.groupby("date"):
        sized = size_bets(day_bets, bankroll)
        if sized.empty:
            continue

        day_profit = 0.0
        day_staked = 0.0
        day_wins = 0

        for _, bet in sized.iterrows():
            stake = bet["stake"]
            day_staked += stake

            if bet["won"]:
                profit = stake * (bet["odds"] - 1)
                day_wins += 1
            else:
                profit = -stake

            day_profit += profit

        bankroll += day_profit
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0

        records.append({
            "date": date,
            "n_bets": len(sized),
            "n_wins": day_wins,
            "staked": round(day_staked, 2),
            "profit": round(day_profit, 2),
            "bankroll": round(bankroll, 2),
            "peak": round(peak, 2),
            "drawdown": round(drawdown, 4),
        })

    return pd.DataFrame(records)


def main():
    BETS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test predictions...")
    predictions = pd.read_parquet(MODELS_DIR / "test_predictions.parquet")

    logger.info("Finding value bets...")
    value_bets = find_value_bets(predictions, odds_type="avg")
    logger.info(f"Found {len(value_bets)} value bets")

    if not value_bets.empty:
        value_bets.to_csv(BETS_DIR / "value_bets.csv", index=False)
        logger.info(f"Win rate: {value_bets['won'].mean():.1%}")

        # Also find bets using max odds
        value_bets_max = find_value_bets(predictions, odds_type="max")
        if not value_bets_max.empty:
            value_bets_max.to_csv(BETS_DIR / "value_bets_max.csv", index=False)
            logger.info(f"Max odds value bets: {len(value_bets_max)}")

        logger.info("Simulating bankroll...")
        simulation = simulate_bankroll(value_bets)
        simulation.to_csv(BETS_DIR / "simulation.csv", index=False)

        logger.info("\n=== Betting Summary ===")
        logger.info(f"  Total bets: {simulation['n_bets'].sum()}")
        logger.info(f"  Total staked: ${simulation['staked'].sum():.2f}")
        final = simulation.iloc[-1]
        logger.info(f"  Final bankroll: ${final['bankroll']:.2f}")
        logger.info(
            f"  Total profit: ${final['bankroll'] - INITIAL_BANKROLL:.2f}"
        )
        total_staked = simulation["staked"].sum()
        if total_staked > 0:
            roi = (final["bankroll"] - INITIAL_BANKROLL) / total_staked
            logger.info(f"  ROI: {roi:.1%}")
        logger.info(f"  Max drawdown: {simulation['drawdown'].max():.1%}")
    else:
        logger.info("No value bets found with current parameters.")


if __name__ == "__main__":
    main()
