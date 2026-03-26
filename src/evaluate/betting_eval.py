"""Evaluate betting performance."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BETS_DIR, EVAL_DIR, INITIAL_BANKROLL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def profit_curve_plot(simulation: pd.DataFrame, output_path: Path):
    """Plot bankroll and profit over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    dates = pd.to_datetime(simulation["date"])

    # Bankroll
    ax1.plot(dates, simulation["bankroll"], "b-", linewidth=1.5)
    ax1.axhline(y=INITIAL_BANKROLL, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Bankroll ($)")
    ax1.set_title("Bankroll Evolution")
    ax1.grid(True, alpha=0.3)

    # Cumulative profit
    cum_profit = simulation["bankroll"] - INITIAL_BANKROLL
    colors = ["green" if p >= 0 else "red" for p in cum_profit]
    ax2.bar(dates, cum_profit, color=colors, alpha=0.7, width=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cumulative Profit ($)")
    ax2.set_xlabel("Date")
    ax2.set_title("Cumulative Profit")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_betting(
    simulation: pd.DataFrame,
    value_bets: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Compute betting performance metrics."""
    metrics = {}

    total_bets = int(simulation["n_bets"].sum())
    total_wins = int(simulation["n_wins"].sum())
    total_staked = float(simulation["staked"].sum())
    final_bankroll = float(simulation.iloc[-1]["bankroll"])
    total_profit = final_bankroll - INITIAL_BANKROLL

    metrics["total_bets"] = total_bets
    metrics["total_wins"] = total_wins
    metrics["win_rate"] = total_wins / total_bets if total_bets > 0 else 0
    metrics["total_staked"] = round(total_staked, 2)
    metrics["total_profit"] = round(total_profit, 2)
    metrics["roi"] = round(total_profit / total_staked, 4) if total_staked > 0 else 0
    metrics["final_bankroll"] = round(final_bankroll, 2)
    metrics["max_drawdown"] = round(float(simulation["drawdown"].max()), 4)
    metrics["avg_edge"] = round(float(value_bets["edge"].mean()), 4)
    metrics["avg_odds"] = round(float(value_bets["odds"].mean()), 2)
    metrics["n_matchdays"] = len(simulation)

    # Daily returns for Sharpe calculation
    daily_returns = simulation["profit"] / simulation["staked"]
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        metrics["sharpe_ratio"] = round(
            float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)), 2
        )

    # Yield by league
    if "league" in value_bets.columns and not value_bets.empty:
        league_stats = []
        for league, group in value_bets.groupby("league"):
            league_stats.append({
                "league": league,
                "bets": len(group),
                "win_rate": f"{group['won'].mean():.1%}",
                "avg_edge": f"{group['edge'].mean():.3f}",
            })
        metrics["league_breakdown"] = league_stats

    # Plots
    profit_curve_plot(simulation, output_dir / "profit_curve.png")

    # Save metrics
    with open(output_dir / "betting_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    sim_path = BETS_DIR / "simulation.csv"
    bets_path = BETS_DIR / "value_bets.csv"

    if not sim_path.exists():
        logger.info("No simulation data found. Run 'task bet' first.")
        return

    simulation = pd.read_csv(sim_path)
    value_bets = pd.read_csv(bets_path) if bets_path.exists() else pd.DataFrame()

    if simulation.empty:
        logger.info("Simulation is empty — no bets were placed.")
        return

    metrics = evaluate_betting(simulation, value_bets, EVAL_DIR)

    logger.info("\n=== Betting Performance ===")
    for k, v in metrics.items():
        if k != "league_breakdown":
            logger.info(f"  {k}: {v}")

    if "league_breakdown" in metrics:
        logger.info("\n  League breakdown:")
        for ls in metrics["league_breakdown"]:
            logger.info(f"    {ls['league']}: {ls['bets']} bets, "
                       f"win rate {ls['win_rate']}, avg edge {ls['avg_edge']}")


if __name__ == "__main__":
    main()
