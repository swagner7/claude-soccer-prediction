"""Evaluate betting performance with rich visualizations."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.betting.simulate import simulate_bankroll
from src.config import BETS_DIR, EVAL_DIR, INITIAL_BANKROLL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def bankroll_plot(simulation: pd.DataFrame, output_path: Path):
    """Plot bankroll evolution with drawdown and bet volume."""
    dates = pd.to_datetime(simulation["date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Bankroll line ---
    ax1 = axes[0]
    bankroll = simulation["bankroll"].values
    ax1.fill_between(dates, INITIAL_BANKROLL, bankroll,
                     where=bankroll >= INITIAL_BANKROLL,
                     color="#55A868", alpha=0.15, interpolate=True)
    ax1.fill_between(dates, INITIAL_BANKROLL, bankroll,
                     where=bankroll < INITIAL_BANKROLL,
                     color="#C44E52", alpha=0.15, interpolate=True)
    ax1.plot(dates, bankroll, color="#2C3E50", linewidth=2, label="Bankroll")
    ax1.axhline(y=INITIAL_BANKROLL, color="gray", linestyle="--", alpha=0.5,
                linewidth=1, label=f"Starting (${INITIAL_BANKROLL:,.0f})")

    # Annotate final value
    final = bankroll[-1]
    profit = final - INITIAL_BANKROLL
    color = "#55A868" if profit >= 0 else "#C44E52"
    sign = "+" if profit >= 0 else ""
    ax1.annotate(f"${final:,.0f} ({sign}${profit:,.0f})",
                 xy=(dates.iloc[-1], final),
                 xytext=(10, 10 if profit >= 0 else -20),
                 textcoords="offset points",
                 fontsize=11, fontweight="bold", color=color,
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax1.set_ylabel("Bankroll ($)", fontsize=12)
    ax1.set_title("Bankroll Evolution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # --- Drawdown ---
    ax2 = axes[1]
    drawdown_pct = simulation["drawdown"].values * 100
    ax2.fill_between(dates, 0, drawdown_pct, color="#C44E52", alpha=0.3)
    ax2.plot(dates, drawdown_pct, color="#C44E52", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_title("Drawdown from Peak", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.invert_yaxis()

    # --- Daily profit bars ---
    ax3 = axes[2]
    profits = simulation["profit"].values
    colors = ["#55A868" if p >= 0 else "#C44E52" for p in profits]
    ax3.bar(dates, profits, color=colors, alpha=0.7, width=2)
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.set_ylabel("Daily P&L ($)", fontsize=12)
    ax3.set_title("Daily Profit / Loss", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=12)
    ax3.grid(True, alpha=0.2, axis="y")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def roi_by_league_plot(value_bets: pd.DataFrame, output_path: Path):
    """Plot win rate and average edge by league."""
    if value_bets.empty or "league" not in value_bets.columns:
        return

    stats = value_bets.groupby("league").agg(
        bets=("won", "count"),
        wins=("won", "sum"),
        win_rate=("won", "mean"),
        avg_edge=("edge", "mean"),
        avg_odds=("odds", "mean"),
    ).sort_values("bets", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Win rate by league
    colors = ["#55A868" if wr > 0.4 else "#4C72B0" if wr > 0.3 else "#C44E52"
              for wr in stats["win_rate"]]
    ax1.barh(range(len(stats)), stats["win_rate"], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(stats)))
    ax1.set_yticklabels(stats.index, fontsize=10)
    ax1.set_xlabel("Win Rate", fontsize=11)
    ax1.set_title("Bet Win Rate by League", fontsize=13, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.grid(True, alpha=0.2, axis="x")

    for i, (wr, bets) in enumerate(zip(stats["win_rate"], stats["bets"])):
        ax1.text(wr + 0.005, i, f"{wr:.0%} ({bets})", va="center", fontsize=9)

    # Avg edge by league
    ax2.barh(range(len(stats)), stats["avg_edge"], color="#4C72B0", alpha=0.8)
    ax2.set_yticks(range(len(stats)))
    ax2.set_yticklabels(stats.index, fontsize=10)
    ax2.set_xlabel("Average Edge (EV)", fontsize=11)
    ax2.set_title("Average Perceived Edge by League", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="x")

    for i, edge in enumerate(stats["avg_edge"]):
        ax2.text(edge + 0.002, i, f"{edge:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def bet_outcome_distribution_plot(value_bets: pd.DataFrame, output_path: Path):
    """Plot distribution of bet types and outcomes."""
    if value_bets.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bet outcome distribution (H/D/A)
    outcome_counts = value_bets["bet_outcome"].value_counts()
    outcome_wins = value_bets.groupby("bet_outcome")["won"].sum()
    outcome_labels = {"H": "Home", "D": "Draw", "A": "Away"}

    labels = [outcome_labels.get(o, o) for o in outcome_counts.index]
    total = outcome_counts.values
    wins = [outcome_wins.get(o, 0) for o in outcome_counts.index]
    losses = [t - w for t, w in zip(total, wins)]

    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, wins, width, label="Won", color="#55A868", alpha=0.8)
    ax1.bar(x + width / 2, losses, width, label="Lost", color="#C44E52", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Number of Bets", fontsize=11)
    ax1.set_title("Bet Outcomes by Type", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")

    # Odds distribution of bets
    ax2.hist(value_bets["odds"], bins=30, color="#4C72B0", alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Odds", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Odds Distribution of Value Bets", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def per_league_bankroll_plot(value_bets: pd.DataFrame, output_path: Path):
    """Plot bankroll evolution broken down by league."""
    if value_bets.empty or "league" not in value_bets.columns:
        return

    leagues = sorted(value_bets["league"].unique())
    n_leagues = len(leagues)
    if n_leagues == 0:
        return

    ncols = min(n_leagues, 3)
    nrows = (n_leagues + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, league in enumerate(leagues):
        ax = axes[idx]
        league_bets = value_bets[value_bets["league"] == league].copy()
        sim = simulate_bankroll(league_bets, initial_bankroll=INITIAL_BANKROLL)

        if sim.empty:
            ax.text(0.5, 0.5, "No bets", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.set_title(league, fontsize=12, fontweight="bold")
            continue

        dates = pd.to_datetime(sim["date"])
        bankroll = sim["bankroll"].values

        ax.fill_between(dates, INITIAL_BANKROLL, bankroll,
                        where=bankroll >= INITIAL_BANKROLL,
                        color="#55A868", alpha=0.15, interpolate=True)
        ax.fill_between(dates, INITIAL_BANKROLL, bankroll,
                        where=bankroll < INITIAL_BANKROLL,
                        color="#C44E52", alpha=0.15, interpolate=True)
        ax.plot(dates, bankroll, color="#2C3E50", linewidth=1.5)
        ax.axhline(y=INITIAL_BANKROLL, color="gray", linestyle="--",
                   alpha=0.4, linewidth=1)

        final = bankroll[-1]
        profit = final - INITIAL_BANKROLL
        roi = profit / sim["staked"].sum() if sim["staked"].sum() > 0 else 0
        color = "#55A868" if profit >= 0 else "#C44E52"
        ax.set_title(f"{league}  ({len(league_bets)} bets, ROI: {roi:+.1%})",
                     fontsize=11, fontweight="bold", color=color)
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(axis="x", rotation=30)

    for ax in axes[n_leagues:]:
        ax.set_visible(False)

    plt.suptitle("Bankroll by League", fontsize=15, fontweight="bold")
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
    metrics["win_rate"] = round(total_wins / total_bets, 4) if total_bets > 0 else 0
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

    # --- Plots ---
    bankroll_plot(simulation, output_dir / "bankroll.png")
    roi_by_league_plot(value_bets, output_dir / "roi_by_league.png")
    bet_outcome_distribution_plot(value_bets, output_dir / "bet_outcomes.png")
    per_league_bankroll_plot(value_bets, output_dir / "bankroll_by_league.png")

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
        logger.info("Simulation is empty - no bets were placed.")
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
