"""Closing Line Value (CLV) analysis.

CLV measures whether the model identifies genuine edge by comparing:
- The odds at which we'd place bets (opening / Pinnacle line)
- The market's closing implied probability (average odds = proxy for true prob)

If the model consistently picks bets where the closing probability > implied prob
from the odds we bet at, that's strong evidence of real edge. Professional bettors
consider CLV the single most important metric — more predictive than ROI over
small samples.

We also analyse:
- CLV by edge bucket (does higher model edge → higher CLV?)
- CLV by league (where does the model genuinely beat the market?)
- CLV vs actual ROI scatterplot (are CLV+ bets actually profitable?)
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_clv(value_bets: pd.DataFrame) -> pd.DataFrame:
    """Add CLV columns to value bets DataFrame.

    CLV is defined as:
        clv = closing_prob / implied_prob_at_bet - 1

    Where:
    - implied_prob_at_bet = 1 / odds (the Pinnacle line we bet at)
    - closing_prob = market_prob (average market, overround-removed = best estimate of true prob)

    Positive CLV means we got better odds than the market's final assessment.
    """
    bets = value_bets.copy()

    if "market_prob" not in bets.columns or "implied_prob" not in bets.columns:
        logger.warning("Missing market_prob or implied_prob columns for CLV calculation")
        return bets

    # CLV: did we bet at better odds than the closing line suggests?
    # market_prob is closing line proxy (avg market, overround-removed)
    # implied_prob = 1/odds is the line we bet at
    bets["clv"] = bets["market_prob"] / bets["implied_prob"] - 1.0

    # CLV in odds space: what odds would closing prob imply?
    bets["closing_odds"] = 1.0 / bets["market_prob"].where(bets["market_prob"] > 0, np.nan)
    bets["clv_odds"] = bets["odds"] - bets["closing_odds"]

    return bets


def clv_summary(bets: pd.DataFrame) -> dict:
    """Compute summary CLV statistics."""
    if "clv" not in bets.columns:
        return {}

    valid = bets.dropna(subset=["clv"])
    if valid.empty:
        return {}

    clv_positive = valid["clv"] > 0
    won = valid["won"].astype(bool)

    stats = {
        "n_bets": len(valid),
        "mean_clv": round(float(valid["clv"].mean()), 4),
        "median_clv": round(float(valid["clv"].median()), 4),
        "clv_positive_pct": round(float(clv_positive.mean()), 4),
        "mean_clv_when_positive": round(float(valid.loc[clv_positive, "clv"].mean()), 4) if clv_positive.any() else None,
        "mean_clv_when_negative": round(float(valid.loc[~clv_positive, "clv"].mean()), 4) if (~clv_positive).any() else None,
        "roi_clv_positive": None,
        "roi_clv_negative": None,
        "win_rate_clv_positive": None,
        "win_rate_clv_negative": None,
    }

    # ROI and win rate split by CLV sign
    for label, mask in [("positive", clv_positive), ("negative", ~clv_positive)]:
        subset = valid[mask]
        if len(subset) > 0:
            # ROI = (wins * (odds-1) - losses) / n_bets ... simplified
            pnl = subset.apply(
                lambda r: (r["odds"] - 1) if r["won"] else -1.0, axis=1
            )
            stats[f"roi_clv_{label}"] = round(float(pnl.mean()), 4)
            stats[f"win_rate_clv_{label}"] = round(float(subset["won"].mean()), 4)
            stats[f"n_bets_clv_{label}"] = int(len(subset))

    return stats


def clv_by_edge_bucket(bets: pd.DataFrame) -> pd.DataFrame:
    """Break down CLV by model edge bucket."""
    if "clv" not in bets.columns or "edge" not in bets.columns:
        return pd.DataFrame()

    valid = bets.dropna(subset=["clv", "edge"])
    if valid.empty:
        return pd.DataFrame()

    # Create edge buckets
    bins = [0, 0.10, 0.15, 0.20, 0.30, 1.0]
    labels = ["8-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    valid = valid.copy()
    valid["edge_bucket"] = pd.cut(valid["edge"], bins=bins, labels=labels, right=False)

    result = valid.groupby("edge_bucket", observed=True).agg(
        n_bets=("clv", "count"),
        mean_clv=("clv", "mean"),
        clv_positive_pct=("clv", lambda x: (x > 0).mean()),
        mean_edge=("edge", "mean"),
        win_rate=("won", "mean"),
        avg_odds=("odds", "mean"),
    ).round(4)

    return result


def clv_by_league(bets: pd.DataFrame) -> pd.DataFrame:
    """Break down CLV by league."""
    if "clv" not in bets.columns or "league" not in bets.columns:
        return pd.DataFrame()

    valid = bets.dropna(subset=["clv"])
    if valid.empty:
        return pd.DataFrame()

    result = valid.groupby("league").agg(
        n_bets=("clv", "count"),
        mean_clv=("clv", "mean"),
        median_clv=("clv", "median"),
        clv_positive_pct=("clv", lambda x: (x > 0).mean()),
        win_rate=("won", "mean"),
        avg_odds=("odds", "mean"),
        avg_edge=("edge", "mean"),
    ).sort_values("mean_clv", ascending=False).round(4)

    # Compute flat-stake ROI per league
    for league in result.index:
        subset = valid[valid["league"] == league]
        pnl = subset.apply(
            lambda r: (r["odds"] - 1) if r["won"] else -1.0, axis=1
        )
        result.loc[league, "flat_roi"] = round(pnl.mean(), 4)

    return result


def clv_by_outcome(bets: pd.DataFrame) -> pd.DataFrame:
    """Break down CLV by bet type (H/D/A)."""
    if "clv" not in bets.columns or "bet_outcome" not in bets.columns:
        return pd.DataFrame()

    valid = bets.dropna(subset=["clv"])
    outcome_labels = {"H": "Home", "D": "Draw", "A": "Away"}

    result = valid.groupby("bet_outcome").agg(
        n_bets=("clv", "count"),
        mean_clv=("clv", "mean"),
        clv_positive_pct=("clv", lambda x: (x > 0).mean()),
        win_rate=("won", "mean"),
        avg_odds=("odds", "mean"),
    ).round(4)

    result.index = result.index.map(lambda x: outcome_labels.get(x, x))
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def clv_distribution_plot(bets: pd.DataFrame, output_path: Path):
    """Plot CLV distribution with win/loss overlay."""
    if "clv" not in bets.columns:
        return

    valid = bets.dropna(subset=["clv"])
    if valid.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of CLV
    ax = axes[0]
    clv_vals = valid["clv"].clip(-0.5, 0.5)
    ax.hist(clv_vals, bins=40, color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
    ax.axvline(x=valid["clv"].mean(), color="#55A868", linestyle="-", linewidth=2,
               label=f"Mean CLV: {valid['clv'].mean():.1%}")
    ax.set_xlabel("CLV (Closing Line Value)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("CLV Distribution", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # CLV vs actual outcome
    ax = axes[1]
    won_mask = valid["won"].astype(bool)
    ax.hist(valid.loc[won_mask, "clv"].clip(-0.5, 0.5), bins=30, alpha=0.6,
            color="#55A868", label="Won", edgecolor="white", linewidth=0.5)
    ax.hist(valid.loc[~won_mask, "clv"].clip(-0.5, 0.5), bins=30, alpha=0.6,
            color="#C44E52", label="Lost", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("CLV", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("CLV by Bet Outcome", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def clv_edge_scatter_plot(bets: pd.DataFrame, output_path: Path):
    """Scatter plot: model edge vs CLV, colored by win/loss."""
    if "clv" not in bets.columns or "edge" not in bets.columns:
        return

    valid = bets.dropna(subset=["clv", "edge"])
    if valid.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Edge vs CLV scatter
    ax = axes[0]
    won_mask = valid["won"].astype(bool)
    ax.scatter(valid.loc[won_mask, "edge"], valid.loc[won_mask, "clv"],
               alpha=0.3, s=15, color="#55A868", label="Won")
    ax.scatter(valid.loc[~won_mask, "edge"], valid.loc[~won_mask, "clv"],
               alpha=0.3, s=15, color="#C44E52", label="Lost")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Add trend line
    z = np.polyfit(valid["edge"], valid["clv"], 1)
    p = np.poly1d(z)
    edge_range = np.linspace(valid["edge"].min(), valid["edge"].max(), 100)
    ax.plot(edge_range, p(edge_range), color="#4C72B0", linewidth=2,
            label=f"Trend (slope={z[0]:.2f})")

    ax.set_xlabel("Model Edge", fontsize=11)
    ax.set_ylabel("CLV", fontsize=11)
    ax.set_title("Model Edge vs CLV", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # CLV by edge bucket (bar chart)
    ax = axes[1]
    by_edge = clv_by_edge_bucket(valid)
    if not by_edge.empty:
        x = np.arange(len(by_edge))
        colors = ["#55A868" if v > 0 else "#C44E52" for v in by_edge["mean_clv"]]
        bars = ax.bar(x, by_edge["mean_clv"], color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(by_edge.index, fontsize=10)
        ax.set_xlabel("Model Edge Bucket", fontsize=11)
        ax.set_ylabel("Mean CLV", fontsize=11)
        ax.set_title("Mean CLV by Edge Bucket", fontsize=13, fontweight="bold")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.2, axis="y")

        for i, (v, n) in enumerate(zip(by_edge["mean_clv"], by_edge["n_bets"])):
            ax.text(i, v + 0.002 if v >= 0 else v - 0.008,
                    f"{v:.1%}\n(n={n})", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def clv_league_plot(bets: pd.DataFrame, output_path: Path):
    """CLV and flat-stake ROI by league comparison."""
    by_league = clv_by_league(bets)
    if by_league.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Mean CLV by league
    ax = axes[0]
    sorted_leagues = by_league.sort_values("mean_clv")
    colors = ["#55A868" if v > 0 else "#C44E52" for v in sorted_leagues["mean_clv"]]
    y_pos = range(len(sorted_leagues))
    ax.barh(y_pos, sorted_leagues["mean_clv"], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_leagues.index, fontsize=10)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean CLV", fontsize=11)
    ax.set_title("Mean CLV by League", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.2, axis="x")
    for i, (clv, n) in enumerate(zip(sorted_leagues["mean_clv"], sorted_leagues["n_bets"])):
        ax.text(clv + 0.003 if clv >= 0 else clv - 0.003, i,
                f"{clv:.1%} (n={n})", va="center", fontsize=9,
                ha="left" if clv >= 0 else "right")

    # CLV vs flat-stake ROI scatter
    ax = axes[1]
    if "flat_roi" in by_league.columns:
        for _, row in by_league.iterrows():
            color = "#55A868" if row["flat_roi"] > 0 else "#C44E52"
            ax.scatter(row["mean_clv"], row["flat_roi"],
                       s=row["n_bets"] * 2, color=color, alpha=0.7, edgecolors="black",
                       linewidth=0.5)
            ax.annotate(row.name, (row["mean_clv"], row["flat_roi"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)

        # Quadrant lines
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Mean CLV", fontsize=11)
        ax.set_ylabel("Flat-Stake ROI", fontsize=11)
        ax.set_title("CLV vs Actual ROI by League", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.2)

        # Quadrant labels
        ax.text(0.95, 0.95, "True Edge ✓", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#55A868", fontweight="bold")
        ax.text(0.05, 0.95, "Lucky", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color="#D4A017")
        ax.text(0.95, 0.05, "Unlucky", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color="#D4A017")
        ax.text(0.05, 0.05, "No Edge ✗", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10, color="#C44E52", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_clv_analysis(value_bets: pd.DataFrame, output_dir: Path) -> dict:
    """Run full CLV analysis and generate all plots.

    Returns a summary dict of CLV metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute CLV
    bets = compute_clv(value_bets)

    if "clv" not in bets.columns:
        logger.warning("Could not compute CLV — missing market_prob or implied_prob")
        return {}

    # Summary stats
    summary = clv_summary(bets)

    # Breakdowns
    by_edge = clv_by_edge_bucket(bets)
    by_league = clv_by_league(bets)
    by_outcome = clv_by_outcome(bets)

    # Log results
    logger.info("\n=== CLV Analysis ===")
    logger.info(f"  Mean CLV: {summary.get('mean_clv', 0):.2%}")
    logger.info(f"  Median CLV: {summary.get('median_clv', 0):.2%}")
    logger.info(f"  CLV-positive bets: {summary.get('clv_positive_pct', 0):.1%}")

    if summary.get("roi_clv_positive") is not None:
        logger.info(f"  ROI on CLV+ bets: {summary['roi_clv_positive']:.1%} "
                     f"({summary.get('n_bets_clv_positive', 0)} bets)")
    if summary.get("roi_clv_negative") is not None:
        logger.info(f"  ROI on CLV- bets: {summary['roi_clv_negative']:.1%} "
                     f"({summary.get('n_bets_clv_negative', 0)} bets)")

    if not by_edge.empty:
        logger.info("\n  CLV by edge bucket:")
        for bucket, row in by_edge.iterrows():
            logger.info(f"    {bucket}: CLV={row['mean_clv']:.2%}, "
                        f"win_rate={row['win_rate']:.1%}, n={int(row['n_bets'])}")

    if not by_league.empty:
        logger.info("\n  CLV by league:")
        for league, row in by_league.iterrows():
            logger.info(f"    {league}: CLV={row['mean_clv']:.2%}, "
                        f"ROI={row.get('flat_roi', 0):.1%}, n={int(row['n_bets'])}")

    if not by_outcome.empty:
        logger.info("\n  CLV by bet type:")
        for outcome, row in by_outcome.iterrows():
            logger.info(f"    {outcome}: CLV={row['mean_clv']:.2%}, "
                        f"win_rate={row['win_rate']:.1%}, n={int(row['n_bets'])}")

    # Generate plots
    clv_distribution_plot(bets, output_dir / "clv_distribution.png")
    clv_edge_scatter_plot(bets, output_dir / "clv_vs_edge.png")
    clv_league_plot(bets, output_dir / "clv_by_league.png")

    # Save full results
    full_results = {
        "summary": summary,
        "by_edge_bucket": by_edge.to_dict(orient="index") if not by_edge.empty else {},
        "by_league": by_league.to_dict(orient="index") if not by_league.empty else {},
        "by_outcome": by_outcome.to_dict(orient="index") if not by_outcome.empty else {},
    }
    with open(output_dir / "clv_analysis.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    return summary
