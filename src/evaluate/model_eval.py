"""Model evaluation: log loss, Brier score, calibration plots, model comparison."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)

from src.config import EVAL_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Home", "Draw", "Away"]

# All possible model names — detect which are present at runtime
ALL_MODEL_NAMES = [
    "logreg", "random_forest", "extra_trees", "gradient_boosting",
    "xgboost", "lightgbm", "mlp",
]
MODEL_LABELS = {
    "logreg": "Logistic Reg.",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Grad. Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
}
_PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#DD8452"]
MODEL_COLORS = {name: _PALETTE[i] for i, name in enumerate(ALL_MODEL_NAMES)}


def _detect_models(predictions: pd.DataFrame) -> list[str]:
    """Return model names that have calibrated probability columns in predictions."""
    return [m for m in ALL_MODEL_NAMES if f"cal_{m}_prob_home" in predictions.columns]


def calibration_plot(
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    output_path: Path,
    n_bins: int = 10,
):
    """Plot reliability diagram per class, showing all models + market."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for class_idx, (ax, class_name) in enumerate(zip(axes, CLASS_NAMES)):
        y_binary = (y_true == class_idx).astype(int)
        outcome = ["home", "draw", "away"][class_idx]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect")

        # Each model (calibrated)
        for model_name in _detect_models(predictions):
            col = f"cal_{model_name}_prob_{outcome}"
            if col not in predictions.columns:
                continue
            probs = predictions[col].values
            _plot_calibration_curve(ax, probs, y_binary, n_bins,
                                   label=MODEL_LABELS.get(model_name, model_name),
                                   color=MODEL_COLORS.get(model_name, None))

        # Market
        market_col = f"market_prob_{outcome}"
        if market_col in predictions.columns:
            market_probs = predictions[market_col].values
            valid = ~np.isnan(market_probs)
            if valid.sum() > 0:
                _plot_calibration_curve(ax, market_probs[valid], y_binary[valid],
                                       n_bins, label="Market", color="#8C8C8C",
                                       linestyle="--")

        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Observed Frequency", fontsize=11)
        ax.set_title(f"{class_name} Win", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Calibration (Reliability) Diagrams", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_calibration_curve(ax, probs, y_binary, n_bins, label, color=None,
                            linestyle="-"):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, freqs = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 5:
            centers.append(probs[mask].mean())
            freqs.append(y_binary[mask].mean())
    if centers:
        ax.plot(centers, freqs, "o-", label=label, color=color, linewidth=1.5,
                markersize=5, linestyle=linestyle)


def model_comparison_bar_chart(comparison: dict, output_path: Path):
    """Bar chart comparing log loss and accuracy across models."""
    model_names = list(comparison["models"].keys())
    raw_ll = [comparison["models"][m]["raw_log_loss"] for m in model_names]
    cal_ll = [comparison["models"][m]["cal_log_loss"] for m in model_names]
    cal_acc = [comparison["models"][m]["cal_accuracy"] for m in model_names]

    labels = [MODEL_LABELS.get(m, m) for m in model_names]
    x = np.arange(len(labels))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log Loss comparison
    bars1 = ax1.bar(x - width / 2, raw_ll, width, label="Raw", alpha=0.7,
                    color="#4C72B0")
    bars2 = ax1.bar(x + width / 2, cal_ll, width, label="Calibrated", alpha=0.7,
                    color="#55A868")

    if comparison.get("market_log_loss"):
        ax1.axhline(y=comparison["market_log_loss"], color="#C44E52", linestyle="--",
                     linewidth=1.5, label=f"Market ({comparison['market_log_loss']:.4f})")

    ax1.set_ylabel("Log Loss (lower is better)", fontsize=11)
    ax1.set_title("Log Loss Comparison", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.4f}",
                     ha="center", va="bottom", fontsize=8)

    # Accuracy comparison
    colors = [MODEL_COLORS.get(m, "#999") for m in model_names]
    bars = ax2.bar(x, cal_acc, width * 1.5, color=colors, alpha=0.8)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Calibrated Model Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.1%}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def feature_importance_plot(output_path: Path, top_n: int = 25):
    """Plot feature importances from all available models."""
    import joblib

    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    importances = {}

    for model_name in ALL_MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}_raw.joblib"
        if not model_path.exists():
            continue
        try:
            model = joblib.load(model_path)
            label = MODEL_LABELS.get(model_name, model_name)
            if hasattr(model, "coef_"):
                importances[label] = np.abs(model.coef_).mean(axis=0)
            elif hasattr(model, "feature_importances_"):
                importances[label] = model.feature_importances_.astype(float)
            elif hasattr(model, "feature_importance"):
                importances[label] = model.feature_importance(
                    importance_type="gain"
                ).astype(float)
        except Exception:
            pass

    if not importances:
        return

    n_models = len(importances)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 10))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, imp) in zip(axes, importances.items()):
        # Normalize to [0, 1] for comparability
        imp_norm = imp / imp.max() if imp.max() > 0 else imp
        n = min(top_n, len(imp_norm))
        idx = np.argsort(imp_norm)[-n:]

        colors = plt.cm.viridis(imp_norm[idx] / imp_norm[idx].max())
        ax.barh(range(len(idx)), imp_norm[idx], color=colors)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_cols[i] for i in idx], fontsize=9)
        ax.set_xlabel("Normalized Importance", fontsize=11)
        ax.set_title(f"{model_name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle(f"Top {top_n} Feature Importances by Model",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def confusion_matrix_plot(y_true, predictions, output_path):
    """Plot confusion matrices for each model side by side."""
    model_names_present = _detect_models(predictions)
    n = len(model_names_present)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, model_name in zip(axes[:n], model_names_present):
        probs = predictions[
            [f"cal_{model_name}_prob_{o}" for o in ["home", "draw", "away"]]
        ].values
        y_pred = np.argmax(probs, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_title(MODEL_LABELS.get(model_name, model_name),
                     fontsize=13, fontweight="bold")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Confusion Matrices (Calibrated Models)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def probability_distribution_plot(predictions, output_path):
    """Plot distribution of predicted probabilities for each model and outcome."""
    model_names_present = _detect_models(predictions)

    fig, axes = plt.subplots(len(model_names_present), 3, figsize=(15, 4 * len(model_names_present)))
    if len(model_names_present) == 1:
        axes = axes.reshape(1, -1)

    for row, model_name in enumerate(model_names_present):
        for col, (outcome, class_name) in enumerate(
            zip(["home", "draw", "away"], CLASS_NAMES)
        ):
            ax = axes[row, col]
            probs = predictions[f"cal_{model_name}_prob_{outcome}"].values
            ax.hist(probs, bins=50, alpha=0.7, color=MODEL_COLORS.get(model_name, "#999"),
                    edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Predicted Probability", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            if row == 0:
                ax.set_title(f"{class_name}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    f"{MODEL_LABELS.get(model_name, model_name)}\nCount",
                    fontsize=10,
                )
            ax.grid(True, alpha=0.2)

    plt.suptitle("Predicted Probability Distributions",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(predictions: pd.DataFrame, output_dir: Path) -> dict:
    """Run full model evaluation."""
    y_true = predictions["result_code"].values.astype(int)
    y_prob = predictions[["prob_home", "prob_draw", "prob_away"]].values
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {}

    # Best model metrics
    metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    for i, name in enumerate(CLASS_NAMES):
        y_binary = (y_true == i).astype(int)
        metrics[f"brier_{name.lower()}"] = float(
            brier_score_loss(y_binary, y_prob[:, i])
        )
    metrics["brier_avg"] = np.mean(
        [metrics[f"brier_{n.lower()}"] for n in CLASS_NAMES]
    )

    # Per-model metrics
    model_metrics = {}
    for model_name in _detect_models(predictions):
        cal_cols = [f"cal_{model_name}_prob_{o}" for o in ["home", "draw", "away"]]
        if not all(c in predictions.columns for c in cal_cols):
            continue
        cal_probs = predictions[cal_cols].values
        model_metrics[model_name] = {
            "cal_log_loss": float(log_loss(y_true, cal_probs, labels=[0, 1, 2])),
            "cal_accuracy": float(
                accuracy_score(y_true, np.argmax(cal_probs, axis=1))
            ),
        }

        raw_cols = [f"raw_{model_name}_prob_{o}" for o in ["home", "draw", "away"]]
        if all(c in predictions.columns for c in raw_cols):
            raw_probs = predictions[raw_cols].values
            model_metrics[model_name]["raw_log_loss"] = float(
                log_loss(y_true, raw_probs, labels=[0, 1, 2])
            )

    metrics["per_model"] = model_metrics

    # Market benchmark
    market_cols = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    if all(c in predictions.columns for c in market_cols):
        market_probs = predictions[market_cols].values
        valid = ~np.isnan(market_probs).any(axis=1)
        if valid.sum() > 0:
            metrics["market_log_loss"] = float(
                log_loss(y_true[valid], market_probs[valid], labels=[0, 1, 2])
            )

    # --- Plots ---
    calibration_plot(y_true, predictions, output_dir / "calibration.png")
    confusion_matrix_plot(y_true, predictions, output_dir / "confusion_matrix.png")
    feature_importance_plot(output_dir / "feature_importance.png")
    probability_distribution_plot(predictions, output_dir / "probability_distributions.png")

    # Model comparison chart
    comparison_path = MODELS_DIR / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)
        model_comparison_bar_chart(comparison, output_dir / "model_comparison.png")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(MODELS_DIR / "test_predictions.parquet")
    logger.info(f"Evaluating {len(predictions)} test predictions")

    metrics = evaluate(predictions, EVAL_DIR)

    logger.info("\n=== Model Evaluation (Best Model) ===")
    for k, v in metrics.items():
        if k == "per_model":
            continue
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    if "per_model" in metrics:
        logger.info("\n=== Per-Model Comparison ===")
        for name, m in sorted(metrics["per_model"].items(),
                               key=lambda x: x[1]["cal_log_loss"]):
            label = MODEL_LABELS.get(name, name)
            parts = [f"cal_ll={m['cal_log_loss']:.4f}"]
            if "raw_log_loss" in m:
                parts.append(f"raw_ll={m['raw_log_loss']:.4f}")
            parts.append(f"acc={m['cal_accuracy']:.4f}")
            logger.info(f"  {label:25s} {', '.join(parts)}")


if __name__ == "__main__":
    main()
