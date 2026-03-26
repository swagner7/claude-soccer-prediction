"""Model evaluation: log loss, Brier score, calibration plots."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: Path,
    n_bins: int = 10,
):
    """Plot reliability diagram for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (ax, name) in enumerate(zip(axes, class_names)):
        y_binary = (y_true == i).astype(int)
        probs = y_prob[:, i]

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        observed_freq = []
        bin_counts = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_centers.append(probs[mask].mean())
                observed_freq.append(y_binary[mask].mean())
                bin_counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(bin_centers, observed_freq, "o-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{name}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Calibration Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def feature_importance_plot(
    model, feature_names: list[str], output_path: Path, top_n: int = 25
):
    """Plot top feature importances from model coefficients."""
    # For logistic regression, use mean absolute coefficient across classes
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    elif hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        return

    n = min(top_n, len(importance))
    idx = np.argsort(importance)[-n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importance[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance (mean |coefficient|)")
    ax.set_title(f"Top {n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def confusion_matrix_plot(y_true, y_pred, class_names, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(predictions: pd.DataFrame, output_dir: Path) -> dict:
    """Run full model evaluation."""
    y_true = predictions["result_code"].values.astype(int)
    y_prob = predictions[["prob_home", "prob_draw", "prob_away"]].values
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {}

    # Log loss
    metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Brier score per class
    for i, name in enumerate(CLASS_NAMES):
        y_binary = (y_true == i).astype(int)
        metrics[f"brier_{name.lower()}"] = float(
            brier_score_loss(y_binary, y_prob[:, i])
        )
    metrics["brier_avg"] = np.mean(
        [metrics[f"brier_{n.lower()}"] for n in CLASS_NAMES]
    )

    # Baseline comparison
    baseline_cols = ["baseline_prob_home", "baseline_prob_draw", "baseline_prob_away"]
    if all(c in predictions.columns for c in baseline_cols):
        baseline_probs = predictions[baseline_cols].values
        metrics["baseline_log_loss"] = float(
            log_loss(y_true, baseline_probs, labels=[0, 1, 2])
        )
        metrics["baseline_accuracy"] = float(
            accuracy_score(y_true, np.argmax(baseline_probs, axis=1))
        )

    # Market benchmark
    market_cols = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    if all(c in predictions.columns for c in market_cols):
        market_probs = predictions[market_cols].values
        valid = ~np.isnan(market_probs).any(axis=1)
        if valid.sum() > 0:
            metrics["market_log_loss"] = float(
                log_loss(y_true[valid], market_probs[valid], labels=[0, 1, 2])
            )

    # Plots
    calibration_plot(y_true, y_prob, CLASS_NAMES, output_dir / "calibration.png")
    confusion_matrix_plot(
        y_true, y_pred, CLASS_NAMES, output_dir / "confusion_matrix.png"
    )

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(MODELS_DIR / "test_predictions.parquet")
    logger.info(f"Evaluating {len(predictions)} test predictions")

    # Load model for feature importance
    import joblib
    model, scaler = joblib.load(MODELS_DIR / "logreg_v1.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    feature_importance_plot(
        model, feature_cols, EVAL_DIR / "feature_importance.png"
    )

    metrics = evaluate(predictions, EVAL_DIR)

    logger.info("\n=== Model Evaluation ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
