# Soccer Match Prediction & Betting System

An end-to-end ML pipeline that predicts soccer match outcomes (Home/Draw/Away) and identifies value betting opportunities across 11 European leagues.

## Overview

The system downloads historical match data with real bookmaker odds, engineers 108 features, screens 7 model types, tunes the best with Optuna, calibrates probabilities, then simulates a betting strategy using Kelly criterion sizing.

## Data Sources

### Football-Data.co.uk (primary)

CSV files for 11 leagues across 5 seasons (2020/21 through 2024/25):

| Code | League |
|------|--------|
| E0 | Premier League |
| E1 | Championship |
| SP1 | La Liga |
| D1 | Bundesliga |
| I1 | Serie A |
| F1 | Ligue 1 |
| N1 | Eredivisie |
| B1 | Belgian Pro League |
| P1 | Primeira Liga |
| T1 | Super Lig |
| G1 | Super League Greece |

Each CSV contains match results, goals, shots, corners, fouls, cards, and betting odds from multiple bookmakers (Bet365, Pinnacle, market average, etc.).

### Understat (xG data)

Match-level expected goals (xG) data for the top 5 European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1), going back to the 2020/21 season. Provides xG for and against per team per match, plus Understat's pre-match win/draw/loss forecast probabilities. Downloaded via the `understatapi` Python package.

### ESPN/Kaggle (supplementary)

Optional dataset from `excel4soccer/espn-soccer-data`. Requires Kaggle API credentials. The system works without it.

## Features (126 total)

All features use only pre-match information to avoid data leakage.

### Elo Ratings
- Home/away Elo ratings and differential (K=20, home advantage=100)

### Team Form (rolling windows of 5 and 10 matches)
- Points per game, goal difference, goals scored/conceded
- Clean sheet percentage, venue-specific form

### Head-to-Head
- Win rates, draw rates, average goals from last 10 meetings

### Match Statistics (rolling averages)
- Shots, shots on target, corners, fouls, yellow cards (for and against, windows of 5 and 10)

### Scoring Patterns
- BTTS percentage, over 2.5 goals percentage, average total goals
- Win streak, unbeaten streak, scoring streak

### League Position Proxy
- Season points, goal difference, points per game, PPG differential

### Goal Supremacy
- Rolling average goal supremacy over 3, 5, and 10 matches

### Rest Days
- Days since each team's last match, rest differential

### Expected Goals (xG) -- from Understat
Available for the top 5 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1). NaN for other leagues.
- Rolling average xG for and against (windows of 5 and 10 matches)
- Rolling xG differential (attack strength vs defense weakness)
- xG overperformance (goals scored minus xG -- measures finishing quality/luck)
- xG differential spread between home and away teams

### Other
- League code (label encoded)

## Model Pipeline

### Phase 1: Screening

Seven model types are trained with default hyperparameters and evaluated on calibration-set log loss:

| Model | Implementation |
|-------|---------------|
| Logistic Regression | sklearn |
| Random Forest | sklearn |
| Extra Trees | sklearn |
| Gradient Boosting | sklearn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| MLP (Neural Network) | sklearn |

### Phase 2: Optuna Tuning

The screening winner is tuned with 50 Optuna trials using 3-fold time-series cross-validation. Each model type has a registered hyperparameter search space.

### Phase 3: Calibration

All models (including the tuned winner) are calibrated using per-class isotonic regression on a held-out calibration set, with row-sum normalization to ensure probabilities sum to 1.

### Data Split

Temporal (chronological) split to prevent data leakage:
- **Train** (60%): earliest matches
- **Calibration** (20%): middle period
- **Test** (20%): most recent matches

## Betting Strategy

### Value Bet Identification

A bet is placed when the model's estimated probability gives positive expected value against the bookmaker odds:

```
edge = model_probability * decimal_odds - 1
```

A bet qualifies when `edge > 0.05` (5% minimum edge), `model_probability > 0.10`, and `odds < 10.0`.

### Kelly Criterion Sizing

Quarter-Kelly fraction with risk controls:
- **Kelly fraction**: 0.25 (conservative)
- **Per-bet cap**: 5% of bankroll
- **Total exposure cap**: 20% of bankroll per matchday

### Simulation

Bankroll simulation processes bets chronologically, grouped by matchday, starting from $1,000.

## Running the Pipeline

### Prerequisites

- Python 3.12+
- [Task](https://taskfile.dev/) (go-task) for pipeline orchestration

### Quick Start

```bash
# Run the full pipeline end-to-end
task all
```

### Individual Steps

```bash
task setup              # Create venv and install dependencies
task download           # Download match data from football-data.co.uk
task parse              # Parse CSVs into cleaned parquet files
task features           # Build feature matrix (108 features)
task train              # Screen 7 models, tune best with Optuna, calibrate
task evaluate           # Model evaluation metrics and plots
task bet                # Find value bets and simulate bankroll
task evaluate-betting   # Betting performance metrics and plots
task clean              # Remove all generated files
task test               # Run unit tests
```

Dependencies are handled automatically -- running `task train` will trigger `setup`, `download`, `parse`, and `features` if they haven't been run yet.

## Output Files

### `outputs/evaluation/`

| File | Description |
|------|-------------|
| `metrics.json` | Log loss, accuracy, Brier scores for all models + market benchmark |
| `calibration.png` | Reliability diagrams per class (Home/Draw/Away) for all models |
| `confusion_matrix.png` | Confusion matrices for each calibrated model |
| `feature_importance.png` | Top 25 features by importance for each model |
| `model_comparison.png` | Bar charts comparing log loss and accuracy across models |
| `probability_distributions.png` | Histograms of predicted probabilities per model and outcome |

### `outputs/bets/`

| File | Description |
|------|-------------|
| `value_bets.csv` | All identified value bets with edge, odds, stake, and outcome |
| `value_bets_max.csv` | Value bets using best available (max) odds |
| `simulation.csv` | Daily bankroll evolution with drawdown and P&L |

### `outputs/evaluation/` (betting)

| File | Description |
|------|-------------|
| `betting_metrics.json` | ROI, win rate, Sharpe ratio, max drawdown, per-league breakdown |
| `bankroll.png` | Bankroll evolution with drawdown and daily P&L (3-panel) |
| `bankroll_by_league.png` | Bankroll evolution broken down by each league |
| `roi_by_league.png` | Win rate and average edge by league |
| `bet_outcomes.png` | Bet type distribution and odds histogram |

### `models/`

| File | Description |
|------|-------------|
| `*_raw.joblib` | Raw (uncalibrated) model objects |
| `*_calibrated.joblib` | Calibrated model objects |
| `best_model.joblib` | The final best calibrated model used for betting |
| `scaler.joblib` | StandardScaler fitted on training data |
| `feature_cols.joblib` | Ordered list of feature column names |
| `impute_values.joblib` | Column means used for NaN imputation |
| `test_predictions.parquet` | Full test set with all model predictions and odds |
| `model_comparison.json` | Screening and calibration results for all models |

## Interpreting Results

### Model Quality

The key metric is **calibrated log loss** -- lower is better. The market benchmark (derived from bookmaker odds with overround removed) typically sits around 0.97. A model that beats the market on log loss has learned something the bookmakers haven't priced in.

In practice, beating the market is very difficult. The model's best calibrated log loss is typically around 0.99, trailing the market by ~0.02. This gap reflects the efficiency of betting markets.

### Calibration Plots

The reliability diagrams show how well predicted probabilities match observed frequencies. Points on the diagonal indicate perfect calibration. Points above the diagonal mean the model underestimates that probability range; below means overestimation.

### Betting Performance

- **ROI**: Total profit divided by total staked. Positive ROI means the strategy is profitable.
- **Win Rate**: Fraction of value bets that won. Expected to be well below 50% since value bets tend to be on higher-odds outcomes.
- **Max Drawdown**: Largest peak-to-trough decline as a fraction of peak bankroll. Important for risk management.
- **Sharpe Ratio**: Risk-adjusted return (annualized). Above 1.0 is considered good.
- **Per-league breakdown**: Some leagues may be more predictable than others due to differences in competitiveness, data quality, or market efficiency.

## Project Structure

```
claude-soccer-prediction/
├── Taskfile.yml                  # Pipeline orchestration
├── pyproject.toml                # Dependencies
├── src/
│   ├── config.py                 # All configuration constants
│   ├── data/
│   │   ├── download.py           # Data acquisition
│   │   ├── parse.py              # CSV parsing and schema normalization
│   │   └── clean.py              # Validation, dedup, encoding
│   ├── features/
│   │   ├── build.py              # Feature orchestration
│   │   ├── elo.py                # Elo rating system
│   │   ├── team_form.py          # Rolling form features
│   │   ├── head_to_head.py       # H2H statistics
│   │   └── match_stats.py        # Rolling match stat averages
│   ├── model/
│   │   ├── train.py              # Multi-model training + Optuna tuning
│   │   ├── calibrate.py          # Isotonic calibration
│   │   ├── baseline.py           # Logistic regression wrapper
│   │   └── predict.py            # Prediction interface
│   ├── betting/
│   │   ├── value.py              # Value bet identification (EV-based)
│   │   ├── kelly.py              # Kelly criterion bet sizing
│   │   └── simulate.py           # Bankroll simulation
│   └── evaluate/
│       ├── model_eval.py         # Model metrics and visualizations
│       └── betting_eval.py       # Betting metrics and visualizations
├── data/                         # Generated data (gitignored)
├── models/                       # Trained models (gitignored)
└── outputs/                      # Evaluation outputs (gitignored)
```

## Configuration

All tuneable parameters are in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEASONS` | Last 5 | Which seasons to download |
| `FORM_WINDOWS` | [5, 10] | Rolling window sizes for form features |
| `ELO_K_FACTOR` | 20 | Elo rating sensitivity |
| `TEST_FRACTION` | 0.20 | Fraction of data for test set |
| `CALIBRATION_FRACTION` | 0.20 | Fraction of data for calibration set |
| `OPTUNA_N_TRIALS` | 50 | Number of hyperparameter tuning trials |
| `INITIAL_BANKROLL` | 1000 | Starting bankroll for simulation ($) |
| `MIN_EDGE` | 0.05 | Minimum EV edge to place a bet |
| `KELLY_FRACTION` | 0.25 | Kelly criterion fraction (quarter-Kelly) |
| `MAX_SINGLE_BET_PCT` | 0.05 | Max stake as fraction of bankroll |
| `MAX_TOTAL_EXPOSURE_PCT` | 0.20 | Max total exposure per matchday |
