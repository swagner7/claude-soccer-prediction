from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FOOTBALL_DATA_DIR = RAW_DIR / "football-data"
ESPN_DIR = RAW_DIR / "espn"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
BETS_DIR = OUTPUTS_DIR / "bets"

# football-data.co.uk configuration
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Super League Greece",
}

# Season codes for football-data.co.uk (last 5 seasons)
SEASONS = ["2021", "2122", "2223", "2324", "2425"]

# League groups
BIG5_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

# Feature engineering
FORM_WINDOWS = [5, 10]
H2H_MAX_LOOKBACK = 10
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_INITIAL_RATING = 1500.0

# Model
RANDOM_STATE = 42
TEST_FRACTION = 0.20
CALIBRATION_FRACTION = 0.20
OPTUNA_N_TRIALS = 50

# Betting
INITIAL_BANKROLL = 1000.0
MIN_EDGE = 0.08
MIN_PROB = 0.15
MAX_ODDS = 4.0
KELLY_FRACTION = 0.15
MAX_SINGLE_BET_PCT = 0.03
MAX_TOTAL_EXPOSURE_PCT = 0.15

# Market shrinkage: blend model prob toward market prob to reduce overconfidence.
# 0.0 = pure model, 1.0 = pure market. 0.3 means 70% model / 30% market.
MARKET_SHRINKAGE = 0.30

# Max divergence from market: skip bets where model prob is more than this far
# from the market-implied prob (filters out the most overconfident predictions).
MAX_MARKET_DIVERGENCE = 0.15

# Use Pinnacle (sharp) odds as the benchmark for edge calculation when available.
# Fall back to average odds if Pinnacle not available.
PREFERRED_ODDS_TYPE = "pin"

# Adjusted probability range: skip bets outside this range.
# Too low = longshots where model is most overconfident.
# Too high = heavy favorites where edge is noise.
MIN_ADJ_PROB = 0.25
MAX_ADJ_PROB = 0.55

# Leagues to exclude from betting (low data quality, illiquid markets).
EXCLUDED_LEAGUES = {"Super League Greece", "Super Lig"}
