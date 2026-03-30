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
MIN_EDGE = 0.05
MIN_PROB = 0.10
MAX_ODDS = 10.0
KELLY_FRACTION = 0.25
MAX_SINGLE_BET_PCT = 0.05
MAX_TOTAL_EXPOSURE_PCT = 0.20
