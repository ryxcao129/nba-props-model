"""Shared paths and settings for the NBA props project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for p in (DATA_RAW, DATA_PROC, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

# Stats we model (one target column each in the game logs)
TARGETS = ["PTS", "REB", "AST", "FG3M"]

# Box-score columns used to build rolling form features
FORM_COLS = ["PTS", "REB", "AST", "FG3M", "MIN", "FGA", "FG3A", "FTA", "TOV"]
WINDOWS = [5, 10, 20]

# Opponent defence: what each team ALLOWS per game
OPP_COLS = ["PTS", "REB", "AST", "FG3M", "FGA"]

MIN_MINUTES = 10        # ignore garbage-time appearances when training
MIN_PRIOR_GAMES = 5     # need some history before we trust the features
