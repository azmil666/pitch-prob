import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.data.clean_data import build_clean_dataset
from src.models.score_model import DixonColesModel
from src.models.scorer_model import GoalScorerModel
from src.predictions.tracker import grade_completed_predictions, log_upcoming_matchweek

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY")
RAW_DIR = ROOT_DIR / "data/raw"
PROCESSED_DIR = ROOT_DIR / "data/processed"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"


def get_current_pl_season() -> int:
    """European football seasons start in August (e.g., Nov 2026 -> 2026 season; Mar 2027 -> 2026 season)."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def refresh_matchweek_data():
    if not FOOTBALL_DATA_KEY:
        raise RuntimeError("FOOTBALL_DATA_ORG_KEY not found in .env")

    season = get_current_pl_season()
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    print(f"=== Refreshing Premier League Season {season}/{season + 1} ===")

    # 1. Refresh current season matches (completed + upcoming)
    matches_url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
    res_m = requests.get(matches_url, headers=headers, timeout=15)
    res_m.raise_for_status()
    with open(RAW_DIR / f"pl_matches_fd_{season}.json", "w") as f:
        json.dump(res_m.json(), f, indent=2)
    print(f"[1/5] Updated pl_matches_fd_{season}.json")

    time.sleep(6)  # Respect 10 req/min rate limit

    # 2. Refresh current season top scorers
    scorers_url = f"https://api.football-data.org/v4/competitions/PL/scorers?season={season}&limit=100"
    res_s = requests.get(scorers_url, headers=headers, timeout=15)
    res_s.raise_for_status()
    with open(RAW_DIR / f"pl_scorers_{season}.json", "w") as f:
        json.dump(res_s.json(), f, indent=2)
    print(f"[2/5] Updated pl_scorers_{season}.json")

    # 3. Rebuild clean finished/upcoming match CSVs
    print("[3/5] Rebuilding clean match tables...")
    build_clean_dataset()

    # 4. Retrain Dixon-Coles & rebuild player table
    print("[4/5] Retraining Dixon-Coles & updating player projections...")
    finished_df = pd.read_csv(PROCESSED_DIR / "matches_finished.csv")
    dc_model = DixonColesModel(xi=0.0018, reg_strength=1.5).fit(finished_df)
    dc_model.save(SAVED_MODELS_DIR / "dixon_coles.joblib")

    scorer_model = GoalScorerModel()
    scorer_model.build_player_table()

    print("[5/5] Grading completed predictions & locking next matchweek forecasts...")
    grade_completed_predictions()
    log_upcoming_matchweek(days_ahead=14)

    print("\nRefresh complete! All models and datasets are up to date.")


if __name__ == "__main__":
    refresh_matchweek_data()