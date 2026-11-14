import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.predictions.predict import PitchProbPredictor

RAW_DIR = ROOT_DIR / "data/raw"
PROCESSED_DIR = ROOT_DIR / "data/processed"
LOG_PATH = PROCESSED_DIR / "prediction_log.csv"

LOG_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "logged_at",
    "status",
    "pred_home_prob",
    "pred_draw_prob",
    "pred_away_prob",
    "pred_outcome",
    "pred_scoreline",
    "pred_xg_home",
    "pred_xg_away",
    "pred_top_scorer",
    "pred_scorer_prob",
    "actual_home_goals",
    "actual_away_goals",
    "actual_result",
    "outcome_hit",
    "scoreline_hit",
    "match_log_loss",
]


def _make_match_id(date_str: str, home: str, away: str) -> str:
    day = pd.to_datetime(date_str).strftime("%Y-%m-%d")
    return f"{day}_{home}_vs_{away}"


def log_upcoming_matchweek(days_ahead: int = 8):
    """Snapshot pre-match predictions for fixtures kicking off in the next `days_ahead` days."""
    upcoming_path = PROCESSED_DIR / "matches_upcoming.csv"
    if not upcoming_path.exists():
        print("No upcoming matches found.")
        return

    upcoming = pd.read_csv(upcoming_path)
    upcoming["date"] = pd.to_datetime(upcoming["date"], utc=True)

    now = datetime.now(timezone.utc)
    window_end = now + timedelta(days=days_ahead)
    target_fixtures = upcoming[(upcoming["date"] >= now) & (upcoming["date"] <= window_end)]

    if LOG_PATH.exists():
        log_df = pd.read_csv(LOG_PATH)
    else:
        log_df = pd.DataFrame(columns=LOG_COLUMNS)

    existing_ids = set(log_df["match_id"].tolist())
    predictor = PitchProbPredictor()
    new_rows = []

    for _, row in target_fixtures.iterrows():
        mid = _make_match_id(row["date"], row["home_team"], row["away_team"])
        if mid in existing_ids:
            continue  # Never overwrite an already-locked pre-match prediction

        pred = predictor.predict_fixture(row["home_team"], row["away_team"], top_scorers_n=1)
        probs = pred["probs"]
        outcome_map = {"H": probs["home_win"], "D": probs["draw"], "A": probs["away_win"]}
        best_outcome = max(outcome_map, key=outcome_map.get)
        top_score = pred["top_scorelines"][0]["score"]
        top_scorer = pred["top_scorers"].iloc[0]

        new_rows.append({
            "match_id": mid,
            "date": row["date"].isoformat(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "logged_at": now.isoformat(),
            "status": "PENDING",
            "pred_home_prob": round(probs["home_win"], 4),
            "pred_draw_prob": round(probs["draw"], 4),
            "pred_away_prob": round(probs["away_win"], 4),
            "pred_outcome": best_outcome,
            "pred_scoreline": top_score,
            "pred_xg_home": pred["xg_home"],
            "pred_xg_away": pred["xg_away"],
            "pred_top_scorer": top_scorer["player_name"],
            "pred_scorer_prob": round(float(top_scorer["anytime_scorer_prob"]), 4),
            "actual_home_goals": pd.NA,
            "actual_away_goals": pd.NA,
            "actual_result": pd.NA,
            "outcome_hit": pd.NA,
            "scoreline_hit": pd.NA,
            "match_log_loss": pd.NA,
        })

    if new_rows:
        log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
        log_df.to_csv(LOG_PATH, index=False)
        print(f"Logged {len(new_rows)} new pre-match predictions -> {LOG_PATH}")
    else:
        print("All upcoming fixtures in window are already logged.")


def grade_completed_predictions():
    """Compare PENDING predictions in prediction_log.csv against matches_finished.csv."""
    if not LOG_PATH.exists():
        print("No prediction log found yet.")
        return

    log_df = pd.read_csv(LOG_PATH)
    finished_df = pd.read_csv(PROCESSED_DIR / "matches_finished.csv")

    # Lookup by (home_team, away_team, season=2026)
    recent_finished = finished_df[finished_df["season"] >= 2026].copy()
    result_lookup = {}
    for _, row in recent_finished.iterrows():
        key = (row["home_team"], row["away_team"])
        result_lookup[key] = {
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "result": row["result"],
        }

    graded_count = 0
    for idx, row in log_df.iterrows():
        if row["status"] == "GRADED":
            continue

        key = (row["home_team"], row["away_team"])
        if key not in result_lookup:
            continue

        actual = result_lookup[key]
        actual_score = f"{actual['home_goals']}-{actual['away_goals']}"
        actual_res = actual["result"]

        prob_for_actual = {
            "H": row["pred_home_prob"],
            "D": row["pred_draw_prob"],
            "A": row["pred_away_prob"],
        }[actual_res]

        log_df.at[idx, "status"] = "GRADED"
        log_df.at[idx, "actual_home_goals"] = actual["home_goals"]
        log_df.at[idx, "actual_away_goals"] = actual["away_goals"]
        log_df.at[idx, "actual_result"] = actual_res
        log_df.at[idx, "outcome_hit"] = bool(row["pred_outcome"] == actual_res)
        log_df.at[idx, "scoreline_hit"] = bool(row["pred_scoreline"] == actual_score)
        log_df.at[idx, "match_log_loss"] = round(-float(np.log(max(prob_for_actual, 1e-6))), 4)
        graded_count += 1

    log_df.to_csv(LOG_PATH, index=False)
    print(f"Graded {graded_count} completed matches in {LOG_PATH}")


if __name__ == "__main__":
    grade_completed_predictions()
    log_upcoming_matchweek(days_ahead=14)