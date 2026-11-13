import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Map varying API team names to a single canonical name
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham Forest",
    "Sheffield United FC": "Sheffield Utd",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull City",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def parse_api_football(seasons=(2022, 2023, 2024)) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = RAW_DIR / f"pl_fixtures_apifb_{season}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        for item in data.get("response", []):
            fixture = item["fixture"]
            teams = item["teams"]
            goals = item["goals"]
            status = fixture["status"]["short"]

            rows.append({
                "source": "api_football",
                "apifb_fixture_id": fixture["id"],
                "season": season,
                "date": pd.to_datetime(fixture["date"]).tz_convert("UTC"),
                "matchday": item["league"].get("round"),
                "status": "FINISHED" if status in ("FT", "AET", "PEN") else status,
                "home_team": normalize_team(teams["home"]["name"]),
                "away_team": normalize_team(teams["away"]["name"]),
                "home_goals": goals["home"],
                "away_goals": goals["away"],
            })
    return pd.DataFrame(rows)


def parse_football_data(seasons=(2025, 2026)) -> pd.DataFrame:
    """Use 2025 and 2026 from football-data.org (2024 is already covered by API-Football)."""
    rows = []
    for season in seasons:
        path = RAW_DIR / f"pl_matches_fd_{season}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        for m in data.get("matches", []):
            ft = m["score"]["fullTime"]
            rows.append({
                "source": "football_data",
                "apifb_fixture_id": pd.NA,
                "season": season,
                "date": pd.to_datetime(m["utcDate"]).tz_convert("UTC"),
                "matchday": f"Regular Season - {m.get('matchday')}",
                "status": m["status"],  # FINISHED, TIMED, SCHEDULED, etc.
                "home_team": normalize_team(m["homeTeam"]["name"]),
                "away_team": normalize_team(m["awayTeam"]["name"]),
                "home_goals": ft["home"],
                "away_goals": ft["away"],
            })
    return pd.DataFrame(rows)


def build_clean_dataset():
    df_apifb = parse_api_football(seasons=(2022, 2023, 2024))
    df_fd = parse_football_data(seasons=(2025, 2026))

    df = pd.concat([df_apifb, df_fd], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Add match outcome label (H = Home win, D = Draw, A = Away win) for finished matches
    def get_result(row):
        if row["status"] != "FINISHED" or pd.isna(row["home_goals"]):
            return pd.NA
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        return "D"

    df["result"] = df.apply(get_result, axis=1)

    # Separate completed historical matches from upcoming fixtures
    finished_df = df[df["status"] == "FINISHED"].copy()
    upcoming_df = df[df["status"] != "FINISHED"].copy()

    finished_df.to_csv(PROCESSED_DIR / "matches_finished.csv", index=False)
    upcoming_df.to_csv(PROCESSED_DIR / "matches_upcoming.csv", index=False)

    print("=== Dataset Summary ===")
    print(f"Total finished matches : {len(finished_df)}")
    print(f"Total upcoming fixtures: {len(upcoming_df)}")
    print(f"Unique teams ({finished_df['home_team'].nunique()}): {sorted(finished_df['home_team'].unique())}")
    print("\nMatches per season (Finished):")
    print(finished_df["season"].value_counts().sort_index())


if __name__ == "__main__":
    build_clean_dataset()