import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_current_pl_squads_and_scorers():
    """Fetch 2026/27 PL teams (with full squads) and top scorers from football-data.org."""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}

    # 1. Fetch all 20 current PL teams + their full 2026/27 player rosters
    teams_url = "https://api.football-data.org/v4/competitions/PL/teams?season=2026"
    res_teams = requests.get(teams_url, headers=headers, timeout=15)
    res_teams.raise_for_status()
    teams_data = res_teams.json()

    teams_out = RAW_DIR / "pl_teams_squads_2026.json"
    with open(teams_out, "w") as f:
        json.dump(teams_data, f, indent=2)
    print(f"Saved {len(teams_data.get('teams', []))} PL squads -> {teams_out}")

    time.sleep(6)  # Respect 10 req/min limit

    # 2. Fetch top scorers for 2026/27 (limit=100 covers almost every goalscorer so far)
    scorers_url = "https://api.football-data.org/v4/competitions/PL/scorers?season=2026&limit=100"
    res_scorers = requests.get(scorers_url, headers=headers, timeout=15)
    res_scorers.raise_for_status()
    scorers_data = res_scorers.json()

    scorers_out = RAW_DIR / "pl_scorers_2026.json"
    with open(scorers_out, "w") as f:
        json.dump(scorers_data, f, indent=2)
    print(f"Saved {len(scorers_data.get('scorers', []))} season scorers -> {scorers_out}")


if __name__ == "__main__":
    fetch_current_pl_squads_and_scorers()