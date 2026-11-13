import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_football_data_seasons(seasons=(2024, 2025, 2026)):
    """Fetch PL matches from football-data.org for given starting years."""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    for season in seasons:
        out_path = RAW_DIR / f"pl_matches_fd_{season}.json"
        if out_path.exists() and season != 2026:
            print(f"[Skip] {out_path} already exists.")
            continue

        url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"[football-data.org] Season {season} failed ({res.status_code}): {res.text}")
            continue

        data = res.json()
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[football-data.org] Saved {len(data.get('matches', []))} matches -> {out_path}")
        time.sleep(6)  # Respect free-tier 10 requests/minute limit


def fetch_api_football_seasons(seasons=(2022, 2023, 2024)):
    """Fetch PL fixtures (league=39) from API-Football for free-tier allowed seasons."""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}

    for season in seasons:
        out_path = RAW_DIR / f"pl_fixtures_apifb_{season}.json"
        if out_path.exists():
            print(f"[Skip] {out_path} already exists.")
            continue

        res = requests.get(url, headers=headers, params={"league": 39, "season": season}, timeout=15)
        data = res.json()

        if data.get("errors"):
            print(f"[API-Football] Season {season} error: {data['errors']}")
            continue

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[API-Football] Saved {len(data.get('response', []))} fixtures -> {out_path}")
        time.sleep(1)


if __name__ == "__main__":
    fetch_football_data_seasons()
    fetch_api_football_seasons()