import json
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull City",
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
}

# Prior expected goals per 90 minutes by position when sample size is small
POSITION_PRIOR_G90 = {
    "Centre-Forward": 0.42,
    "Left Winger": 0.30,
    "Right Winger": 0.30,
    "Offence": 0.32,
    "Attacking Midfield": 0.24,
    "Central Midfield": 0.12,
    "Left Midfield": 0.16,
    "Right Midfield": 0.16,
    "Defensive Midfield": 0.07,
    "Midfield": 0.14,
    "Left-Back": 0.06,
    "Right-Back": 0.06,
    "Centre-Back": 0.05,
    "Defence": 0.05,
    "Goalkeeper": 0.001,
}


class GoalScorerModel:
    def __init__(self, prior_matches: float = 6.0):
        """
        prior_matches: Pseudo-count weight for empirical Bayes smoothing toward position prior.
        """
        self.prior_matches = prior_matches
        self.players_df = pd.DataFrame()

    def build_player_table(self) -> pd.DataFrame:
        with open(RAW_DIR / "pl_teams_squads_2026.json") as f:
            teams_data = json.load(f)
        with open(RAW_DIR / "pl_scorers_2026.json") as f:
            scorers_data = json.load(f)

        # Count how many 2026 matches each team has completed so far
        finished_df = pd.read_csv(PROCESSED_DIR / "matches_finished.csv")
        df_2026 = finished_df[finished_df["season"] == 2026]
        team_matches = {}
        for team in set(df_2026["home_team"]).union(set(df_2026["away_team"])):
            played = ((df_2026["home_team"] == team) | (df_2026["away_team"] == team)).sum()
            team_matches[team] = max(int(played), 1)

        # Map player_id -> 2026/27 scorer stats
        scorer_lookup = {}
        for item in scorers_data.get("scorers", []):
            p = item["player"]
            scorer_lookup[p["id"]] = {
                "goals": item.get("goals") or 0,
                "assists": item.get("assists") or 0,
                "penalties": item.get("penalties") or 0,
                "played_matches": item.get("playedMatches") or 0,
            }

        rows = []
        for team_obj in teams_data.get("teams", []):
            team_name = TEAM_NAME_MAP.get(team_obj["name"], team_obj["name"])
            matches_played_team = team_matches.get(team_name, 10)

            for member in team_obj.get("squad", []):
                pid = member["id"]
                pos = member.get("position") or "Midfield"
                stats = scorer_lookup.get(pid, {})

                goals = stats.get("goals", 0)
                assists = stats.get("assists", 0)
                penalties = stats.get("penalties", 0)
                appearances = stats.get("played_matches", 0)

                # Estimate starting/playing likelihood from appearances vs team matches
                if appearances > 0:
                    start_prob = min(0.95, max(0.35, appearances / matches_played_team))
                    eff_matches = appearances
                else:
                    # Conservative rotation weight for players without recorded goals yet
                    start_prob = 0.03 if pos == "Goalkeeper" else 0.15
                    eff_matches = matches_played_team * start_prob

                prior_rate = POSITION_PRIOR_G90.get(pos, 0.14)
                # Slight boost for proven penalty takers
                pen_bonus = 0.08 if penalties > 0 else 0.0
                smoothed_rate = ((goals + self.prior_matches * prior_rate) / (
                    eff_matches + self.prior_matches
                )) + pen_bonus

                rows.append({
                    "player_id": pid,
                    "player_name": member["name"],
                    "team": team_name,
                    "position": pos,
                    "goals": goals,
                    "assists": assists,
                    "penalties": penalties,
                    "appearances": appearances,
                    "start_prob": round(start_prob, 3),
                    "base_goal_rate": round(smoothed_rate, 4),
                })

        self.players_df = pd.DataFrame(rows)
        self.players_df.to_csv(PROCESSED_DIR / "players_2026.csv", index=False)
        return self.players_df

    def predict_match_scorers(
        self,
        home_team: str,
        away_team: str,
        xg_home: float,
        xg_away: float,
        confirmed_starters: list[str] | None = None,
        ruled_out: list[str] | None = None,
        top_n: int = 8,
    ) -> pd.DataFrame:
        if self.players_df.empty:
            self.build_player_table()

        confirmed_starters = set(confirmed_starters or [])
        ruled_out = set(ruled_out or [])

        results = []
        for team, team_xg in [(home_team, xg_home), (away_team, xg_away)]:
            squad = self.players_df[self.players_df["team"] == team].copy()
            if squad.empty:
                continue

            # Exclude injured/suspended players
            squad = squad[~squad["player_name"].isin(ruled_out)]

            # Apply lineup overrides if confirmed XI is known
            if confirmed_starters:
                squad["adj_start_prob"] = np.where(
                    squad["player_name"].isin(confirmed_starters),
                    0.95,
                    squad["start_prob"] * 0.25,
                )
            else:
                squad["adj_start_prob"] = squad["start_prob"]

            # Expected goal share weighted by base rate * expected minutes factor
            squad["raw_weight"] = squad["base_goal_rate"] * squad["adj_start_prob"]
            total_weight = squad["raw_weight"].sum()

            if total_weight > 0:
                squad["expected_goals"] = team_xg * (squad["raw_weight"] / total_weight)
            else:
                squad["expected_goals"] = 0.0

            # Poisson P(Goals >= 1) = 1 - exp(-lambda_p)
            squad["anytime_scorer_prob"] = 1.0 - np.exp(-squad["expected_goals"])
            results.append(squad)

        combined = pd.concat(results, ignore_index=True)
        combined = combined.sort_values("anytime_scorer_prob", ascending=False).head(top_n)
        return combined[
            [
                "player_name",
                "team",
                "position",
                "goals",
                "adj_start_prob",
                "expected_goals",
                "anytime_scorer_prob",
            ]
        ].reset_index(drop=True)