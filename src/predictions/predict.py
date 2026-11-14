import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.score_model import DixonColesModel
from src.models.scorer_model import GoalScorerModel


class FootIQPredictor:
    def __init__(self, force_rebuild: bool = False):
        model_path = Path("saved_models/dixon_coles.joblib")
        players_path = Path("data/processed/players_2026.csv")

        if force_rebuild or not model_path.exists():
            df = pd.read_csv("data/processed/matches_finished.csv")
            model = DixonColesModel(xi=0.0018, reg_strength=1.5).fit(df)
            model.save(model_path)

        self.score_model = DixonColesModel.load(model_path)
        self.scorer_model = GoalScorerModel()

        if force_rebuild or not players_path.exists():
            self.scorer_model.build_player_table()
        else:
            self.scorer_model.players_df = pd.read_csv(players_path)

    def predict_fixture(
        self,
        home_team: str,
        away_team: str,
        confirmed_starters: list[str] | None = None,
        ruled_out: list[str] | None = None,
        top_scorers_n: int = 8,
    ) -> dict:
        match_pred = self.score_model.predict_match(home_team, away_team)
        scorers_df = self.scorer_model.predict_match_scorers(
            home_team=home_team,
            away_team=away_team,
            xg_home=match_pred["xg_home"],
            xg_away=match_pred["xg_away"],
            confirmed_starters=confirmed_starters,
            ruled_out=ruled_out,
            top_n=top_scorers_n,
        )
        match_pred["top_scorers"] = scorers_df
        return match_pred


if __name__ == "__main__":
    predictor = FootIQPredictor(force_rebuild=True)
    res = predictor.predict_fixture("Manchester City", "Fulham")
    print(f"=== Updated Forecast: {res['home_team']} vs {res['away_team']} ===")
    for idx, row in res["top_scorers"].iterrows():
        print(
            f"  {idx+1}. {row['player_name']} ({row['team']}) | "
            f"Goals: {row['goals']} | Score Prob: {row['anytime_scorer_prob']*100:.1f}%"
        )