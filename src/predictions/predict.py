import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.score_model import DixonColesModel
from src.models.scorer_model import GoalScorerModel


class FootIQPredictor:
    def __init__(self):
        model_path = Path("saved_models/dixon_coles.joblib")
        if not model_path.exists():
            import pandas as pd
            df = pd.read_csv("data/processed/matches_finished.csv")
            model = DixonColesModel(xi=0.0018).fit(df)
            model.save(model_path)
        self.score_model = DixonColesModel.load()
        self.scorer_model = GoalScorerModel()
        self.scorer_model.build_player_table()

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
    predictor = FootIQPredictor()
    res = predictor.predict_fixture("Manchester City", "Fulham")

    print(f"=== FootIQ Full Match Forecast: {res['home_team']} vs {res['away_team']} ===")
    print(f"Expected Goals : {res['xg_home']} - {res['xg_away']}")
    print(
        f"1X2 Probs      : Home {res['probs']['home_win']*100:.1f}% | "
        f"Draw {res['probs']['draw']*100:.1f}% | Away {res['probs']['away_win']*100:.1f}%"
    )
    print(
        "Top Scorelines : "
        + ", ".join(f"{s['score']} ({s['prob']*100:.1f}%)" for s in res["top_scorelines"])
    )
    print("\nMost Likely Anytime Goal Scorers:")
    for idx, row in res["top_scorers"].iterrows():
        print(
            f"  {idx+1}. {row['player_name']} ({row['team']} - {row['position']}) | "
            f"2026 Goals: {row['goals']} | Score Prob: {row['anytime_scorer_prob']*100:.1f}%"
        )