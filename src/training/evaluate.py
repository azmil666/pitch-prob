import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Ensure src imports work when running from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.score_model import DixonColesModel

PROCESSED_DIR = Path("data/processed")


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multi-class Brier score (range 0 to 2, lower is better)."""
    n_classes = y_prob.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def walk_forward_backtest(df: pd.DataFrame, xi: float = 0.0018, start_season: int = 2025):
    """
    Train on all matches prior to each matchweek in `start_season`+,
    predict that matchweek, and record out-of-sample probabilities.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Group test matches into 7-day blocks (matchweeks) to keep evaluation fast and realistic
    test_mask = df["season"] >= start_season
    test_df = df[test_mask].copy()
    test_df["week_bin"] = test_df["date"].dt.to_period("W")

    label_map = {"H": 0, "D": 1, "A": 2}
    y_true, y_probs, goal_errors = [], [], []

    for week, week_matches in test_df.groupby("week_bin"):
        cutoff_date = week_matches["date"].min()
        train_df = df[df["date"] < cutoff_date]

        if len(train_df) < 380:
            continue

        model = DixonColesModel(xi=xi)
        model.fit(train_df, ref_date=cutoff_date)

        for _, row in week_matches.iterrows():
            pred = model.predict_match(row["home_team"], row["away_team"])
            probs = [
                pred["probs"]["home_win"],
                pred["probs"]["draw"],
                pred["probs"]["away_win"],
            ]
            y_true.append(label_map[row["result"]])
            y_probs.append(probs)

            # Expected goals MAE
            err = abs(pred["xg_home"] - row["home_goals"]) + abs(pred["xg_away"] - row["away_goals"])
            goal_errors.append(err / 2.0)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    accuracy = float(np.mean(np.argmax(y_probs, axis=1) == y_true))
    ll = float(log_loss(y_true, y_probs, labels=[0, 1, 2]))
    bs = brier_score_multiclass(y_true, y_probs)
    mae = float(np.mean(goal_errors))

    return {
        "xi": xi,
        "matches_tested": len(y_true),
        "accuracy": round(accuracy, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(bs, 4),
        "goals_mae": round(mae, 4),
    }


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "matches_finished.csv")
    print("Running walk-forward backtest across 2025 & 2026 seasons...\n")

    for xi in [0.0005, 0.0018, 0.0035]:
        metrics = walk_forward_backtest(df, xi=xi, start_season=2025)
        print(
            f"xi={metrics['xi']:.4f} | Tested: {metrics['matches_tested']} matches | "
            f"Acc: {metrics['accuracy']*100:.1f}% | LogLoss: {metrics['log_loss']:.4f} | "
            f"Brier: {metrics['brier_score']:.4f} | Goals MAE: {metrics['goals_mae']:.3f}"
        )