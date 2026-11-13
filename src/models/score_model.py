import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

PROCESSED_DIR = Path("data/processed")
SAVED_MODELS_DIR = Path("saved_models")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DixonColesModel:
    def __init__(self, xi: float = 0.0018, max_goals: int = 8):
        """
        xi: Time-decay rate (in days^-1). ~0.0018 gives a half-life of ~385 days (~1 full season).
        max_goals: Grid size for exact scoreline probability matrix (0..max_goals).
        """
        self.xi = xi
        self.max_goals = max_goals
        self.params_ = {}
        self.teams_ = []

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """Dixon-Coles low-score correlation adjustment."""
        if x == 0 and y == 0:
            return 1.0 - (lam * mu * rho)
        elif x == 0 and y == 1:
            return 1.0 + (lam * rho)
        elif x == 1 and y == 0:
            return 1.0 + (mu * rho)
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        if ref_date is None:
            ref_date = df["date"].max()

        self.teams_ = sorted(set(df["home_team"]).union(set(df["away_team"])))
        n_teams = len(self.teams_)
        team_idx = {team: i for i, team in enumerate(self.teams_)}

        home_i = df["home_team"].map(team_idx).to_numpy()
        away_i = df["away_team"].map(team_idx).to_numpy()
        h_goals = df["home_goals"].astype(int).to_numpy()
        a_goals = df["away_goals"].astype(int).to_numpy()

        days_ago = (ref_date - df["date"]).dt.total_seconds() / 86400.0
        weights = np.exp(-self.xi * np.maximum(days_ago.to_numpy(), 0))

        # Parameter vector: [attack_0..n-1, defense_0..n-1, home_adv, rho]
        x0 = np.concatenate([np.ones(n_teams), np.ones(n_teams), [0.25, -0.1]])

        def neg_log_likelihood(params):
            attack = params[:n_teams]
            defense = params[n_teams : 2 * n_teams]
            home_adv = params[-2]
            rho = params[-1]

            lam = np.exp(attack[home_i] + defense[away_i] + home_adv)
            mu = np.exp(attack[away_i] + defense[home_i])

            # Vectorized tau adjustment
            tau = np.ones_like(lam)
            m00 = (h_goals == 0) & (a_goals == 0)
            m01 = (h_goals == 0) & (a_goals == 1)
            m10 = (h_goals == 1) & (a_goals == 0)
            m11 = (h_goals == 1) & (a_goals == 1)

            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho

            if np.any(tau <= 1e-6):
                return 1e12

            log_lik = (
                np.log(tau)
                + poisson.logpmf(h_goals, lam)
                + poisson.logpmf(a_goals, mu)
            )
            return -np.sum(weights * log_lik)

        # Identifiability constraint: mean(attack) == 0
        constraints = [{"type": "eq", "fun": lambda p: np.mean(p[:n_teams])}]
        bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-1.5, 1.5), (-0.5, 0.5)]

        res = minimize(
            neg_log_likelihood,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-7},
        )

        if not res.success:
            print(f"[Warning] Optimizer message: {res.message}")

        attack = res.x[:n_teams]
        defense = res.x[n_teams : 2 * n_teams]
        self.params_ = {
            "attack": dict(zip(self.teams_, attack)),
            "defense": dict(zip(self.teams_, defense)),
            "home_adv": float(res.x[-2]),
            "rho": float(res.x[-1]),
        }
        return self

    def predict_match(self, home_team: str, away_team: str) -> dict:
        # Fallback to league-average (0.0 in log space) if an unseen team appears
        att_h = self.params_["attack"].get(home_team, 0.0)
        def_h = self.params_["defense"].get(home_team, 0.0)
        att_a = self.params_["attack"].get(away_team, 0.0)
        def_a = self.params_["defense"].get(away_team, 0.0)
        home_adv = self.params_["home_adv"]
        rho = self.params_["rho"]

        lam = float(np.exp(att_h + def_a + home_adv))
        mu = float(np.exp(att_a + def_h))

        g = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))

        # Apply Dixon-Coles low-score adjustments
        for x in (0, 1):
            for y in (0, 1):
                matrix[x, y] *= self._tau(x, y, lam, mu, rho)

        matrix /= matrix.sum()  # Normalize probabilities to sum to 1.0

        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        # Flatten top 5 exact scorelines
        flat_indices = np.argsort(matrix.ravel())[::-1][:5]
        top_scorelines = [
            {"score": f"{idx // (self.max_goals + 1)}-{idx % (self.max_goals + 1)}", "prob": float(matrix.ravel()[idx])}
            for idx in flat_indices
        ]

        # Over 2.5 goals & Both Teams To Score (BTTS)
        grid_h, grid_a = np.meshgrid(g, g, indexing="ij")
        over_2_5 = float(matrix[(grid_h + grid_a) > 2].sum())
        btts = float(matrix[(grid_h >= 1) & (grid_a >= 1)].sum())

        return {
            "home_team": home_team,
            "away_team": away_team,
            "xg_home": round(lam, 2),
            "xg_away": round(mu, 2),
            "probs": {
                "home_win": round(home_win, 4),
                "draw": round(draw, 4),
                "away_win": round(away_win, 4),
            },
            "top_scorelines": top_scorelines,
            "over_2_5": round(over_2_5, 4),
            "btts": round(btts, 4),
            "matrix": matrix,
        }

    def save(self, path: Path = SAVED_MODELS_DIR / "dixon_coles.joblib"):
        payload = {
            "xi": self.xi,
            "max_goals": self.max_goals,
            "params_": self.params_,
            "teams_": self.teams_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path = SAVED_MODELS_DIR / "dixon_coles.joblib") -> "DixonColesModel":
        payload = joblib.load(path)
        model = cls(xi=payload["xi"], max_goals=payload["max_goals"])
        model.params_ = payload["params_"]
        model.teams_ = payload["teams_"]
        return model


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "matches_finished.csv")
    upcoming = pd.read_csv(PROCESSED_DIR / "matches_upcoming.csv")

    model = DixonColesModel(xi=0.0018)
    model.fit(df)
    model.save()

    print(f"Fitted Dixon-Coles on {len(df)} matches | Home Adv: {model.params_['home_adv']:.3f} | Rho: {model.params_['rho']:.3f}\n")

    # Preview predictions on the next 3 upcoming Premier League fixtures
    for _, row in upcoming.head(3).iterrows():
        pred = model.predict_match(row["home_team"], row["away_team"])
        top3 = ", ".join(f"{s['score']} ({s['prob']*100:.1f}%)" for s in pred["top_scorelines"][:3])
        print(f"{pred['home_team']} vs {pred['away_team']} (xG: {pred['xg_home']} - {pred['xg_away']})")
        print(f"  1X2  -> Home: {pred['probs']['home_win']*100:.1f}% | Draw: {pred['probs']['draw']*100:.1f}% | Away: {pred['probs']['away_win']*100:.1f}%")
        print(f"  Top Scores: {top3} | O2.5: {pred['over_2_5']*100:.1f}% | BTTS: {pred['btts']*100:.1f}%\n")