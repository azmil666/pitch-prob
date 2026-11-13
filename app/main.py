import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure root imports work when launching via `streamlit run app/main.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predictions.predict import FootIQPredictor

st.set_page_config(
    page_title="FootIQ | Premier League Analytics & Predictions",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> FootIQPredictor:
    return FootIQPredictor()


@st.cache_data
def load_fixtures() -> pd.DataFrame:
    upcoming_path = ROOT_DIR / "data/processed/matches_upcoming.csv"
    if upcoming_path.exists():
        df = pd.read_csv(upcoming_path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    return pd.DataFrame()


predictor = load_predictor()
upcoming_df = load_fixtures()
players_df = predictor.scorer_model.players_df
teams_2026 = sorted(players_df["team"].unique())

# --- Sidebar: Match & Lineup Controls ---
st.sidebar.title("⚽ FootIQ Controls")

mode = st.sidebar.radio(
    "Fixture Selection Mode",
    ["Upcoming PL Fixtures", "Custom Matchup"],
)

if mode == "Upcoming PL Fixtures" and not upcoming_df.empty:
    upcoming_df["label"] = (
        upcoming_df["date"].dt.strftime("%b %d")
        + " | "
        + upcoming_df["home_team"]
        + " vs "
        + upcoming_df["away_team"]
    )
    selected_label = st.sidebar.selectbox("Select Fixture", upcoming_df["label"].head(40))
    match_row = upcoming_df[upcoming_df["label"] == selected_label].iloc[0]
    home_team = match_row["home_team"]
    away_team = match_row["away_team"]
else:
    home_team = st.sidebar.selectbox("Home Team", teams_2026, index=teams_2026.index("Manchester City") if "Manchester City" in teams_2026 else 0)
    away_options = [t for t in teams_2026 if t != home_team]
    away_team = st.sidebar.selectbox("Away Team", away_options, index=away_options.index("Fulham") if "Fulham" in away_options else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Lineup & Injury Adjustments")
st.sidebar.caption("Optional: Update when team news drops 30 mins before kickoff.")

home_squad = sorted(players_df[players_df["team"] == home_team]["player_name"].tolist())
away_squad = sorted(players_df[players_df["team"] == away_team]["player_name"].tolist())
all_match_players = home_squad + away_squad

confirmed_starters = st.sidebar.multiselect(
    "Confirmed Starting XI (Boosts Expected Minutes)",
    options=all_match_players,
)
ruled_out = st.sidebar.multiselect(
    "Injured / Suspended (Excluded)",
    options=[p for p in all_match_players if p not in confirmed_starters],
)

# --- Run All 3 Prediction Engines ---
forecast = predictor.predict_fixture(
    home_team=home_team,
    away_team=away_team,
    confirmed_starters=confirmed_starters,
    ruled_out=ruled_out,
    top_scorers_n=10,
)

# --- Main Header ---
st.title(f"{home_team} vs {away_team}")
st.caption("Premier League 2026/27 • Dixon-Coles Scoreline Engine + Bayesian Poisson Scorer Model")

# Top KPI Cards
probs = forecast["probs"]
top_score = forecast["top_scorelines"][0]
top_scorer = forecast["top_scorers"].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(f"{home_team} Win", f"{probs['home_win']*100:.1f}%")
col2.metric("Draw", f"{probs['draw']*100:.1f}%")
col3.metric(f"{away_team} Win", f"{probs['away_win']*100:.1f}%")
col4.metric("Most Likely Score", top_score["score"], f"{top_score['prob']*100:.1f}% prob")
col5.metric("Top Anytime Scorer", top_scorer["player_name"], f"{top_scorer['anytime_scorer_prob']*100:.1f}% prob")

st.markdown("---")

# --- Engine 1 & 2 Visuals ---
left_col, right_col = st.columns([1.1, 1.2])

with left_col:
    st.subheader("Engine 1: Match Outcome & Goal Markets")

    fig_1x2 = go.Figure(
        go.Bar(
            x=[probs["home_win"] * 100, probs["draw"] * 100, probs["away_win"] * 100],
            y=[f"{home_team} Win", "Draw", f"{away_team} Win"],
            orientation="h",
            text=[f"{probs['home_win']*100:.1f}%", f"{probs['draw']*100:.1f}%", f"{probs['away_win']*100:.1f}%"],
            textposition="auto",
            marker_color=["#1f77b4", "#7f7f7f", "#d62728"],
        )
    )
    fig_1x2.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], title="Probability (%)"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_1x2, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Goals (xG)", f"{forecast['xg_home']} – {forecast['xg_away']}")
    m2.metric("Over 2.5 Goals", f"{forecast['over_2_5']*100:.1f}%")
    m3.metric("Both Teams To Score", f"{forecast['btts']*100:.1f}%")

    st.markdown("**Top 5 Exact Scorelines**")
    scores_df = pd.DataFrame(forecast["top_scorelines"])
    scores_df["Probability"] = (scores_df["prob"] * 100).round(1).astype(str) + "%"
    st.dataframe(
        scores_df[["score", "Probability"]].rename(columns={"score": "Scoreline"}),
        hide_index=True,
        use_container_width=True,
    )

with right_col:
    st.subheader("Engine 2: Scoreline Probability Matrix")
    matrix_6x6 = forecast["matrix"][:6, :6] * 100

    fig_heat = px.imshow(
        matrix_6x6,
        labels=dict(x=f"{away_team} Goals", y=f"{home_team} Goals", color="Prob (%)"),
        x=[str(i) for i in range(6)],
        y=[str(i) for i in range(6)],
        text_auto=".1f",
        color_continuous_scale="Blues",
    )
    fig_heat.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# --- Engine 3 Visuals ---
st.subheader("Engine 3: Anytime Goal Scorer Probabilities")

scorers_display = forecast["top_scorers"].copy()
scorers_display["Prob (%)"] = (scorers_display["anytime_scorer_prob"] * 100).round(1)
scorers_display["Start Likelihood"] = (scorers_display["adj_start_prob"] * 100).round(0).astype(int).astype(str) + "%"
scorers_display["Expected Goals"] = scorers_display["expected_goals"].round(2)

sc_col1, sc_col2 = st.columns([1.2, 1])

with sc_col1:
    fig_scorers = px.bar(
        scorers_display.iloc[::-1],
        x="Prob (%)",
        y="player_name",
        color="team",
        orientation="h",
        text="Prob (%)",
        labels={"player_name": "", "team": "Club"},
    )
    fig_scorers.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_scorers.update_layout(
        height=380,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, max(65, scorers_display["Prob (%)"].max() + 10)]),
    )
    st.plotly_chart(fig_scorers, use_container_width=True)

with sc_col2:
    st.dataframe(
        scorers_display[
            ["player_name", "team", "position", "goals", "Start Likelihood", "Expected Goals", "Prob (%)"]
        ].rename(
            columns={
                "player_name": "Player",
                "team": "Club",
                "position": "Position",
                "goals": "2026 Goals",
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=380,
    )