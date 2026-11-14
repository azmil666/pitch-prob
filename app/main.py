import json
import os
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
from src.data.refresh import refresh_matchweek_data
from src.data.clean_data import normalize_team

st.set_page_config(
    page_title="FootIQ | Premier League Analytics & Predictions",
    page_icon=":material/sports_soccer:",
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


@st.cache_data
def load_team_crests() -> dict[str, str]:
    crests_path = ROOT_DIR / "data/raw/pl_teams_squads_2026.json"
    if not crests_path.exists():
        return {}
    with open(crests_path) as f:
        data = json.load(f)
    return {
        normalize_team(t["name"]): t.get("crest", "")
        for t in data.get("teams", [])
        if t.get("crest")
    }


predictor = load_predictor()
upcoming_df = load_fixtures()
team_crests = load_team_crests()
players_df = predictor.scorer_model.players_df
teams_2026 = sorted(players_df["team"].unique())

# --- Sidebar: Match, Lineup & Refresh Controls ---
st.sidebar.title(":material/sports_soccer: FootIQ Controls")

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
    home_team = st.sidebar.selectbox(
        "Home Team",
        teams_2026,
        index=teams_2026.index("Manchester City") if "Manchester City" in teams_2026 else 0,
    )
    away_options = [t for t in teams_2026 if t != home_team]
    away_team = st.sidebar.selectbox(
        "Away Team",
        away_options,
        index=away_options.index("Fulham") if "Fulham" in away_options else 0,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Lineup & Injury Adjustments")
st.sidebar.caption("Update when team news drops 30 mins before kickoff.")

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

st.sidebar.markdown("---")
if st.sidebar.button("Refresh Matchweek Data", icon=":material/refresh:", use_container_width=True):
    if not os.getenv("FOOTBALL_DATA_ORG_KEY"):
        st.sidebar.warning("API key not configured in this environment. Using cached data.")
    else:
        with st.spinner("Fetching latest 2026/27 results and retraining models..."):
            refresh_matchweek_data()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

# --- Tabs ---
tab_forecast, tab_rankings = st.tabs([":material/analytics: Match Forecast", ":material/leaderboard: Club Power Rankings"])

# ==========================================
# TAB 1: MATCH FORECAST
# ==========================================
with tab_forecast:
    forecast = predictor.predict_fixture(
        home_team=home_team,
        away_team=away_team,
        confirmed_starters=confirmed_starters,
        ruled_out=ruled_out,
        top_scorers_n=10,
    )

    home_crest = team_crests.get(home_team, "")
    away_crest = team_crests.get(away_team, "")
    home_badge_html = (
        f'<img src="{home_crest}" width="40" height="40" style="object-fit: contain; vertical-align: middle; margin-right: 8px;">'
        if home_crest
        else ""
    )
    away_badge_html = (
        f'<img src="{away_crest}" width="40" height="40" style="object-fit: contain; vertical-align: middle; margin-left: 8px;">'
        if away_crest
        else ""
    )

    header_html = f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        {home_badge_html}
        <h1 style="margin: 0; padding: 0; font-size: 2.3rem; font-weight: 700; line-height: 1.2;">
            {home_team} <span style="font-weight: 400; color: #888; font-size: 1.8rem; margin: 0 4px;">vs</span> {away_team}
        </h1>
        {away_badge_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.caption("Premier League 2026/27 • Regularized Dixon-Coles Scoreline Engine + Bayesian Poisson Scorer Model")

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

    st.subheader("Engine 3: Anytime Goal Scorer Probabilities")

    scorers_display = forecast["top_scorers"].copy()
    scorers_display["Badge"] = scorers_display["team"].map(team_crests).fillna("")
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
                ["Badge", "player_name", "team", "position", "goals", "Start Likelihood", "Expected Goals", "Prob (%)"]
            ].rename(
                columns={
                    "player_name": "Player",
                    "team": "Club",
                    "position": "Position",
                    "goals": "2026 Goals",
                }
            ),
            column_config={
                "Badge": st.column_config.ImageColumn("Badge", help="Club Badge"),
            },
            hide_index=True,
            use_container_width=True,
            height=380,
        )

# ==========================================
# TAB 2: CLUB POWER RANKINGS
# ==========================================
with tab_rankings:
    st.title("Premier League 2026/27 Club Strength Ratings")
    st.caption("Latent Attack & Defense parameters estimated by the time-decayed Dixon-Coles model.")

    dc_params = predictor.score_model.params_
    ranking_rows = []
    for team in teams_2026:
        att = dc_params["attack"].get(team, 0.0)
        # Invert sign so positive = stronger defense (fewer goals conceded)
        def_solidity = -dc_params["defense"].get(team, 0.0)
        net_rating = att + def_solidity
        ranking_rows.append({
            "Badge": team_crests.get(team, ""),
            "Club": team,
            "Attack Rating": round(att, 3),
            "Defense Solidity": round(def_solidity, 3),
            "Overall Power": round(net_rating, 3),
        })

    rank_df = pd.DataFrame(ranking_rows).sort_values("Overall Power", ascending=False).reset_index(drop=True)
    rank_df.index += 1

    r_col1, r_col2 = st.columns([1.3, 1])

    with r_col1:
        fig_scatter = px.scatter(
            rank_df,
            x="Attack Rating",
            y="Defense Solidity",
            text="Club",
            color="Overall Power",
            color_continuous_scale="RdYlGn",
            size_max=14,
            title="Attack Strength vs. Defensive Solidity (Top-Right = Title Contenders)",
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(size=11))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with r_col2:
        st.subheader("Power Index Table")
        st.dataframe(
            rank_df[["Badge", "Club", "Attack Rating", "Defense Solidity", "Overall Power"]],
            column_config={
                "Badge": st.column_config.ImageColumn("Badge", help="Club Badge"),
            },
            use_container_width=True,
            height=500,
        )