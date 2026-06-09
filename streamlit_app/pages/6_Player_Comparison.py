from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_loader import load_all_data
from utils.filters import (
    get_tournament_options,
    get_team_options,
    get_player_options,
    filter_player
)
from utils.style import apply_global_style
from utils.cards import metric_card, text_card


st.set_page_config(page_title="Player Comparison", layout="wide")
apply_global_style()

data = load_all_data()

matches = data["matches"]
player_master = data["player_master"]
player_similarity = data["player_similarity"]
player_role_fit = data["player_role_fit"]
assets = data["assets"]

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# HELPERS
# ============================================================

def get_team_asset(team_name):
    asset_row = assets[assets["team"] == team_name]

    if len(asset_row) == 0:
        return None, "#2563EB"

    primary_color = asset_row.iloc[0].get("primary_color", "#2563EB")
    raw_flag_path = asset_row.iloc[0].get("flag_path", None)

    if not raw_flag_path or pd.isna(raw_flag_path):
        return None, primary_color

    raw_flag_path = str(raw_flag_path)

    possible_paths = [
        BASE_DIR / raw_flag_path,
        BASE_DIR / "streamlit_app" / raw_flag_path,
        Path(raw_flag_path),
    ]

    for p in possible_paths:
        if p.exists():
            return str(p), primary_color

    return None, primary_color


def get_row(profile):
    if len(profile) == 0:
        return None
    return profile.iloc[0]


def safe_value(row, col, default=0):
    if row is None:
        return default

    value = row.get(col, default)

    if pd.isna(value):
        return default

    return value


def get_first_available(row, cols, default=0):
    if row is None:
        return default

    for col in cols:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)

    return default


def build_player_scores(row):
    if row is None:
        return pd.DataFrame()

    attack = get_first_available(row, ["attacking_volume", "attack_score", "xg", "shots"], 0)
    passing = get_first_available(row, ["passing_volume", "passes", "build_up_score"], 0)
    carrying = get_first_available(row, ["carrying_volume", "carries", "progressive_actions"], 0)
    defense = get_first_available(row, ["defensive_volume", "defense_score", "recoveries", "duels"], 0)
    pressing = get_first_available(row, ["pressures", "pressure_volume", "defensive_activity"], 0)
    importance = get_first_available(row, ["importance_score"], 0)

    df = pd.DataFrame({
        "dimension": ["Attack", "Passing", "Carrying", "Defense", "Pressing", "Importance"],
        "value": [attack, passing, carrying, defense, pressing, importance]
    })

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    max_value = df["value"].max()

    if max_value > 100:
        df["score"] = (df["value"] / max_value) * 100
    else:
        df["score"] = df["value"]

    df["score"] = df["score"].clip(0, 100)

    return df


def build_comparison_table(row_a, row_b, player_a, player_b):
    metrics = [
        ("Position", "position"),
        ("Role", "general_role"),
        ("Matches", "matches"),
        ("Minutes", "minutes"),
        ("Importance", "importance_score"),
        ("Goals", "goals"),
        ("Assists", "assists"),
        ("xG", "xg"),
        ("xA", "xa"),
        ("Shots", "shots"),
        ("Passes", "passes"),
        ("Carries", "carries"),
        ("Pressures", "pressures"),
        ("Recoveries", "recoveries"),
        ("Duels", "duels"),
        ("Attacking Volume", "attacking_volume"),
        ("Passing Volume", "passing_volume"),
        ("Defensive Volume", "defensive_volume")
    ]

    rows = []

    for label, col in metrics:
        rows.append({
            "Metric": label,
            player_a: safe_value(row_a, col, None),
            player_b: safe_value(row_b, col, None)
        })

    return pd.DataFrame(rows)


def plot_comparison_radar(row_a, row_b, player_a, player_b, color_a="#2563EB", color_b="#DC2626"):
    scores_a = build_player_scores(row_a)
    scores_b = build_player_scores(row_b)

    if len(scores_a) == 0 or len(scores_b) == 0:
        return None

    labels = scores_a["dimension"].tolist()

    values_a = scores_a["score"].astype(float).tolist()
    values_b = scores_b["score"].astype(float).tolist()

    values_a += values_a[:1]
    values_b += values_b[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, values_a, linewidth=2.5, color=color_a, label=player_a)
    ax.fill(angles, values_a, color=color_a, alpha=0.12)

    ax.plot(angles, values_b, linewidth=2.5, color=color_b, label=player_b)
    ax.fill(angles, values_b, color=color_b, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, weight="bold")
    ax.set_ylim(0, 100)

    ax.set_title("Player Profile Radar", fontsize=16, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    return fig


def plot_metric_comparison(row_a, row_b, player_a, player_b, color_a="#2563EB", color_b="#DC2626"):
    metrics = {
        "xG": "xg",
        "Shots": "shots",
        "Passes": "passes",
        "Carries": "carries",
        "Pressures": "pressures",
        "Recoveries": "recoveries",
        "Duels": "duels",
        "Importance": "importance_score"
    }

    rows = []

    for label, col in metrics.items():
        rows.append({
            "metric": label,
            player_a: pd.to_numeric(safe_value(row_a, col, 0), errors="coerce"),
            player_b: pd.to_numeric(safe_value(row_b, col, 0), errors="coerce")
        })

    df = pd.DataFrame(rows).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(len(df))
    width = 0.36

    ax.bar(x - width / 2, df[player_a], width, label=player_a, color=color_a, alpha=0.85)
    ax.bar(x + width / 2, df[player_b], width, label=player_b, color=color_b, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["metric"])
    ax.set_title("Core Metric Comparison", fontsize=15, weight="bold")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_role_fit_bar(rf, player, color="#2563EB"):
    if len(rf) == 0:
        return None

    df = rf.copy()

    role_col = next((c for c in ["role", "role_name", "target_role"] if c in df.columns), None)
    score_col = next((c for c in ["role_fit_score", "fit_score", "score"] if c in df.columns), None)

    if role_col is None or score_col is None:
        return None

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    df = df.sort_values(score_col, ascending=False).head(6)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(df[role_col][::-1], df[score_col][::-1], color=color, alpha=0.85)

    ax.set_title(f"{player} | Role Fit", fontsize=15, weight="bold")
    ax.set_xlabel("Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_similarity_bar(sim, player, color="#2563EB"):
    if len(sim) == 0:
        return None

    df = sim.copy()

    player_col = next((c for c in ["similar_player", "player_2", "similar_name"] if c in df.columns), None)
    score_col = next((c for c in ["similarity_score", "score"] if c in df.columns), None)

    if player_col is None:
        return None

    if score_col is None:
        if "similarity_rank" in df.columns:
            df["similarity_score_plot"] = 100 - pd.to_numeric(df["similarity_rank"], errors="coerce").fillna(100)
            score_col = "similarity_score_plot"
        else:
            return None

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    if "similarity_rank" in df.columns:
        df = df.sort_values("similarity_rank")
    else:
        df = df.sort_values(score_col, ascending=False)

    df = df.head(6)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(df[player_col][::-1], df[score_col][::-1], color=color, alpha=0.85)

    ax.set_title(f"{player} | Similar Players", fontsize=15, weight="bold")
    ax.set_xlabel("Similarity")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def render_player_card(player, team, row, color):
    if row is None:
        st.info(f"No profile available for {player}.")
        return

    text_card(
        f"{player} Profile",
        (
            f"Team: {team} | "
            f"Position: {safe_value(row, 'position', 'N/A')} | "
            f"Role: {safe_value(row, 'general_role', 'N/A')} | "
            f"Importance: {round(safe_value(row, 'importance_score', 0), 2)} | "
            f"xG: {round(safe_value(row, 'xg', 0), 2)}"
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

tournaments = get_tournament_options(matches)

# ------------------------------------------------
# PLAYER A
# ------------------------------------------------

tournament_a = st.sidebar.selectbox(
    "Tournament A",
    tournaments,
    index=0
)

teams_a = get_team_options(
    player_master,
    tournament_a
)

team_a = st.sidebar.selectbox(
    "Team A",
    teams_a,
    index=0
)

players_a = get_player_options(
    player_master,
    tournament_a,
    team_a
)

player_a = st.sidebar.selectbox(
    "Player A",
    players_a,
    index=0
)

# ------------------------------------------------
# PLAYER B
# ------------------------------------------------

tournament_b = st.sidebar.selectbox(
    "Tournament B",
    tournaments,
    index=1 if len(tournaments) > 1 else 0
)

teams_b = get_team_options(
    player_master,
    tournament_b
)

team_b = st.sidebar.selectbox(
    "Team B",
    teams_b,
    index=0
)

players_b = get_player_options(
    player_master,
    tournament_b,
    team_b
)

player_b = st.sidebar.selectbox(
    "Player B",
    players_b,
    index=0
)

profile_a = filter_player(
    player_master,
    tournament_a,
    team_a,
    player_a
)

profile_b = filter_player(
    player_master,
    tournament_b,
    team_b,
    player_b
)

row_a = get_row(profile_a)
row_b = get_row(profile_b)

flag_a, color_a = get_team_asset(team_a)
flag_b, color_b = get_team_asset(team_b)


# ============================================================
# FILTERED EXTRA DATA
# ============================================================

rf_a = player_role_fit[
    (player_role_fit["tournament_id"] == tournament_a)
    &
    (player_role_fit["team"] == team_a)
    &
    (player_role_fit["player"] == player_a)
].copy()

rf_b = player_role_fit[
    (player_role_fit["tournament_id"] == tournament_b)
    &
    (player_role_fit["team"] == team_b)
    &
    (player_role_fit["player"] == player_b)
].copy()

sim_a = player_similarity[
    (player_similarity["tournament_id"] == tournament_a)
    &
    (player_similarity["team"] == team_a)
    &
    (player_similarity["player"] == player_a)
].copy()

sim_b = player_similarity[
    (player_similarity["tournament_id"] == tournament_b)
    &
    (player_similarity["team"] == team_b)
    &
    (player_similarity["player"] == player_b)
].copy()

# ============================================================
# HEADER
# ============================================================

h1, h2, h3 = st.columns([1, 6, 1])

with h1:
    if flag_a:
        st.image(flag_a, width=85)

with h2:
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0;'>{player_a} vs {player_b}</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='page-subtitle' style='text-align:center;'>
        {team_a} ({tournament_a})
        vs
        {team_b} ({tournament_b})
        </div>
        """,
        unsafe_allow_html=True
    )

with h3:
    if flag_b:
        st.image(flag_b, width=85)

# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    metric_card(f"{player_a} Role", safe_value(row_a, "general_role", "N/A"))

with k2:
    metric_card(f"{player_b} Role", safe_value(row_b, "general_role", "N/A"))

with k3:
    metric_card(f"{player_a} Importance", round(safe_value(row_a, "importance_score", 0), 2))

with k4:
    metric_card(f"{player_b} Importance", round(safe_value(row_b, "importance_score", 0), 2))

with k5:
    metric_card(f"{player_a} xG", round(safe_value(row_a, "xg", 0), 2))

with k6:
    metric_card(f"{player_b} xG", round(safe_value(row_b, "xg", 0), 2))


# ============================================================
# SINGLE MAIN VIEW
# ============================================================

left_col, radar_col, right_col = st.columns([1.05, 1.35, 1.05])

with left_col:
    render_player_card(player_a, team_a, row_a, color_a)

with radar_col:
    fig = plot_comparison_radar(
        row_a,
        row_b,
        player_a,
        player_b,
        color_a=color_a,
        color_b=color_b
    )

    if fig is not None:
        st.pyplot(fig, use_container_width=True)

with right_col:
    render_player_card(player_b, team_b, row_b, color_b)


st.markdown("### Core Metric Comparison")

fig = plot_metric_comparison(
    row_a,
    row_b,
    player_a,
    player_b,
    color_a=color_a,
    color_b=color_b
)

if fig is not None:
    st.pyplot(fig, use_container_width=True)


# ============================================================
# DETAILS
# ============================================================

tab_metrics, tab_roles, tab_similarity = st.tabs([
    "Metrics",
    "Role Fit",
    "Similarity"
])


with tab_metrics:

    comparison_table = build_comparison_table(row_a, row_b, player_a, player_b)

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True
    )


with tab_roles:

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"### {player_a} | {tournament_a}"
        )

        fig = plot_role_fit_bar(
            rf_a,
            player_a,
            color=color_a
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with c2:

        st.markdown(
            f"### {player_b} | {tournament_b}"
        )

        fig = plot_role_fit_bar(
            rf_b,
            player_b,
            color=color_b
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)


with tab_similarity:

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"### {player_a} | {tournament_a}"
        )

        fig = plot_similarity_bar(
            sim_a,
            player_a,
            color=color_a
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with c2:

        st.markdown(
            f"### {player_b} | {tournament_b}"
        )

        fig = plot_similarity_bar(
            sim_b,
            player_b,
            color=color_b
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
# ============================================================

# FOOTER

# ============================================================
st.divider()
st.caption(

    "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"

)