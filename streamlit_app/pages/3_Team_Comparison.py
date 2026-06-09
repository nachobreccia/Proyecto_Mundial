import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from utils.data_loader import load_all_data
from utils.filters import get_tournament_options, get_team_options, filter_team
from utils.style import apply_global_style
from utils.cards import metric_card, text_card


st.set_page_config(page_title="Team Comparison", layout="wide")
apply_global_style()

data = load_all_data()

matches = data["matches"]

if "team_master" not in data:
    st.error("team_master not found. Run team aggregation notebook first.")
    st.stop()

team_master = data["team_master"]
assets = data.get("assets", pd.DataFrame())

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# HELPERS
# ============================================================

def get_team_asset(team_name):

    if assets is None or len(assets) == 0:
        return None, "#2563EB"

    if "team" not in assets.columns:
        return None, "#2563EB"

    asset_row = assets[assets["team"] == team_name]

    if len(asset_row) == 0:
        return None, "#2563EB"

    primary_color = asset_row.iloc[0].get("primary_color", "#2563EB")
    raw_flag_path = asset_row.iloc[0].get("flag_path", None)

    if pd.isna(raw_flag_path) or raw_flag_path is None:
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


def get_value(row, col, default=0):

    if row is None:
        return default

    if col not in row.index:
        return default

    value = row[col]

    if pd.isna(value):
        return default

    return value


def safe_metric(row, col, fallback=None):
    if fallback is None:
        return get_value(row, col, 0)

    return get_value(row, col, get_value(row, fallback, 0))


def render_team_header(team_name, tournament_name, flag_path, color):
    c1, c2 = st.columns([1, 4])

    with c1:
        if flag_path:
            st.image(flag_path, width=70)

    with c2:
        st.markdown(
            f"<h2 style='color:{color}; margin-bottom:0;'>{team_name}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='page-subtitle'>{tournament_name}</div>",
            unsafe_allow_html=True
        )


def render_team_side(team_name, tournament_name, row, flag_path, color):
    render_team_header(team_name, tournament_name, flag_path, color)

    st.markdown("")

    c1, c2 = st.columns(2)

    with c1:
        metric_card("Matches", int(get_value(row, "matches", 0) or 0))

    with c2:
        metric_card("Goals", int(safe_metric(row, "goals_for", "goals") or 0))

    c3, c4 = st.columns(2)

    with c3:
        metric_card("Attack", round(get_value(row, "attack_score", 0) or 0, 1))

    with c4:
        metric_card("Defense", round(get_value(row, "defense_score", 0) or 0, 1))

    st.markdown("#### Style")

    text_card(
        "Tactical Identity",
        (
            f"General: {get_value(row, 'general_style', 'N/A')}<br>"
            f"Offensive: {get_value(row, 'offensive_style', 'N/A')}<br>"
            f"Defensive: {get_value(row, 'defensive_style', 'N/A')}"
        )
    )

    st.markdown("#### Key Metrics")

    metrics_df = pd.DataFrame([
        {"Metric": "Build-Up Score", "Value": round(get_value(row, "build_up_score", 0) or 0, 1)},
        {"Metric": "Directness Score", "Value": round(get_value(row, "directness_score", 0) or 0, 1)},
        {"Metric": "Passes / Match", "Value": round(get_value(row, "passes_per_match", 0) or 0, 1)},
        {"Metric": "Pass Accuracy %", "Value": round(get_value(row, "avg_pass_completion_pct", 0) or 0, 1)},
        {"Metric": "Tempo", "Value": round(get_value(row, "tempo", 0) or 0, 2)},
        {"Metric": "Pressures / Match", "Value": round(get_value(row, "pressures_per_match", 0) or 0, 1)},
        {"Metric": "Recoveries / Match", "Value": round(get_value(row, "recoveries_per_match", 0) or 0, 1)},
    ])

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        height=285
    )


def plot_two_team_radar(row_a, row_b, team_a, team_b, color_a, color_b):
    metrics = [
        ("Attack", "attack_score"),
        ("Build-Up", "build_up_score"),
        ("Defense", "defense_score"),
        ("Directness", "directness_score"),
    ]

    labels = [m[0] for m in metrics]

    values_a = [float(get_value(row_a, col, 0) or 0) for _, col in metrics]
    values_b = [float(get_value(row_b, col, 0) or 0) for _, col in metrics]

    values_a += values_a[:1]
    values_b += values_b[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

    ax.plot(angles, values_a, linewidth=2.5, color=color_a, label=team_a)
    ax.fill(angles, values_a, color=color_a, alpha=0.16)

    ax.plot(angles, values_b, linewidth=2.5, color=color_b, label=team_b)
    ax.fill(angles, values_b, color=color_b, alpha=0.16)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, weight="bold")

    max_val = max(
        max(values_a),
        max(values_b),
        1
    )

    upper = np.ceil(max_val / 10) * 10
    ax.set_ylim(0, upper)

    ax.set_title(
        f"{team_a} vs {team_b}",
        fontsize=18,
        weight="bold",
        pad=28
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

tournaments = get_tournament_options(matches)

tournament_a = st.sidebar.selectbox(
    "Tournament A",
    tournaments,
    index=0
)

teams_a = get_team_options(team_master, tournament_a)

team_a = st.sidebar.selectbox(
    "Team A",
    teams_a,
    index=0
)

default_idx = min(1, len(tournaments) - 1)

tournament_b = st.sidebar.selectbox(
    "Tournament B",
    tournaments,
    index=default_idx
)

teams_b = get_team_options(team_master, tournament_b)

team_b = st.sidebar.selectbox(
    "Team B",
    teams_b,
    index=0
)

team_a_profile = filter_team(team_master, tournament_a, team_a)
team_b_profile = filter_team(team_master, tournament_b, team_b)

row_a = get_row(team_a_profile)
row_b = get_row(team_b_profile)

if row_a is None:
    st.error(f"No data found for {team_a}")
    st.stop()

if row_b is None:
    st.error(f"No data found for {team_b}")
    st.stop()

team_a_flag, team_a_color = get_team_asset(team_a)
team_b_flag, team_b_color = get_team_asset(team_b)


# ============================================================
# HEADER
# ============================================================

h1, h2, h3 = st.columns([1, 6, 1])

with h1:
    if team_a_flag:
        st.image(team_a_flag, width=85)

with h2:
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0;'>{team_a} vs {team_b}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='page-subtitle' style='text-align:center;'>{team_a} ({tournament_a}) vs {team_b} ({tournament_b})</div>",
        unsafe_allow_html=True
    )

with h3:
    if team_b_flag:
        st.image(team_b_flag, width=85)


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    metric_card(f"{team_a} Goals", int(safe_metric(row_a, "goals_for", "goals") or 0))

with k2:
    metric_card(f"{team_b} Goals", int(safe_metric(row_b, "goals_for", "goals") or 0))

with k3:
    metric_card(f"{team_a} Attack", round(get_value(row_a, "attack_score", 0) or 0, 1))

with k4:
    metric_card(f"{team_b} Attack", round(get_value(row_b, "attack_score", 0) or 0, 1))

with k5:
    metric_card(f"{team_a} Defense", round(get_value(row_a, "defense_score", 0) or 0, 1))

with k6:
    metric_card(f"{team_b} Defense", round(get_value(row_b, "defense_score", 0) or 0, 1))


# ============================================================
# MAIN VIEW
# ============================================================

left_col, radar_col, right_col = st.columns([1.15, 1.45, 1.15])

with left_col:
    render_team_side(
        team_a,
        tournament_a,
        row_a,
        team_a_flag,
        team_a_color
    )

with radar_col:
    st.markdown("### Team DNA Radar")

    fig = plot_two_team_radar(
        row_a,
        row_b,
        team_a,
        team_b,
        team_a_color,
        team_b_color
    )

    if fig is not None:
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Quick Comparison")

    quick_df = pd.DataFrame([
        {
            "Metric": "Attack Score",
            team_a: round(get_value(row_a, "attack_score", 0) or 0, 1),
            team_b: round(get_value(row_b, "attack_score", 0) or 0, 1),
        },
        {
            "Metric": "Build-Up Score",
            team_a: round(get_value(row_a, "build_up_score", 0) or 0, 1),
            team_b: round(get_value(row_b, "build_up_score", 0) or 0, 1),
        },
        {
            "Metric": "Defense Score",
            team_a: round(get_value(row_a, "defense_score", 0) or 0, 1),
            team_b: round(get_value(row_b, "defense_score", 0) or 0, 1),
        },
        {
            "Metric": "Directness Score",
            team_a: round(get_value(row_a, "directness_score", 0) or 0, 1),
            team_b: round(get_value(row_b, "directness_score", 0) or 0, 1),
        },
        {
            "Metric": "Passes / Match",
            team_a: round(get_value(row_a, "passes_per_match", 0) or 0, 1),
            team_b: round(get_value(row_b, "passes_per_match", 0) or 0, 1),
        },
        {
            "Metric": "Pressures / Match",
            team_a: round(get_value(row_a, "pressures_per_match", 0) or 0, 1),
            team_b: round(get_value(row_b, "pressures_per_match", 0) or 0, 1),
        },
    ])

    st.dataframe(
        quick_df,
        use_container_width=True,
        hide_index=True,
        height=280
    )

with right_col:
    render_team_side(
        team_b,
        tournament_b,
        row_b,
        team_b_flag,
        team_b_color
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"
)