import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import Pitch

from utils.data_loader import load_all_data
from utils.filters import (
    get_tournament_options,
    get_team_options,
    get_player_options,
    filter_player
)
from utils.style import apply_global_style
from utils.cards import metric_card, text_card


st.set_page_config(page_title="Player Tournament", layout="wide")
apply_global_style()

data = load_all_data()

matches = data["matches"]
events = data["events"]
player_master = data["player_master"]
player_role_fit = data["player_role_fit"]
player_similarity = data["player_similarity"]
highlighted_matches = data["highlighted_matches"]
assets = data["assets"]

if "event_type" not in events.columns and "type" in events.columns:
    events["event_type"] = events["type"]

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# HELPERS
# ============================================================

def parse_location(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None

    return None


def parse_list_value(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]

        if value.strip():
            return [value.strip()]

    return []


def list_card(title, values):
    values = parse_list_value(values)

    if len(values) == 0:
        values = ["No data available"]

    html_items = "".join([f"<li>{v}</li>" for v in values])

    st.markdown(
        f"""
        <div class="custom-card">
            <div class="section-label">{title}</div>
            <ul style="margin-bottom:0; padding-left:18px; line-height:1.6;">
                {html_items}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def ensure_xy(df):
    df = df.copy()

    if "x" not in df.columns:
        df["x"] = np.nan

    if "y" not in df.columns:
        df["y"] = np.nan

    if "location" in df.columns:
        loc = df["location"].apply(parse_location)

        df["x"] = df["x"].fillna(
            loc.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        )

        df["y"] = df["y"].fillna(
            loc.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)
        )

    return df


def draw_pitch():
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#F8FAFC",
        line_color="#111827",
        linewidth=1.1
    )

    fig, ax = pitch.draw(figsize=(10, 7))
    return pitch, fig, ax


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
        Path(raw_flag_path)
    ]

    for p in possible_paths:
        if p.exists():
            return str(p), primary_color

    return None, primary_color


def safe_value(row, col, default=0):
    if row is None:
        return default

    if col not in row.index:
        return default

    value = row.get(col, default)

    if pd.isna(value):
        return default

    return value


def get_row(df):
    if len(df) == 0:
        return None

    return df.iloc[0]


def compact_table(df, preferred_cols):
    existing_cols = [c for c in preferred_cols if c in df.columns]

    if len(existing_cols) == 0:
        return df

    return df[existing_cols]


def pct(numerator, denominator):
    numerator = 0 if pd.isna(numerator) else numerator
    denominator = 0 if pd.isna(denominator) else denominator

    if denominator == 0:
        return 0

    return round((numerator / denominator) * 100, 1)


def get_first_available(row, cols, default=0):
    if row is None:
        return default

    for col in cols:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)

    return default


def player_events_filter(events, tournament_id, team, player):
    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["player"] == player)
    ].copy()

    if "period" in df.columns:
        df = df[df["period"].fillna(0).astype(float) <= 4].copy()

    return df


def build_player_event_kpis(player_events):
    df = player_events.copy()

    shots = len(df[df["event_type"] == "Shot"])
    goals = 0

    if "shot_outcome" in df.columns:
        goals = len(
            df[
                (df["event_type"] == "Shot") &
                (df["shot_outcome"] == "Goal")
            ]
        )

    passes = len(df[df["event_type"] == "Pass"])
    completed_passes = 0

    if "pass_outcome" in df.columns:
        completed_passes = len(
            df[
                (df["event_type"] == "Pass") &
                (df["pass_outcome"].isna())
            ]
        )

    duels = len(df[df["event_type"] == "Duel"])
    duels_won = 0

    if "duel_outcome" in df.columns:
        duels_won = len(
            df[
                (df["event_type"] == "Duel") &
                (df["duel_outcome"].astype(str).str.contains("Won|Success", case=False, na=False))
            ]
        )

    pressures = len(df[df["event_type"] == "Pressure"])
    recoveries = len(df[df["event_type"].isin(["Ball Recovery", "Interception"])])
    carries = len(df[df["event_type"] == "Carry"])

    crosses = 0
    completed_crosses = 0

    if "pass_cross" in df.columns:
        cross_df = df[
            (df["event_type"] == "Pass") &
            (df["pass_cross"].fillna(False) == True)
        ]

        crosses = len(cross_df)

        if "pass_outcome" in cross_df.columns:
            completed_crosses = len(cross_df[cross_df["pass_outcome"].isna()])

    shot_on_target = 0

    if "shot_outcome" in df.columns:
        shot_on_target = len(
            df[
                (df["event_type"] == "Shot") &
                (df["shot_outcome"].astype(str).isin(["Goal", "Saved", "Saved To Post"]))
            ]
        )

    xg = 0
    if "shot_statsbomb_xg" in df.columns:
        xg = df["shot_statsbomb_xg"].fillna(0).sum()

    return {
        "shots": shots,
        "goals": goals,
        "xg": xg,
        "shot_accuracy_pct": pct(shot_on_target, shots),
        "goal_conversion_pct": pct(goals, shots),
        "passes": passes,
        "completed_passes": completed_passes,
        "pass_completion_pct": pct(completed_passes, passes),
        "duels": duels,
        "duels_won": duels_won,
        "duel_win_pct": pct(duels_won, duels),
        "crosses": crosses,
        "completed_crosses": completed_crosses,
        "cross_completion_pct": pct(completed_crosses, crosses),
        "pressures": pressures,
        "recoveries": recoveries,
        "carries": carries,
    }


def build_player_dimension_scores(row, event_kpis):
    if row is None:
        return pd.DataFrame()

    attack = get_first_available(row, ["attacking_volume", "attack_score", "xg", "shots"], event_kpis["shots"])
    passing = get_first_available(row, ["passing_volume", "passes", "build_up_score"], event_kpis["passes"])
    carrying = get_first_available(row, ["carrying_volume", "carries", "progressive_actions"], event_kpis["carries"])
    defense = get_first_available(row, ["defensive_volume", "defense_score", "recoveries", "duels"], event_kpis["recoveries"])
    pressing = get_first_available(row, ["pressing_volume", "pressures", "pressure_volume"], event_kpis["pressures"])
    importance = get_first_available(row, ["importance_score", "gk_importance_score", "outfield_importance_score"], 0)

    raw = pd.DataFrame({
        "dimension": ["Attack", "Passing", "Carrying", "Defense", "Pressing", "Importance"],
        "value": [attack, passing, carrying, defense, pressing, importance]
    })

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").fillna(0)

    max_value = raw["value"].max()

    if max_value > 100:
        raw["score"] = (raw["value"] / max_value) * 100
    else:
        raw["score"] = raw["value"]

    raw["score"] = raw["score"].clip(0, 100)

    return raw


def generate_player_summary(row, player, team, tournament_id, event_kpis):
    if row is None:
        return "No player profile available."

    position = safe_value(row, "position", "N/A")
    general_role = safe_value(row, "general_role", "N/A")
    offensive_role = safe_value(row, "offensive_role", "N/A")
    defensive_role = safe_value(row, "defensive_role", "N/A")
    importance = round(safe_value(row, "importance_score", 0), 2)
    tier = safe_value(row, "importance_tier", "N/A")

    return (
        f"{player} played mainly as {position} for {team} in {tournament_id}. "
        f"The player profiles as a {general_role}, with an offensive role of {offensive_role} "
        f"and a defensive role of {defensive_role}. "
        f"His importance score is {importance}, classified as {tier}. "
        f"Across the selected tournament data, he registered {event_kpis['shots']} shots, "
        f"{event_kpis['passes']} passes, {event_kpis['carries']} carries, "
        f"{event_kpis['pressures']} pressures and {event_kpis['recoveries']} recoveries/interceptions."
    )


# ============================================================
# VISUALS
# ============================================================

def plot_player_radar(row, event_kpis, player, team_color="#2563EB"):
    scores = build_player_dimension_scores(row, event_kpis)

    if len(scores) == 0:
        return None

    labels = scores["dimension"].tolist()
    values = scores["score"].astype(float).tolist()
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2.5, color=team_color)
    ax.fill(angles, values, color=team_color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, weight="bold")
    ax.set_ylim(0, 100)

    ax.set_title(f"{player} | Profile Radar", fontsize=15, weight="bold", pad=20)

    return fig


def plot_player_profile_bars(row, event_kpis, player, team_color="#2563EB"):
    scores = build_player_dimension_scores(row, event_kpis)

    if len(scores) == 0:
        return None

    scores = scores.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(scores["dimension"], scores["score"], color=team_color, alpha=0.85)
    ax.set_xlim(0, 100)
    ax.set_title(f"{player} | Profile Scores", fontsize=15, weight="bold")
    ax.set_xlabel("Score")

    for i, value in enumerate(scores["score"]):
        ax.text(value + 1, i, f"{value:.1f}", va="center", fontsize=9, weight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_player_zone_heatmap(player_events, event_types, title, cmap="Blues"):
    df = player_events[player_events["event_type"].isin(event_types)].copy()

    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(
        df["x"],
        df["y"],
        statistic="count",
        bins=(6, 4)
    )

    pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, alpha=0.78)
    pitch.label_heatmap(
        bin_statistic,
        color="#111827",
        fontsize=10,
        ax=ax,
        ha="center",
        va="center"
    )

    ax.set_title(title, fontsize=15, weight="bold")

    return fig


def plot_shot_map(player_events, player, team_color="#2563EB"):
    df = player_events[player_events["event_type"] == "Shot"].copy()

    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    if "shot_statsbomb_xg" not in df.columns:
        df["shot_statsbomb_xg"] = 0

    df["shot_xg"] = df["shot_statsbomb_xg"].fillna(0)

    pitch, fig, ax = draw_pitch()

    goals = df[df["shot_outcome"] == "Goal"] if "shot_outcome" in df.columns else pd.DataFrame()
    shots = df[df["shot_outcome"] != "Goal"] if "shot_outcome" in df.columns else df

    if len(shots) > 0:
        pitch.scatter(
            shots["x"],
            shots["y"],
            s=shots["shot_xg"] * 1300 + 65,
            ax=ax,
            color=team_color,
            alpha=0.50,
            label="Shot"
        )

    if len(goals) > 0:
        pitch.scatter(
            goals["x"],
            goals["y"],
            s=goals["shot_xg"] * 1500 + 100,
            ax=ax,
            color="#DC2626",
            edgecolors="#111827",
            linewidth=1.5,
            label="Goal"
        )

    ax.set_title(
        f"{player} Shot Map | Shots {len(df)} | Goals {len(goals)} | xG {df['shot_xg'].sum():.2f}",
        fontsize=15,
        weight="bold"
    )

    ax.legend(loc="upper left")

    return fig


def plot_cross_origin_zones(player_events, player):
    df = player_events[player_events["event_type"] == "Pass"].copy()

    if "pass_cross" not in df.columns:
        return None

    df = df[df["pass_cross"].fillna(False) == True].copy()

    return plot_player_zone_heatmap(
        df,
        ["Pass"],
        f"{player} | Cross Origin Zones",
        cmap="Blues"
    )


def plot_reception_zones(events, tournament_id, team, player):
    if "pass_recipient" not in events.columns:
        return None

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["pass_recipient"] == player) &
        (events["event_type"] == "Pass")
    ].copy()

    if "pass_end_location" not in df.columns:
        return None

    pass_end = df["pass_end_location"].apply(parse_location)

    df["x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap="Purples", alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax)

    ax.set_title(f"{player} | Reception Zones", fontsize=15, weight="bold")

    return fig


def plot_progressive_actions(player_events, player, team_color="#2563EB"):
    df = player_events[player_events["event_type"].isin(["Pass", "Carry"])].copy()

    if len(df) == 0:
        return None

    df = ensure_xy(df)

    df["end_x"] = np.nan
    df["end_y"] = np.nan

    if "pass_end_location" in df.columns:
        pass_end = df["pass_end_location"].apply(parse_location)
        df["pass_end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["pass_end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)
        df["end_x"] = df["end_x"].fillna(df["pass_end_x"])
        df["end_y"] = df["end_y"].fillna(df["pass_end_y"])

    if "carry_end_location" in df.columns:
        carry_end = df["carry_end_location"].apply(parse_location)
        df["carry_end_x"] = carry_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["carry_end_y"] = carry_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)
        df["end_x"] = df["end_x"].fillna(df["carry_end_x"])
        df["end_y"] = df["end_y"].fillna(df["carry_end_y"])

    df = df.dropna(subset=["x", "y", "end_x", "end_y"])
    df["progression"] = df["end_x"] - df["x"]
    df = df[df["progression"] >= 15].copy()

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    passes = df[df["event_type"] == "Pass"]
    carries = df[df["event_type"] == "Carry"]

    if len(passes) > 0:
        pitch.arrows(
            passes["x"],
            passes["y"],
            passes["end_x"],
            passes["end_y"],
            ax=ax,
            width=2,
            headwidth=4,
            color=team_color,
            alpha=0.55,
            label="Progressive Pass"
        )

    if len(carries) > 0:
        pitch.arrows(
            carries["x"],
            carries["y"],
            carries["end_x"],
            carries["end_y"],
            ax=ax,
            width=2,
            headwidth=4,
            color="#111827",
            alpha=0.55,
            label="Progressive Carry"
        )

    ax.set_title(f"{player} | Progressive Actions", fontsize=15, weight="bold")
    ax.legend(loc="upper left")

    return fig


def plot_role_fit_bar(rf, player, team_color="#2563EB"):
    if len(rf) == 0:
        return None

    df = rf.copy()

    role_col = next((c for c in ["target_role", "role", "role_name"] if c in df.columns), None)
    score_col = next((c for c in ["role_fit_score", "fit_score", "score"] if c in df.columns), None)

    if role_col is None or score_col is None:
        return None

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    df = df.sort_values(score_col, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(df[role_col][::-1], df[score_col][::-1], color=team_color, alpha=0.85)
    ax.set_title(f"{player} | Role Fit", fontsize=15, weight="bold")
    ax.set_xlabel("Role Fit Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_similarity_bar(sim, player, team_color="#2563EB"):
    if len(sim) == 0:
        return None

    df = sim.copy()

    player_col = next((c for c in ["similar_player", "player_2", "similar_name"] if c in df.columns), None)
    score_col = next((c for c in ["similarity_score", "score"] if c in df.columns), None)

    if player_col is None:
        return None

    if score_col is None:
        return None

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    if "similarity_rank" in df.columns:
        df = df.sort_values("similarity_rank")
    else:
        df = df.sort_values(score_col, ascending=False)

    df = df.head(8)

    labels = df[player_col].astype(str)

    if "similar_team" in df.columns:
        labels = labels + " | " + df["similar_team"].astype(str)

    if "similar_tournament_id" in df.columns:
        labels = labels + " | " + df["similar_tournament_id"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(labels[::-1], df[score_col][::-1], color=team_color, alpha=0.85)

    ax.set_title(f"{player} | Similar Players", fontsize=15, weight="bold")
    ax.set_xlabel("Similarity")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_highlighted_matches_bar(hm, player, team_color="#2563EB"):
    if len(hm) == 0:
        return None

    df = hm.copy()

    score_col = next((c for c in ["match_score", "player_match_score", "score", "importance_score"] if c in df.columns), None)
    label_col = next((c for c in ["match_label", "opponent", "match_name"] if c in df.columns), None)

    if score_col is None:
        return None

    if label_col is None:
        if "match_id" in df.columns:
            label_col = "match_id"
        else:
            return None

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    if "match_score_rank" in df.columns:
        df = df.sort_values("match_score_rank")
    else:
        df = df.sort_values(score_col, ascending=False)

    df = df.head(8)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(df[label_col].astype(str)[::-1], df[score_col][::-1], color=team_color, alpha=0.85)
    ax.set_title(f"{player} | Highlighted Matches", fontsize=15, weight="bold")
    ax.set_xlabel("Match Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

tournament_id = st.sidebar.selectbox("Tournament", get_tournament_options(matches))
team = st.sidebar.selectbox("Team", get_team_options(player_master, tournament_id))
player = st.sidebar.selectbox("Player", get_player_options(player_master, tournament_id, team))

profile = filter_player(player_master, tournament_id, team, player)
row = get_row(profile)

flag_path, primary_color = get_team_asset(team)

player_events = player_events_filter(events, tournament_id, team, player)
event_kpis = build_player_event_kpis(player_events)


# ============================================================
# FILTERED DATA
# ============================================================

rf = player_role_fit[
    (player_role_fit["tournament_id"] == tournament_id) &
    (player_role_fit["team"] == team) &
    (player_role_fit["player"] == player)
].copy()

sim = player_similarity[
    (player_similarity["tournament_id"] == tournament_id) &
    (player_similarity["team"] == team) &
    (player_similarity["player"] == player)
].copy()

hm = highlighted_matches[
    (highlighted_matches["tournament_id"] == tournament_id) &
    (highlighted_matches["team"] == team) &
    (highlighted_matches["player"] == player)
].copy()

if "level" in hm.columns:
    hm = hm[hm["level"] == "player"].copy()


# ============================================================
# HEADER
# ============================================================

col_flag, col_title = st.columns([1, 6])

with col_flag:
    if flag_path:
        st.image(flag_path, width=85)

with col_title:
    st.markdown(
        f"<h1 style='color:{primary_color}; margin-bottom:0;'>{player}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='page-subtitle'>{team} | {tournament_id} | Player Tournament Analysis</div>",
        unsafe_allow_html=True
    )


# ============================================================
# MAIN KPI ROW
# ============================================================

if row is not None:

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        metric_card("Matches", int(safe_value(row, "matches", 0)))

    with k2:
        metric_card("Position", safe_value(row, "position", "N/A"))

    with k3:
        metric_card("Role", safe_value(row, "general_role", "N/A"))

    with k4:
        metric_card("Importance", round(safe_value(row, "importance_score", 0), 2))

    with k5:
        metric_card("Tier", safe_value(row, "importance_tier", "N/A"))

    with k6:
        metric_card("xG", round(safe_value(row, "xg", event_kpis["xg"]), 2))


# ============================================================
# EFFICIENCY KPI ROW
# ============================================================

e1, e2, e3, e4, e5, e6 = st.columns(6)

with e1:
    metric_card("Shots", event_kpis["shots"])

with e2:
    metric_card("Goals", event_kpis["goals"])

with e3:
    metric_card("Shot Acc. %", f"{event_kpis['shot_accuracy_pct']}%")

with e4:
    metric_card("Pass Acc. %", f"{event_kpis['pass_completion_pct']}%")

with e5:
    metric_card("Duel Win %", f"{event_kpis['duel_win_pct']}%")

with e6:
    metric_card("Recoveries", event_kpis["recoveries"])


# ============================================================
# TABS
# ============================================================

tab_overview, tab_attack, tab_organization, tab_defense, tab_matches, tab_similarity = st.tabs([
    "Overview",
    "Attack",
    "Organization",
    "Defense",
    "Matches",
    "Role & Similarity"
])


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown("## Player Overview")

    if row is not None:

        text_card(
            "Auto Player Summary",
            generate_player_summary(row, player, team, tournament_id, event_kpis)
        )

        info_col, radar_col, bars_col = st.columns([1, 1, 1])

        with info_col:
            text_card("General Role", safe_value(row, "general_role", "N/A"))
            text_card("Offensive Role", safe_value(row, "offensive_role", "N/A"))
            text_card("Defensive Role", safe_value(row, "defensive_role", "N/A"))

            list_card("Strengths", safe_value(row, "strengths", []))
            list_card("Weaknesses", safe_value(row, "weaknesses", []))

            if "best_role_fits" in row.index:
                text_card("Best Role Fits", safe_value(row, "best_role_fits", "N/A"))

        with radar_col:
            fig = plot_player_radar(row, event_kpis, player, team_color=primary_color)

            if fig is not None:
                st.pyplot(fig, use_container_width=True)

        with bars_col:
            fig = plot_player_profile_bars(row, event_kpis, player, team_color=primary_color)

            if fig is not None:
                st.pyplot(fig, use_container_width=True)

    else:
        st.info("No player profile available.")


# ============================================================
# ATTACK
# ============================================================

with tab_attack:

    st.markdown("## Attack")

    a1, a2, a3, a4, a5 = st.columns(5)

    with a1:
        metric_card("Goals", event_kpis["goals"])

    with a2:
        metric_card("Shots", event_kpis["shots"])

    with a3:
        metric_card("xG", round(safe_value(row, "xg", event_kpis["xg"]), 2))

    with a4:
        metric_card("Shot Acc. %", f"{event_kpis['shot_accuracy_pct']}%")

    with a5:
        metric_card("Crosses", event_kpis["crosses"])

    st.markdown("### Shooting")

    s1, s2 = st.columns(2)

    with s1:
        fig = plot_shot_map(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No shot map available.")

    with s2:
        fig = plot_player_zone_heatmap(
            player_events,
            ["Shot"],
            f"{player} | Shot Zones",
            cmap="Reds"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No shot zone data available.")

    st.markdown("### Crossing & Offensive Duels")

    c1, c2 = st.columns(2)

    with c1:
        fig = plot_cross_origin_zones(player_events, player)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No cross origin data available.")

    with c2:
        offensive_duels = ensure_xy(player_events)
        offensive_duels = offensive_duels[offensive_duels["x"].fillna(0) >= 60].copy()

        fig = plot_player_zone_heatmap(
            offensive_duels,
            ["Duel"],
            f"{player} | Offensive Duel Zones",
            cmap="Oranges"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No offensive duel data available.")


# ============================================================
# ORGANIZATION
# ============================================================

with tab_organization:

    st.markdown("## Organization & Build-Up")

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        metric_card("Passes", event_kpis["passes"])

    with o2:
        metric_card("Pass Acc. %", f"{event_kpis['pass_completion_pct']}%")

    with o3:
        metric_card("Carries", event_kpis["carries"])

    with o4:
        metric_card("Progression", round(safe_value(row, "progressive_actions", 0), 1))

    z1, z2 = st.columns(2)

    with z1:
        fig = plot_player_zone_heatmap(
            player_events,
            ["Pass"],
            f"{player} | Pass Origin Zones",
            cmap="Blues"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No pass zone data available.")

    with z2:
        fig = plot_player_zone_heatmap(
            player_events,
            ["Carry"],
            f"{player} | Carry Zones",
            cmap="Purples"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No carry zone data available.")

    st.markdown("### Progressive Actions & Receptions")

    p1, p2 = st.columns(2)

    with p1:
        fig = plot_progressive_actions(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No progressive actions available.")

    with p2:
        fig = plot_reception_zones(events, tournament_id, team, player)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No reception zones available.")


# ============================================================
# DEFENSE
# ============================================================

with tab_defense:

    st.markdown("## Defense")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        metric_card("Pressures", event_kpis["pressures"])

    with d2:
        metric_card("Recoveries", event_kpis["recoveries"])

    with d3:
        metric_card("Duels", event_kpis["duels"])

    with d4:
        metric_card("Duel Win %", f"{event_kpis['duel_win_pct']}%")

    z1, z2 = st.columns(2)

    with z1:
        fig = plot_player_zone_heatmap(
            player_events,
            ["Pressure"],
            f"{player} | Pressure Zones",
            cmap="Oranges"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No pressure zone data available.")

    with z2:
        fig = plot_player_zone_heatmap(
            player_events,
            ["Ball Recovery", "Interception"],
            f"{player} | Recovery & Interception Zones",
            cmap="Greens"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No recovery zone data available.")

    st.markdown("### Duel Zones")

    fig = plot_player_zone_heatmap(
        player_events,
        ["Duel"],
        f"{player} | Duel Zones",
        cmap="Purples"
    )

    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No duel zone data available.")


# ============================================================
# MATCHES
# ============================================================

with tab_matches:

    st.markdown("## Highlighted Matches")

    if len(hm) > 0:

        if "match_score_rank" in hm.columns:
            hm = hm.sort_values("match_score_rank")

        fig = plot_highlighted_matches_bar(hm, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

        preferred_cols = [
            "match_score_rank",
            "match_label",
            "opponent",
            "match_score",
            "minutes",
            "xg",
            "shots",
            "passes",
            "carries",
            "pressures",
            "recoveries",
            "duels"
        ]

        st.dataframe(
            compact_table(hm, preferred_cols).head(10),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No highlighted matches available.")


# ============================================================
# ROLE & SIMILARITY
# ============================================================

with tab_similarity:

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("## Role Fit")

        if len(rf) > 0:
            if "role_fit_score" in rf.columns:
                rf = rf.sort_values("role_fit_score", ascending=False)

            fig = plot_role_fit_bar(rf, player, team_color=primary_color)

            if fig is not None:
                st.pyplot(fig, use_container_width=True)

            st.dataframe(
                compact_table(
                    rf,
                    [
                        "target_role",
                        "role_fit_score",
                        "role_fit_rank",
                        "current_role",
                        "position"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No role fit data available.")

    with s2:
        st.markdown("## Similar Players")

        if len(sim) > 0:
            if "similarity_rank" in sim.columns:
                sim = sim.sort_values("similarity_rank")

            fig = plot_similarity_bar(sim, player, team_color=primary_color)

            if fig is not None:
                st.pyplot(fig, use_container_width=True)

            st.dataframe(
                compact_table(
                    sim,
                    [
                        "similarity_rank",
                        "similar_player",
                        "similar_team",
                        "similar_tournament_id",
                        "similar_position",
                        "similar_general_role",
                        "similarity_score"
                    ]
                ).head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No similar players available.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"
)