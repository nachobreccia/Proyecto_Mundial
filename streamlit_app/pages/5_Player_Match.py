import ast
from pathlib import Path
from utils.data_loader import load_all_data, load_events
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import Pitch

from utils.data_loader import load_all_data
from utils.filters import get_tournament_options, get_team_options, get_player_options
from utils.style import apply_global_style
from utils.cards import metric_card, text_card


st.set_page_config(page_title="Player Match", layout="wide")
apply_global_style()

data = load_all_data()

matches = data["matches"]

assets = data["assets"]
player_match_stats = data.get("player_match_stats_with_positions", None)

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


def remove_penalty_shootout(df):
    df = df.copy()

    if "period" in df.columns:
        df = df[df["period"].fillna(0).astype(float) <= 4].copy()

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


def build_match_label(df):
    return (
        df["match_date"].astype(str)
        + " | "
        + df["home_team"].astype(str)
        + " "
        + df["home_score"].astype(str)
        + "-"
        + df["away_score"].astype(str)
        + " "
        + df["away_team"].astype(str)
    )


def pct(numerator, denominator):
    if denominator == 0:
        return 0
    return round((numerator / denominator) * 100, 1)


def build_kpis(player_events):
    shots = len(player_events[player_events["event_type"] == "Shot"])
    passes = len(player_events[player_events["event_type"] == "Pass"])
    carries = len(player_events[player_events["event_type"] == "Carry"])
    pressures = len(player_events[player_events["event_type"] == "Pressure"])
    recoveries = len(player_events[player_events["event_type"].isin(["Ball Recovery", "Interception"])])
    duels = len(player_events[player_events["event_type"] == "Duel"])
    fouls = len(player_events[player_events["event_type"] == "Foul Committed"])
    xg = player_events["shot_statsbomb_xg"].fillna(0).sum() if "shot_statsbomb_xg" in player_events.columns else 0

    goals = 0
    shots_on_target = 0

    if "shot_outcome" in player_events.columns:
        goals = len(player_events[player_events["shot_outcome"] == "Goal"])
        shots_on_target = len(
            player_events[
                (player_events["event_type"] == "Shot") &
                (player_events["shot_outcome"].astype(str).isin(["Goal", "Saved", "Saved To Post"]))
            ]
        )

    completed_passes = 0

    if "pass_outcome" in player_events.columns:
        completed_passes = len(
            player_events[
                (player_events["event_type"] == "Pass") &
                (player_events["pass_outcome"].isna())
            ]
        )

    crosses = 0
    completed_crosses = 0

    if "pass_cross" in player_events.columns:
        cross_df = player_events[
            (player_events["event_type"] == "Pass") &
            (player_events["pass_cross"].fillna(False) == True)
        ].copy()

        crosses = len(cross_df)

        if "pass_outcome" in cross_df.columns:
            completed_crosses = len(cross_df[cross_df["pass_outcome"].isna()])

    duels_won = 0

    if "duel_outcome" in player_events.columns:
        duels_won = len(
            player_events[
                (player_events["event_type"] == "Duel") &
                (player_events["duel_outcome"].astype(str).str.contains("Won|Success", case=False, na=False))
            ]
        )

    return {
        "events": len(player_events),
        "shots": shots,
        "goals": goals,
        "xg": xg,
        "shots_on_target": shots_on_target,
        "shot_accuracy_pct": pct(shots_on_target, shots),
        "conversion_pct": pct(goals, shots),
        "passes": passes,
        "completed_passes": completed_passes,
        "pass_accuracy_pct": pct(completed_passes, passes),
        "carries": carries,
        "pressures": pressures,
        "recoveries": recoveries,
        "duels": duels,
        "duels_won": duels_won,
        "duel_win_pct": pct(duels_won, duels),
        "fouls": fouls,
        "crosses": crosses,
        "completed_crosses": completed_crosses,
        "cross_accuracy_pct": pct(completed_crosses, crosses)
    }


# ============================================================
# VISUALS
# ============================================================

def plot_action_map(player_events, player, team_color="#2563EB"):
    df = ensure_xy(player_events)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    event_colors = {
        "Shot": "#DC2626",
        "Pass": team_color,
        "Carry": "#7C3AED",
        "Pressure": "#F97316",
        "Ball Recovery": "#16A34A",
        "Interception": "#15803D",
        "Duel": "#6B7280"
    }

    for event_type, color in event_colors.items():
        plot_df = df[df["event_type"] == event_type]

        if len(plot_df) == 0:
            continue

        pitch.scatter(
            plot_df["x"],
            plot_df["y"],
            s=80,
            ax=ax,
            color=color,
            alpha=0.65,
            label=event_type
        )

    ax.set_title(f"{player} | Match Action Map", fontsize=15, weight="bold")
    ax.legend(loc="upper left")

    return fig


def plot_zone_heatmap(player_events, event_types, title, cmap="Blues"):
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
            alpha=0.5,
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
    if "pass_cross" not in player_events.columns:
        return None

    df = player_events[
        (player_events["event_type"] == "Pass") &
        (player_events["pass_cross"].fillna(False) == True)
    ].copy()

    return plot_zone_heatmap(
        df,
        ["Pass"],
        f"{player} | Cross Origin Zones",
        cmap="Blues"
    )


def plot_cross_target_zones(player_events, player):
    if "pass_cross" not in player_events.columns or "pass_end_location" not in player_events.columns:
        return None

    df = player_events[
        (player_events["event_type"] == "Pass") &
        (player_events["pass_cross"].fillna(False) == True)
    ].copy()

    if len(df) == 0:
        return None

    pass_end = df["pass_end_location"].apply(parse_location)

    df["x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap="Oranges", alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax)

    ax.set_title(f"{player} | Cross Target Zones", fontsize=15, weight="bold")

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


def plot_pass_map(player_events, player, team_color="#2563EB"):
    df = player_events[player_events["event_type"] == "Pass"].copy()

    if "pass_end_location" not in df.columns:
        return None

    df = ensure_xy(df)

    pass_end = df["pass_end_location"].apply(parse_location)

    df["end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    df = df.dropna(subset=["x", "y", "end_x", "end_y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    completed = df[df["pass_outcome"].isna()] if "pass_outcome" in df.columns else df
    incomplete = df[df["pass_outcome"].notna()] if "pass_outcome" in df.columns else pd.DataFrame()

    if len(completed) > 0:
        pitch.arrows(
            completed["x"],
            completed["y"],
            completed["end_x"],
            completed["end_y"],
            ax=ax,
            width=2,
            headwidth=4,
            color=team_color,
            alpha=0.55,
            label="Completed"
        )

    if len(incomplete) > 0:
        pitch.arrows(
            incomplete["x"],
            incomplete["y"],
            incomplete["end_x"],
            incomplete["end_y"],
            ax=ax,
            width=2,
            headwidth=4,
            color="#DC2626",
            alpha=0.45,
            label="Incomplete"
        )

    ax.set_title(f"{player} | Pass Map", fontsize=15, weight="bold")
    ax.legend(loc="upper left")

    return fig


def plot_carry_map(player_events, player, team_color="#2563EB"):
    df = player_events[player_events["event_type"] == "Carry"].copy()

    if "carry_end_location" not in df.columns:
        return None

    df = ensure_xy(df)

    carry_end = df["carry_end_location"].apply(parse_location)

    df["end_x"] = carry_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["end_y"] = carry_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    df = df.dropna(subset=["x", "y", "end_x", "end_y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    pitch.arrows(
        df["x"],
        df["y"],
        df["end_x"],
        df["end_y"],
        ax=ax,
        width=2,
        headwidth=4,
        color=team_color,
        alpha=0.55
    )

    ax.set_title(f"{player} | Carry Map", fontsize=15, weight="bold")

    return fig


def plot_aerial_duel_zones(player_events, player):
    df = player_events[player_events["event_type"] == "Duel"].copy()

    if "duel_type" in df.columns:
        df = df[df["duel_type"].astype(str).str.contains("Aerial", case=False, na=False)]
    else:
        return None

    return plot_zone_heatmap(
        df,
        ["Duel"],
        f"{player} | Aerial Duel Zones",
        cmap="Greens"
    )


def plot_minutes_timeline(player_events, player, team_color="#2563EB"):
    if len(player_events) == 0 or "minute" not in player_events.columns:
        return None

    df = player_events.copy()
    df["minute_bin"] = (df["minute"] // 15) * 15

    plot_df = (
        df.groupby("minute_bin")
        .size()
        .reset_index(name="events")
        .sort_values("minute_bin")
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(plot_df["minute_bin"].astype(str), plot_df["events"], color=team_color, alpha=0.85)

    ax.set_title(f"{player} | Event Timeline", fontsize=15, weight="bold")
    ax.set_xlabel("Minute Block")
    ax.set_ylabel("Events")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

tournament_id = st.sidebar.selectbox("Tournament", get_tournament_options(matches))
events = load_events(tournament_id)

if events.empty or "match_id" not in events.columns:
    st.error(f"No events data available for tournament: {tournament_id}")
    st.stop()
team = st.sidebar.selectbox("Team", get_team_options(events, tournament_id))
player = st.sidebar.selectbox("Player", get_player_options(events, tournament_id, team))

player_match_ids = events[
    (events["tournament_id"] == tournament_id) &
    (events["team"] == team) &
    (events["player"] == player)
]["match_id"].dropna().unique()

player_matches = matches[matches["match_id"].isin(player_match_ids)].copy()

if "match_label" not in player_matches.columns:
    player_matches["match_label"] = build_match_label(player_matches)

match_label = st.sidebar.selectbox("Match", player_matches["match_label"].tolist())

match_row = player_matches[player_matches["match_label"] == match_label].iloc[0]
match_id = match_row["match_id"]

opponent = match_row["away_team"] if team == match_row["home_team"] else match_row["home_team"]

match_events = events[events["match_id"] == match_id].copy()
match_events = remove_penalty_shootout(match_events)

player_events = match_events[
    (match_events["team"] == team) &
    (match_events["player"] == player)
].copy()

flag_path, primary_color = get_team_asset(team)

kpis = build_kpis(player_events)


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
        f"<div class='page-subtitle'>{team} vs {opponent} | {match_label} | Player Match Analysis</div>",
        unsafe_allow_html=True
    )


# ============================================================
# KPI ROW 1
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    metric_card("Events", kpis["events"])

with k2:
    metric_card("Shots", kpis["shots"])

with k3:
    metric_card("xG", round(kpis["xg"], 2))

with k4:
    metric_card("Passes", kpis["passes"])

with k5:
    metric_card("Carries", kpis["carries"])

with k6:
    metric_card("Pressures", kpis["pressures"])


# ============================================================
# KPI ROW 2
# ============================================================

e1, e2, e3, e4, e5, e6 = st.columns(6)

with e1:
    metric_card("Goals", kpis["goals"])

with e2:
    metric_card("Shot Acc. %", f"{kpis['shot_accuracy_pct']}%")

with e3:
    metric_card("Pass Acc. %", f"{kpis['pass_accuracy_pct']}%")

with e4:
    metric_card("Duel Win %", f"{kpis['duel_win_pct']}%")

with e5:
    metric_card("Crosses", kpis["crosses"])

with e6:
    metric_card("Recoveries", kpis["recoveries"])


# ============================================================
# TABS
# ============================================================

tab_overview, tab_attack, tab_build, tab_defense, tab_stats = st.tabs([
    "Overview",
    "Attack",
    "Build-Up",
    "Defense",
    "Stats & Events"
])


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown("## Player Match Summary")

    summary = (
        f"{player} recorded {kpis['events']} total events for {team} against {opponent}. "
        f"He produced {kpis['shots']} shots, {round(kpis['xg'], 2)} xG, "
        f"{kpis['passes']} passes, {kpis['carries']} carries, "
        f"{kpis['pressures']} pressures, {kpis['recoveries']} recoveries/interceptions "
        f"and {kpis['duels']} duels."
    )

    text_card("Auto Match Summary", summary)

    o1, o2 = st.columns(2)

    with o1:
        fig = plot_action_map(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No action map available.")

    with o2:
        fig = plot_minutes_timeline(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No event timeline available.")


# ============================================================
# ATTACK
# ============================================================

with tab_attack:

    st.markdown("## Attack")

    a1, a2, a3, a4, a5 = st.columns(5)

    with a1:
        metric_card("Shots", kpis["shots"])

    with a2:
        metric_card("Goals", kpis["goals"])

    with a3:
        metric_card("xG", round(kpis["xg"], 2))

    with a4:
        metric_card("Shot Acc. %", f"{kpis['shot_accuracy_pct']}%")

    with a5:
        metric_card("Cross Acc. %", f"{kpis['cross_accuracy_pct']}%")

    s1, s2 = st.columns(2)

    with s1:
        fig = plot_shot_map(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No player shots available.")

    with s2:
        fig = plot_zone_heatmap(
            player_events,
            ["Shot"],
            f"{player} | Shot Zones",
            cmap="Reds"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No shot zones available.")

    st.markdown("### Crosses & Offensive Duels")

    c1, c2 = st.columns(2)

    with c1:
        fig = plot_cross_origin_zones(player_events, player)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No cross origin zones available.")

    with c2:
        offensive_duels = ensure_xy(player_events)
        offensive_duels = offensive_duels[offensive_duels["x"].fillna(0) >= 60].copy()

        fig = plot_zone_heatmap(
            offensive_duels,
            ["Duel"],
            f"{player} | Offensive Duel Zones",
            cmap="Oranges"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No offensive duel zones available.")


# ============================================================
# BUILD-UP
# ============================================================

with tab_build:

    st.markdown("## Build-Up & Progression")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        metric_card("Passes", kpis["passes"])

    with b2:
        metric_card("Pass Acc. %", f"{kpis['pass_accuracy_pct']}%")

    with b3:
        metric_card("Carries", kpis["carries"])

    with b4:
        metric_card("Pass / Carry", round(kpis["passes"] / kpis["carries"], 2) if kpis["carries"] > 0 else 0)

    p1, p2 = st.columns(2)

    with p1:
        fig = plot_pass_map(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No pass map available.")

    with p2:
        fig = plot_carry_map(player_events, player, team_color=primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No carry map available.")

    st.markdown("### Progressive Actions")

    fig = plot_progressive_actions(player_events, player, team_color=primary_color)

    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No progressive actions available.")


# ============================================================
# DEFENSE
# ============================================================

with tab_defense:

    st.markdown("## Defense")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        metric_card("Pressures", kpis["pressures"])

    with d2:
        metric_card("Recoveries + Interceptions", kpis["recoveries"])

    with d3:
        metric_card("Duels", kpis["duels"])

    with d4:
        metric_card("Fouls", kpis["fouls"])

    r1, r2 = st.columns(2)

    with r1:
        fig = plot_zone_heatmap(
            player_events,
            ["Pressure"],
            f"{player} | Pressure Zones",
            cmap="Oranges"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No pressure zones available.")

    with r2:
        fig = plot_zone_heatmap(
            player_events,
            ["Ball Recovery", "Interception"],
            f"{player} | Recovery & Interception Zones",
            cmap="Greens"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No recovery zones available.")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        fig = plot_zone_heatmap(
            player_events,
            ["Duel"],
            f"{player} | Duel Zones",
            cmap="Purples"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No duel zones available.")

    with dcol2:
        fig = plot_zone_heatmap(
            player_events,
            ["Foul Committed"],
            f"{player} | Foul Zones",
            cmap="Greys"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No foul zones available.")


# ============================================================
# STATS & EVENTS
# ============================================================

with tab_stats:

    st.markdown("## Player Match Stats")

    if player_match_stats is not None:
        stats = player_match_stats[
            (player_match_stats["match_id"] == match_id) &
            (player_match_stats["team"] == team) &
            (player_match_stats["player"] == player)
        ].copy()

        if len(stats) > 0:
            st.dataframe(
                stats,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No player match stats available for this match.")
    else:
        st.info("player_match_stats_with_positions not loaded yet.")

    st.markdown("## Compact Event Log")

    event_cols = [
        "minute",
        "second",
        "period",
        "type",
        "event_type",
        "player",
        "team",
        "location",
        "shot_statsbomb_xg",
        "shot_outcome",
        "pass_outcome",
        "pass_length",
        "duel_outcome"
    ]

    existing_cols = [c for c in event_cols if c in player_events.columns]

    st.dataframe(
        player_events[existing_cols],
        use_container_width=True,
        hide_index=True,
        height=520
    )

# ============================================================

# FOOTER

# ============================================================
st.divider()
st.caption(

    "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"

)