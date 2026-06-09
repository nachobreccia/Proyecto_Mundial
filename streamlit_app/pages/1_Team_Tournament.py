import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_loader import load_all_data, load_events

from utils.filters import get_tournament_options, get_team_options, filter_team
from utils.style import apply_global_style
from utils.cards import metric_card, text_card, list_card, app_footer

from utils.visuals import (
    plot_team_xi_pitch,
    plot_team_shot_map,
    plot_shot_zones,
    plot_passing_zones,
    plot_carry_zones,
    plot_recovery_zones,
    plot_pressure_zones,
    plot_foul_zones,
    plot_team_similarity_bar,
    plot_team_radar_comparison,
    plot_team_dna_bars,
    plot_box_entry_zones_by_type,
    plot_cross_origin_zones,
    plot_cross_target_zones,
    plot_offensive_duel_zones,
    plot_defensive_duel_zones,
    plot_set_piece_shot_zones,
    plot_goal_timing,
    plot_long_pass_origin_zones,
    plot_long_pass_target_zones,
    plot_progressive_pass_zones,
    plot_progressive_carry_zones,
    plot_conceded_shot_zones,
    plot_aerial_duel_zones,
    plot_set_piece_profile
)


st.set_page_config(page_title="Team Tournament", layout="wide")
apply_global_style()

data = load_all_data()

matches = data["matches"]
team_master = data["team_master"]
player_master = data["player_master"]
most_frequent_xi = data["most_frequent_xi"]
team_similarity = data["team_similarity"]
assets = data["assets"]

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# HELPERS
# ============================================================

position_order = {
    "Goalkeeper": 1,
    "Right Back": 2,
    "Right Center Back": 3,
    "Center Back": 4,
    "Left Center Back": 5,
    "Left Back": 6,
    "Right Wing Back": 7,
    "Left Wing Back": 8,
    "Right Defensive Midfield": 9,
    "Center Defensive Midfield": 10,
    "Left Defensive Midfield": 11,
    "Right Center Midfield": 12,
    "Center Midfield": 13,
    "Left Center Midfield": 14,
    "Right Midfield": 15,
    "Left Midfield": 16,
    "Right Attacking Midfield": 17,
    "Center Attacking Midfield": 18,
    "Left Attacking Midfield": 19,
    "Right Wing": 20,
    "Left Wing": 21,
    "Right Center Forward": 22,
    "Center Forward": 23,
    "Left Center Forward": 24,
    "Secondary Striker": 25
}

line_mapping = {
    "Goalkeeper": "Goalkeeper",
    "Right Back": "Defenders",
    "Right Center Back": "Defenders",
    "Center Back": "Defenders",
    "Left Center Back": "Defenders",
    "Left Back": "Defenders",
    "Right Wing Back": "Defenders",
    "Left Wing Back": "Defenders",
    "Right Defensive Midfield": "Midfielders",
    "Center Defensive Midfield": "Midfielders",
    "Left Defensive Midfield": "Midfielders",
    "Right Center Midfield": "Midfielders",
    "Center Midfield": "Midfielders",
    "Left Center Midfield": "Midfielders",
    "Right Midfield": "Midfielders",
    "Left Midfield": "Midfielders",
    "Right Attacking Midfield": "Midfielders",
    "Center Attacking Midfield": "Midfielders",
    "Left Attacking Midfield": "Midfielders",
    "Right Wing": "Forwards",
    "Left Wing": "Forwards",
    "Right Center Forward": "Forwards",
    "Center Forward": "Forwards",
    "Left Center Forward": "Forwards",
    "Secondary Striker": "Forwards"
}

line_order = {
    "Goalkeeper": 1,
    "Defenders": 2,
    "Midfielders": 3,
    "Forwards": 4,
    "Other": 99
}


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


def prepare_xi_table(most_frequent_xi, tournament_id, team):
    xi = most_frequent_xi[
        (most_frequent_xi["tournament_id"] == tournament_id) &
        (most_frequent_xi["team"] == team)
    ].copy()

    if len(xi) == 0:
        return xi

    xi["position_order"] = xi["position"].map(position_order).fillna(99)

    sort_cols = ["starts"]
    ascending_values = [False]

    if "importance_score" in xi.columns:
        sort_cols.append("importance_score")
        ascending_values.append(False)

    xi = xi.sort_values(sort_cols, ascending=ascending_values).copy()

    selected_rows = []
    used_positions = set()
    used_players = set()

    for _, row in xi.iterrows():
        if row["position"] not in used_positions and row["player"] not in used_players:
            selected_rows.append(row)
            used_positions.add(row["position"])
            used_players.add(row["player"])

        if len(selected_rows) == 11:
            break

    if len(selected_rows) < 11:
        for _, row in xi.iterrows():
            if row["player"] not in used_players:
                selected_rows.append(row)
                used_players.add(row["player"])

            if len(selected_rows) == 11:
                break

    xi = pd.DataFrame(selected_rows)

    if len(xi) == 0:
        return xi

    xi["position_order"] = xi["position"].map(position_order).fillna(99)
    xi = xi.sort_values("position_order").head(11).copy()
    xi["#"] = range(1, len(xi) + 1)

    return xi


def prepare_squad_table(player_master, tournament_id, team):
    squad = player_master[
        (player_master["tournament_id"] == tournament_id) &
        (player_master["team"] == team)
    ].copy()

    if len(squad) == 0:
        return squad

    squad["line"] = squad["position"].map(line_mapping).fillna("Other")
    squad["line_order"] = squad["line"].map(line_order).fillna(99)
    squad["position_order"] = squad["position"].map(position_order).fillna(99)

    sort_cols = ["line_order", "position_order"]
    ascending_values = [True, True]

    if "importance_score" in squad.columns:
        sort_cols.append("importance_score")
        ascending_values.append(False)

    return squad.sort_values(sort_cols, ascending=ascending_values)


def render_top_player_cards(df, value_col, subtitle_col="general_role", top_n=5):
    if len(df) == 0 or value_col not in df.columns:
        return

    players = df.sort_values(value_col, ascending=False).head(top_n)
    cols = st.columns(top_n)

    for i, (_, p) in enumerate(players.iterrows()):
        with cols[i]:
            metric_card(
                p["player"],
                round(p.get(value_col, 0), 1),
                p.get(subtitle_col, "")
            )


def generate_team_summary(row, team, tournament_id):
    scores = {
        "attacking threat": row.get("attack_score", 0),
        "build-up control": row.get("build_up_score", 0),
        "defensive activity": row.get("defense_score", 0),
        "directness": row.get("directness_score", 0)
    }

    main_strength = max(scores, key=scores.get)

    return (
        f"{team} shows a {row.get('general_style', 'N/A')} profile in {tournament_id}. "
        f"Offensively, the team is classified as {row.get('offensive_style', 'N/A')}, "
        f"while defensively it profiles as {row.get('defensive_style', 'N/A')}. "
        f"The strongest team DNA indicator is {main_strength}."
    )


def count_box_entries(events, tournament_id, team):
    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"].isin(["Pass", "Carry"]))
    ].copy()

    if len(df) == 0:
        return 0, 0, 0

    def parse_loc(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                return None
        return None

    loc = df["location"].apply(parse_loc)
    df["x"] = loc.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else None)
    df["y"] = loc.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else None)

    df["end_x"] = None
    df["end_y"] = None

    if "pass_end_location" in df.columns:
        pass_end = df["pass_end_location"].apply(parse_loc)
        df["pass_end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else None)
        df["pass_end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else None)
        df["end_x"] = df["end_x"].fillna(df["pass_end_x"])
        df["end_y"] = df["end_y"].fillna(df["pass_end_y"])

    if "carry_end_location" in df.columns:
        carry_end = df["carry_end_location"].apply(parse_loc)
        df["carry_end_x"] = carry_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else None)
        df["carry_end_y"] = carry_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else None)
        df["end_x"] = df["end_x"].fillna(df["carry_end_x"])
        df["end_y"] = df["end_y"].fillna(df["carry_end_y"])

    entries = df[
        (df["end_x"] >= 102) &
        (df["end_y"] >= 18) &
        (df["end_y"] <= 62) &
        ~(
            (df["x"] >= 102) &
            (df["y"] >= 18) &
            (df["y"] <= 62)
        )
    ].copy()

    pass_entries = len(entries[entries["type"] == "Pass"])
    carry_entries = len(entries[entries["type"] == "Carry"])

    crosses = 0
    if "pass_cross" in entries.columns:
        crosses = len(entries[entries["pass_cross"].fillna(False) == True])

    return pass_entries, carry_entries, crosses


def get_team_match_ids(matches, tournament_id, team):
    team_matches = matches[
        (matches["tournament_id"] == tournament_id) &
        (
            (matches["home_team"] == team) |
            (matches["away_team"] == team)
        )
    ].copy()

    return team_matches["match_id"].dropna().unique()


def plot_goal_timing_team_tournament(events, matches, tournament_id, team, team_color="#2563EB"):
    match_ids = get_team_match_ids(matches, tournament_id, team)

    shots = events[
        (events["match_id"].isin(match_ids)) &
        (events["type"] == "Shot") &
        (events["shot_outcome"] == "Goal")
    ].copy()

    if len(shots) == 0 or "minute" not in shots.columns:
        return None

    shots["goal_type"] = shots["team"].apply(
        lambda x: "Goals For" if x == team else "Goals Against"
    )

    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]

    shots["period_bin"] = pd.cut(
        shots["minute"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    pivot = (
        shots
        .groupby(["period_bin", "goal_type"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(labels, fill_value=0)
    )

    for col in ["Goals For", "Goals Against"]:
        if col not in pivot.columns:
            pivot[col] = 0

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(labels))
    width = 0.36

    ax.bar(
        x - width / 2,
        pivot["Goals For"],
        width,
        label="Goals For",
        color=team_color,
        alpha=0.9
    )

    ax.bar(
        x + width / 2,
        pivot["Goals Against"],
        width,
        label="Goals Against",
        color="#DC2626",
        alpha=0.9
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(f"{team} | Goal Timing", fontsize=15, weight="bold")
    ax.set_xlabel("Match Period")
    ax.set_ylabel("Goals")
    ax.legend(loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_team_style_profile(events, matches, tournament_id, team, team_color="#2563EB"):
    match_ids = get_team_match_ids(matches, tournament_id, team)

    df = events[
        (events["match_id"].isin(match_ids)) &
        (events["team"] == team)
    ].copy()

    if len(df) == 0:
        return None

    crosses = 0
    if "pass_cross" in df.columns:
        crosses = len(
            df[
                (df["type"] == "Pass") &
                (df["pass_cross"].fillna(False) == True)
            ]
        )

    progressive_passes = 0
    if "pass_length" in df.columns:
        progressive_passes = len(
            df[
                (df["type"] == "Pass") &
                (df["pass_length"].fillna(0) >= 30)
            ]
        )

    values = {
        "Shots": len(df[df["type"] == "Shot"]),
        "Passes": len(df[df["type"] == "Pass"]),
        "Carries": len(df[df["type"] == "Carry"]),
        "Crosses": crosses,
        "Progressive Passes": progressive_passes,
        "Pressures": len(df[df["type"] == "Pressure"]),
        "Recoveries": len(df[df["type"].isin(["Ball Recovery", "Interception"])])
    }

    plot_df = pd.DataFrame({
        "metric": list(values.keys()),
        "value": list(values.values())
    }).sort_values("value", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        plot_df["metric"],
        plot_df["value"],
        color=team_color,
        alpha=0.85
    )

    ax.set_title(f"{team} | Team Style Profile", fontsize=15, weight="bold")
    ax.set_xlabel("Events")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def get_team_transition_data(events, tournament_id, team):
    df = events[events["tournament_id"] == tournament_id].copy()

    if "index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "event_index"})
        idx_col = "event_index"
    else:
        idx_col = "index"

    rows_passes_before_shot = []
    rows_recovery_to_shot = []

    for match_id, match_df in df.groupby("match_id"):
        match_df = match_df.sort_values(idx_col)

        shots = match_df[
            (match_df["team"] == team) &
            (match_df["type"] == "Shot")
        ].copy()

        for _, shot in shots.iterrows():
            previous = match_df[
                (match_df[idx_col] < shot[idx_col]) &
                (match_df[idx_col] >= shot[idx_col] - 25)
            ]

            passes = len(
                previous[
                    (previous["team"] == team) &
                    (previous["type"] == "Pass")
                ]
            )

            rows_passes_before_shot.append(passes)

        recoveries = match_df[
            (match_df["team"] == team) &
            (match_df["type"].isin(["Ball Recovery", "Interception"]))
        ].copy()

        for _, recovery in recoveries.iterrows():
            next_shots = shots[shots[idx_col] > recovery[idx_col]]

            if len(next_shots) == 0:
                continue

            next_shot = next_shots.iloc[0]

            if "minute" not in recovery.index or "minute" not in next_shot.index:
                continue

            seconds = (next_shot["minute"] - recovery["minute"]) * 60

            if 0 <= seconds <= 30:
                rows_recovery_to_shot.append(seconds)

    return rows_passes_before_shot, rows_recovery_to_shot


def plot_passes_before_shot_buckets(passes_before_shot, team, team_color="#2563EB"):
    if len(passes_before_shot) == 0:
        return None

    bins = [0, 2, 4, 6, 8, 100]
    labels = ["0-2", "3-4", "5-6", "7-8", "9+"]

    df = pd.DataFrame({"passes": passes_before_shot})
    df["bucket"] = pd.cut(df["passes"], bins=bins, labels=labels, include_lowest=True, right=True)

    plot_df = df["bucket"].value_counts().reindex(labels, fill_value=0).reset_index()
    plot_df.columns = ["bucket", "shots"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(plot_df["bucket"], plot_df["shots"], color=team_color, alpha=0.85)

    ax.set_title(f"{team} | Passes Before Shot", fontsize=15, weight="bold")
    ax.set_xlabel("Passes before shot")
    ax.set_ylabel("Shots")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_recovery_to_shot_buckets(recovery_to_shot, team, team_color="#2563EB"):
    if len(recovery_to_shot) == 0:
        return None

    bins = [0, 5, 10, 15, 20, 30]
    labels = ["0-5s", "5-10s", "10-15s", "15-20s", "20-30s"]

    df = pd.DataFrame({"seconds": recovery_to_shot})
    df["bucket"] = pd.cut(df["seconds"], bins=bins, labels=labels, include_lowest=True, right=True)

    plot_df = df["bucket"].value_counts().reindex(labels, fill_value=0).reset_index()
    plot_df.columns = ["bucket", "sequences"]

    plot_df["percentage"] = (plot_df["sequences"] / plot_df["sequences"].sum() * 100).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        plot_df["bucket"][::-1],
        plot_df["percentage"][::-1],
        color=team_color,
        alpha=0.85
    )

    ax.set_title(f"{team} | Recovery to Shot Speed", fontsize=15, weight="bold")
    ax.set_xlabel("% of sequences")

    for i, value in enumerate(plot_df["percentage"][::-1]):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=9, weight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_similarity_bar_safe(team_similarity, tournament_id, team, primary_color):
    try:
        return plot_team_similarity_bar(team_similarity, tournament_id, team, team_color=primary_color)
    except TypeError:
        return plot_team_similarity_bar(team_similarity, tournament_id, team)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

tournament_id = st.sidebar.selectbox("Tournament", get_tournament_options(matches))

events = load_events(tournament_id)

if events.empty or "match_id" not in events.columns:
    st.error(f"No events data available for tournament: {tournament_id}")
    st.stop()

team = st.sidebar.selectbox("Team", get_team_options(team_master, tournament_id))

team_row = filter_team(team_master, tournament_id, team)
flag_path, primary_color = get_team_asset(team)


# ============================================================
# HEADER
# ============================================================

col_flag, col_title, col_empty = st.columns([1, 6, 1])

with col_flag:
    if flag_path:
        st.image(flag_path, width=90)

with col_title:
    st.markdown(
        f"<h1 style='color:{primary_color}; margin-bottom:0; text-align:center;'>{team}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='page-subtitle' style='text-align:center;'>{tournament_id} · Team Tournament Analysis</div>",
        unsafe_allow_html=True
    )


# ============================================================
# KPI ROW
# ============================================================

if len(team_row) > 0:
    row = team_row.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Matches", int(row.get("matches", 0)))

    with c2:
        metric_card("Wins", int(row.get("wins", 0)))

    with c3:
        metric_card("Losses", int(row.get("losses", 0)))

    with c4:
        metric_card("Goals For", int(row.get("goals_for", row.get("goals", 0))))

    with c5:
        metric_card("Goals Against", int(row.get("goals_against", 0)))

    c6, c7, c8, c9 = st.columns(4)

    with c6:
        metric_card("Attack Score", round(row.get("attack_score", 0), 1))

    with c7:
        metric_card("Build-Up", round(row.get("build_up_score", 0), 1))

    with c8:
        metric_card("Defense", round(row.get("defense_score", 0), 1))

    with c9:
        metric_card("Directness", round(row.get("directness_score", 0), 1))


# ============================================================
# TABS
# ============================================================

tab_overview, tab_build, tab_attack, tab_defense, tab_transitions, tab_setpieces, tab_players, tab_similarity = st.tabs([
    "Overview",
    "Build-Up",
    "Attack",
    "Defense",
    "Transitions",
    "Set Pieces",
    "Players",
    "Similarity"
])


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown("### Team Structure")

    pitch_col, xi_col = st.columns([1.05, 1])

    with pitch_col:
        fig = plot_team_xi_pitch(
            most_frequent_xi,
            tournament_id,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with xi_col:
        xi = prepare_xi_table(most_frequent_xi, tournament_id, team)

        if len(xi) > 0:
            xi_cols = [
                "#",
                "player",
                "position",
                "general_role",
                "offensive_role",
                "defensive_role",
                "starts",
                "importance_score",
                "importance_tier"
            ]

            existing_xi_cols = [c for c in xi_cols if c in xi.columns]

            st.dataframe(
                xi[existing_xi_cols],
                use_container_width=True,
                hide_index=True,
                height=420
            )
        else:
            st.info("No XI table available.")

    st.markdown("### Tactical Summary & Team DNA")

    summary_col, dna_col = st.columns([1, 1])

    with summary_col:

        if len(team_row) > 0:
            row = team_row.iloc[0]

            text_card(
                "Auto Scouting Summary",
                generate_team_summary(row, team, tournament_id)
            )

            text_card(
                "General Style",
                row.get("general_style", "N/A")
            )

            text_card(
                "Offensive Style",
                row.get("offensive_style", "N/A")
            )

            text_card(
                "Defensive Style",
                row.get("defensive_style", "N/A")
            )

            if "strengths" in row.index:
                list_card(
                    "Team Strengths",
                    row.get("strengths", [])
                )

            if "weaknesses" in row.index:
                list_card(
                    "Team Weaknesses",
                    row.get("weaknesses", [])
                )

    with dna_col:
        fig = plot_team_dna_bars(
            team_master,
            tournament_id,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Match Dynamics")

    md1, md2 = st.columns(2)

    with md1:
        fig = plot_goal_timing_team_tournament(
            events,
            matches,
            tournament_id,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No goal timing available.")

    with md2:
        fig = plot_team_style_profile(
            events,
            matches,
            tournament_id,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No team style profile available.")

    st.markdown("### Key Players")

    key_players = player_master[
        (player_master["tournament_id"] == tournament_id) &
        (player_master["team"] == team)
    ].copy()

    render_top_player_cards(key_players, "importance_score")


# ============================================================
# BUILD-UP
# ============================================================

with tab_build:

    st.markdown("### Build-Up & Progression")

    if len(team_row) > 0:
        row = team_row.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Passes / Match", round(row.get("passes_per_match", 0), 2))

        with c2:
            metric_card("Pass Accuracy", round(row.get("avg_pass_completion_pct", 0), 1))

        with c3:
            metric_card("Tempo", round(row.get("tempo", 0), 2))

        with c4:
            metric_card("Build-Up Score", round(row.get("build_up_score", 0), 1))

    b1, b2 = st.columns(2)

    with b1:
        fig = plot_passing_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with b2:
        fig = plot_carry_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Long Passing")

    lp1, lp2 = st.columns(2)

    with lp1:
        fig = plot_long_pass_origin_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with lp2:
        fig = plot_long_pass_target_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Progression")

    pr1, pr2 = st.columns(2)

    with pr1:
        fig = plot_progressive_pass_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with pr2:
        fig = plot_progressive_carry_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Best Build-Up Players")

    control_players = player_master[
        (player_master["tournament_id"] == tournament_id) &
        (player_master["team"] == team)
    ].copy()

    render_top_player_cards(control_players, "passing_volume")


# ============================================================
# ATTACK
# ============================================================

with tab_attack:

    st.markdown("### Attacking Identity")

    pass_entries, carry_entries, cross_entries = count_box_entries(
        events,
        tournament_id,
        team
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        metric_card("Passes Into Box", pass_entries)

    with a2:
        metric_card("Carries Into Box", carry_entries)

    with a3:
        metric_card("Crosses Into Box", cross_entries)

    with a4:
        metric_card("Total Box Entries", pass_entries + carry_entries)

    st.markdown("### Shooting")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig = plot_team_shot_map(
            events,
            tournament_id,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r1c2:
        fig = plot_shot_zones(
            events,
            tournament_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Crossing")

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig = plot_cross_origin_zones(
            events,
            tournament_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r2c2:
        fig = plot_cross_target_zones(
            events,
            tournament_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Penalty Area Access")

    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig = plot_box_entry_zones_by_type(
            events,
            tournament_id,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r3c2:
        fig = plot_offensive_duel_zones(
            events,
            tournament_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Most Dangerous Players")

    attacking_players = player_master[
        (player_master["tournament_id"] == tournament_id) &
        (player_master["team"] == team)
    ]

    render_top_player_cards(attacking_players, "attacking_volume")


# ============================================================
# DEFENSE
# ============================================================

with tab_defense:

    st.markdown("### Defensive Structure")

    if len(team_row) > 0:
        row = team_row.iloc[0]

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            metric_card("Pressures / Match", round(row.get("pressures_per_match", 0), 2))

        with d2:
            metric_card("Recoveries / Match", round(row.get("recoveries_per_match", 0), 2))

        with d3:
            metric_card("Defensive Activity", round(row.get("defensive_activity", 0), 2))

        with d4:
            metric_card("Defense Score", round(row.get("defense_score", 0), 1))

    st.markdown("### Pressing & Recoveries")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig = plot_recovery_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r1c2:
        fig = plot_pressure_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Defensive Duels & Fouls")

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig = plot_defensive_duel_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r2c2:
        fig = plot_foul_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Defensive Vulnerability")

    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig = plot_conceded_shot_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r3c2:
        fig = plot_aerial_duel_zones(events, tournament_id, team)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Defensive Leaders")

    defensive_players = player_master[
        (player_master["tournament_id"] == tournament_id) &
        (player_master["team"] == team)
    ]

    render_top_player_cards(defensive_players, "defensive_volume")


# ============================================================
# TRANSITIONS
# ============================================================

with tab_transitions:

    st.markdown("### Transition Profile")

    passes_before_shot, recovery_to_shot = get_team_transition_data(
        events,
        tournament_id,
        team
    )

    avg_passes = round(np.mean(passes_before_shot), 2) if len(passes_before_shot) > 0 else 0
    median_passes = round(np.median(passes_before_shot), 2) if len(passes_before_shot) > 0 else 0
    avg_recovery_speed = round(np.mean(recovery_to_shot), 2) if len(recovery_to_shot) > 0 else 0

    fast_attacks_pct = 0

    if len(recovery_to_shot) > 0:
        fast_attacks_pct = round(
            len([x for x in recovery_to_shot if x <= 10]) / len(recovery_to_shot) * 100,
            1
        )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card("Avg Passes Before Shot", avg_passes)

    with k2:
        metric_card("Median Passes", median_passes)

    with k3:
        metric_card("Avg Recovery → Shot", f"{avg_recovery_speed}s")

    with k4:
        metric_card("Fast Attacks ≤10s", f"{fast_attacks_pct}%")

    text_card(
        "Transition Reading",
        "This section evaluates how directly the team attacks before shooting and how quickly it creates shots after recoveries."
    )

    tr1, tr2 = st.columns(2)

    with tr1:
        fig = plot_passes_before_shot_buckets(
            passes_before_shot,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No passes-before-shot data available.")

    with tr2:
        fig = plot_recovery_to_shot_buckets(
            recovery_to_shot,
            team,
            team_color=primary_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No recovery-to-shot data available.")


# ============================================================
# SET PIECES
# ============================================================

with tab_setpieces:

    st.markdown("### Set Pieces")

    team_events = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team)
    ].copy()

    if "play_pattern" in team_events.columns:
        corners = len(team_events[team_events["play_pattern"] == "From Corner"])
        free_kicks = len(team_events[team_events["play_pattern"] == "From Free Kick"])
        throw_ins = len(team_events[team_events["play_pattern"] == "From Throw In"])
    else:
        corners = 0
        free_kicks = 0
        throw_ins = 0

    s1, s2, s3 = st.columns(3)

    with s1:
        metric_card("Corner Events", corners)

    with s2:
        metric_card("Free Kick Events", free_kicks)

    with s3:
        metric_card("Throw-In Events", throw_ins)

    sp1, sp2 = st.columns(2)

    with sp1:
        fig = plot_set_piece_shot_zones(events, tournament_id, team)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No set piece shot zones available.")

    with sp2:
        fig = plot_set_piece_profile(events, tournament_id, team)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No set piece profile available.")

    st.markdown("### Main Set Piece Takers")

    taker_col1, taker_col2 = st.columns(2)

    with taker_col1:
        st.markdown("#### Corner Takers")

        if "play_pattern" in team_events.columns and "player" in team_events.columns:
            corner_df = team_events[team_events["play_pattern"] == "From Corner"].copy()

            if len(corner_df) > 0:
                corner_takers = (
                    corner_df
                    .dropna(subset=["player"])
                    .groupby("player")
                    .size()
                    .reset_index(name="corners")
                    .sort_values("corners", ascending=False)
                    .head(10)
                )

                st.dataframe(corner_takers, use_container_width=True, hide_index=True)
            else:
                st.info("No corner takers available.")
        else:
            st.info("Corner taker data not available.")

    with taker_col2:
        st.markdown("#### Free Kick Takers")

        if "play_pattern" in team_events.columns and "player" in team_events.columns:
            free_kick_df = team_events[team_events["play_pattern"] == "From Free Kick"].copy()

            if len(free_kick_df) > 0:
                free_kick_takers = (
                    free_kick_df
                    .dropna(subset=["player"])
                    .groupby("player")
                    .size()
                    .reset_index(name="free_kicks")
                    .sort_values("free_kicks", ascending=False)
                    .head(10)
                )

                st.dataframe(free_kick_takers, use_container_width=True, hide_index=True)
            else:
                st.info("No free kick takers available.")
        else:
            st.info("Free kick taker data not available.")

    st.markdown("### Main Set Piece Targets")

    if "play_pattern" in team_events.columns and "pass_recipient" in team_events.columns:
        target_df = team_events[
            team_events["play_pattern"].isin([
                "From Corner",
                "From Free Kick",
                "From Throw In"
            ])
        ].copy()

        target_df = target_df.dropna(subset=["pass_recipient"])

        if len(target_df) > 0:
            targets = (
                target_df
                .groupby("pass_recipient")
                .size()
                .reset_index(name="times_targeted")
                .sort_values("times_targeted", ascending=False)
                .head(10)
            )

            st.dataframe(targets, use_container_width=True, hide_index=True)
        else:
            st.info("No set piece target data available.")
    else:
        st.info("Set piece target data not available.")


# ============================================================
# PLAYERS
# ============================================================

with tab_players:

    st.markdown("### Player Roles")

    squad_table = prepare_squad_table(player_master, tournament_id, team)

    st.markdown("#### Most Important Players")
    render_top_player_cards(squad_table, "importance_score")

    st.markdown("#### Most Dangerous Players")
    render_top_player_cards(squad_table, "attacking_volume")

    st.markdown("#### Best Build-Up Players")
    render_top_player_cards(squad_table, "passing_volume")

    st.markdown("#### Defensive Anchors")
    render_top_player_cards(squad_table, "defensive_volume")

    st.markdown("### Squad Table")

    squad_cols = [
        "line",
        "position",
        "player",
        "general_role",
        "offensive_role",
        "defensive_role",
        "importance_score",
        "importance_tier",
        "matches",
        "xg",
        "shots",
        "passes",
        "pressures",
        "recoveries",
        "duels",
        "strengths",
        "weaknesses"
    ]

    existing_cols = [c for c in squad_cols if c in squad_table.columns]

    st.dataframe(
        squad_table[existing_cols],
        use_container_width=True,
        hide_index=True,
        height=520
    )


# ============================================================
# SIMILARITY
# ============================================================

with tab_similarity:

    sim1, sim2 = st.columns([1, 1])

    with sim1:
        fig = plot_similarity_bar_safe(team_similarity, tournament_id, team, primary_color)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with sim2:
        fig = plot_team_radar_comparison(team_master, team_similarity, tournament_id, team)

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Similar Teams Table")

    sim = team_similarity[
        (team_similarity["tournament_id"] == tournament_id) &
        (team_similarity["team"] == team)
    ].copy()

    sim_cols = [
        "similar_team",
        "similar_tournament_id",
        "similarity_score",
        "similarity_rank"
    ]

    existing_cols = [c for c in sim_cols if c in sim.columns]

    st.dataframe(
        sim[existing_cols].sort_values("similarity_rank").head(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

app_footer()