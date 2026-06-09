import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import Pitch

from utils.data_loader import load_all_data, load_events
from utils.style import apply_global_style
from utils.cards import metric_card, text_card


st.set_page_config(page_title="Team Match", layout="wide")
apply_global_style()

data = load_all_data()

matches = data["matches"]
assets = data.get("assets", pd.DataFrame())

BASE_DIR = Path(__file__).resolve().parents[2]


POSITION_ORDER = {
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


def get_col(df, options, default=None):
    for col in options:
        if col in df.columns:
            return col
    return default


def get_row_value(row, options, default=""):
    for col in options:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


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


def parse_any(value):
    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
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


def draw_pitch():
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#F8FAFC",
        line_color="#111827",
        linewidth=1.1
    )

    fig, ax = pitch.draw(figsize=(10, 7))
    return pitch, fig, ax


def clean_player_name(name):
    parts = str(name).split()
    return parts[-1] if len(parts) > 1 else str(name)


def get_team_asset(team_name):
    if assets is None or len(assets) == 0 or "team" not in assets.columns:
        return None, "#2563EB"

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

    for path in possible_paths:
        if path.exists():
            return str(path), primary_color

    return None, primary_color


def build_match_label(df):
    date_col = get_col(df, ["match_date", "date"], default=None)

    if date_col is None:
        date_part = "Match"
    else:
        date_part = df[date_col].astype(str)

    return (
        date_part
        + " | "
        + df["home_team"].astype(str)
        + " "
        + df["home_score"].astype(str)
        + "-"
        + df["away_score"].astype(str)
        + " "
        + df["away_team"].astype(str)
    )


def remove_penalty_shootout(df):
    df = df.copy()

    if "period" in df.columns:
        df = df[df["period"].fillna(0).astype(float) <= 4].copy()

    return df


def get_lineup_df(match_events, match_id, team):
    xi_events = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"].astype(str).str.lower().eq("starting xi"))
    ].copy()

    rows = []

    if len(xi_events) == 0:
        return pd.DataFrame(columns=["player", "position", "position_order"])

    row = xi_events.iloc[0]
    parsed = None

    if "tactics_lineup" in row.index:
        parsed = parse_any(row["tactics_lineup"])

    if parsed is None and "tactics" in row.index:
        tactics = parse_any(row["tactics"])
        if isinstance(tactics, dict):
            parsed = tactics.get("lineup", None)

    if isinstance(parsed, dict) and "lineup" in parsed:
        parsed = parsed["lineup"]

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                player_data = item.get("player", {})
                position_data = item.get("position", {})

                player_name = player_data.get("name", None) if isinstance(player_data, dict) else None
                position_name = position_data.get("name", "") if isinstance(position_data, dict) else ""

                if player_name:
                    rows.append({
                        "player": player_name,
                        "position": position_name,
                        "position_order": POSITION_ORDER.get(position_name, 99)
                    })

    return pd.DataFrame(rows)


def build_player_match_table(team_events):
    if len(team_events) == 0 or "player" not in team_events.columns:
        return pd.DataFrame()

    team_events = team_events.copy()

    if "shot_statsbomb_xg" not in team_events.columns:
        team_events["shot_statsbomb_xg"] = 0

    player_match = (
        team_events
        .dropna(subset=["player"])
        .groupby("player")
        .agg(
            events=("event_type", "count"),
            shots=("event_type", lambda x: (x == "Shot").sum()),
            passes=("event_type", lambda x: (x == "Pass").sum()),
            carries=("event_type", lambda x: (x == "Carry").sum()),
            recoveries=("event_type", lambda x: x.isin(["Ball Recovery", "Interception"]).sum()),
            pressures=("event_type", lambda x: (x == "Pressure").sum()),
            duels=("event_type", lambda x: (x == "Duel").sum()),
            fouls=("event_type", lambda x: (x == "Foul Committed").sum()),
            xg=("shot_statsbomb_xg", "sum")
        )
        .reset_index()
    )

    player_match["impact_score"] = (
        player_match["xg"] * 15
        + player_match["shots"] * 2
        + player_match["passes"] * 0.15
        + player_match["carries"] * 0.25
        + player_match["recoveries"] * 1.2
        + player_match["pressures"] * 0.6
        + player_match["duels"] * 0.5
    )

    return player_match.sort_values("impact_score", ascending=False)


def add_progressive_metrics_to_players(team_events, player_match):
    if len(player_match) == 0:
        return player_match

    df = team_events[team_events["type"].isin(["Pass", "Carry"])].copy()

    if len(df) == 0:
        player_match["progressive_actions"] = 0
        return player_match

    df = ensure_xy(df)
    df["end_x"] = np.nan

    if "pass_end_location" in df.columns:
        pass_end = df["pass_end_location"].apply(parse_location)
        df["pass_end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["end_x"] = df["end_x"].fillna(df["pass_end_x"])

    if "carry_end_location" in df.columns:
        carry_end = df["carry_end_location"].apply(parse_location)
        df["carry_end_x"] = carry_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["end_x"] = df["end_x"].fillna(df["carry_end_x"])

    df = df.dropna(subset=["player", "x", "end_x"])
    df["progression"] = df["end_x"] - df["x"]

    progressive = (
        df[df["progression"] >= 15]
        .groupby("player")
        .size()
        .reset_index(name="progressive_actions")
    )

    player_match = player_match.merge(progressive, on="player", how="left")
    player_match["progressive_actions"] = player_match["progressive_actions"].fillna(0)

    return player_match


def plot_match_xi_comparison(match_events, match_id, home_team, away_team, home_color="#2563EB", away_color="#DC2626"):
    home_xi = get_lineup_df(match_events, match_id, home_team)
    away_xi = get_lineup_df(match_events, match_id, away_team)

    if len(home_xi) == 0 and len(away_xi) == 0:
        return None

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    teams_data = [
        (home_team, home_xi, home_color, 0.25),
        (away_team, away_xi, away_color, 0.75)
    ]

    for team_name, xi_df, color, x in teams_data:
        ax.text(x, 0.95, team_name, ha="center", va="top", fontsize=15, weight="bold", color=color)

        if len(xi_df) == 0:
            ax.text(x, 0.50, "No Starting XI available", ha="center", va="center", fontsize=11)
        else:
            xi_df = xi_df.sort_values("position_order").copy()

            for i, (_, p) in enumerate(xi_df.head(11).iterrows()):
                text = f"{i + 1}. {clean_player_name(p['player'])} · {p['position']}"
                ax.text(x, 0.86 - i * 0.07, text, ha="center", va="center", fontsize=10)

    ax.set_title("Starting XI Comparison", fontsize=16, weight="bold")

    return fig


def plot_match_xg_flow(match_events, match_id, team, opponent, team_color="#2563EB", opponent_color="#DC2626"):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["type"] == "Shot")
    ].copy()

    df = remove_penalty_shootout(df)

    if len(df) == 0 or "minute" not in df.columns or "shot_statsbomb_xg" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.8))

    for team_name, color in [(team, team_color), (opponent, opponent_color)]:
        shots_df = df[df["team"] == team_name].copy()

        if len(shots_df) == 0:
            continue

        shots_df["shot_xg"] = shots_df["shot_statsbomb_xg"].fillna(0)
        shots_df = shots_df.sort_values("minute")
        shots_df["cumulative_xg"] = shots_df["shot_xg"].cumsum()

        ax.step(shots_df["minute"], shots_df["cumulative_xg"], where="post", linewidth=2.5, color=color, label=team_name)
        ax.scatter(shots_df["minute"], shots_df["cumulative_xg"], s=45, color=color)

    ax.set_title(f"xG Flow | {team} vs {opponent}", fontsize=15, weight="bold")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Cumulative xG")
    ax.set_xlim(0, 120)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_match_shot_map(match_events, match_id, team, team_color="#2563EB"):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Shot")
    ].copy()

    df = remove_penalty_shootout(df)
    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    if "shot_statsbomb_xg" not in df.columns:
        df["shot_statsbomb_xg"] = 0

    outcome_col = get_col(df, ["shot_outcome", "shot_outcome_name"], default=None)

    df["shot_xg"] = df["shot_statsbomb_xg"].fillna(0)

    if outcome_col:
        goals = df[df[outcome_col] == "Goal"]
        shots_df = df[df[outcome_col] != "Goal"]
    else:
        goals = pd.DataFrame()
        shots_df = df

    pitch, fig, ax = draw_pitch()

    if len(shots_df) > 0:
        pitch.scatter(
            shots_df["x"],
            shots_df["y"],
            s=shots_df["shot_xg"] * 1300 + 45,
            ax=ax,
            color=team_color,
            alpha=0.45,
            label="Shot"
        )

    if len(goals) > 0:
        pitch.scatter(
            goals["x"],
            goals["y"],
            s=goals["shot_xg"] * 1500 + 90,
            ax=ax,
            color="#DC2626",
            edgecolors="#111827",
            linewidth=1.5,
            label="Goal"
        )

    ax.set_title(
        f"{team} Shot Map | Shots {len(df)} | Goals {len(goals)} | xG {df['shot_xg'].sum():.2f}",
        fontsize=15,
        weight="bold"
    )

    ax.legend(loc="upper left")
    return fig


def plot_match_zone_heatmap(match_events, match_id, team, event_types, title, cmap="Blues"):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"].isin(event_types))
    ].copy()

    df = remove_penalty_shootout(df)
    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax, ha="center", va="center")

    ax.set_title(title, fontsize=15, weight="bold")
    return fig


def plot_box_entries(match_events, match_id, team, team_color="#2563EB"):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"].isin(["Pass", "Carry"]))
    ].copy()

    df = remove_penalty_shootout(df)
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

    if len(entries) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    passes_df = entries[entries["type"] == "Pass"]
    carries_df = entries[entries["type"] == "Carry"]

    if len(passes_df) > 0:
        pitch.arrows(
            passes_df["x"], passes_df["y"],
            passes_df["end_x"], passes_df["end_y"],
            ax=ax,
            width=2,
            headwidth=4,
            color=team_color,
            alpha=0.65,
            label="Pass"
        )

    if len(carries_df) > 0:
        pitch.arrows(
            carries_df["x"], carries_df["y"],
            carries_df["end_x"], carries_df["end_y"],
            ax=ax,
            width=2,
            headwidth=4,
            color="#111827",
            alpha=0.65,
            label="Carry"
        )

    ax.set_title(f"{team} | Box Entries", fontsize=15, weight="bold")
    ax.legend(loc="upper left")

    return fig


def plot_cross_origin_zones(match_events, match_id, team):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Pass")
    ].copy()

    df = remove_penalty_shootout(df)

    if "pass_cross" not in df.columns:
        return None

    df = df[df["pass_cross"].fillna(False) == True].copy()
    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap="Blues", alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax)

    ax.set_title(f"{team} | Cross Origin Zones", fontsize=15, weight="bold")
    return fig


def plot_cross_target_zones(match_events, match_id, team):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Pass")
    ].copy()

    df = remove_penalty_shootout(df)

    if "pass_cross" not in df.columns or "pass_end_location" not in df.columns:
        return None

    df = df[df["pass_cross"].fillna(False) == True].copy()

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

    ax.set_title(f"{team} | Cross Target Zones", fontsize=15, weight="bold")
    return fig


def plot_offensive_duel_zones(match_events, match_id, team):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Duel")
    ].copy()

    df = remove_penalty_shootout(df)
    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])
    df = df[df["x"] >= 60].copy()

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap="Oranges", alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax)

    ax.set_title(f"{team} | Offensive Duel Zones", fontsize=15, weight="bold")
    return fig


def plot_aerial_duel_zones(match_events, match_id, team):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Duel")
    ].copy()

    df = remove_penalty_shootout(df)

    if "duel_type" in df.columns:
        df = df[df["duel_type"].astype(str).str.contains("Aerial", case=False, na=False)]

    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap="Greens", alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax)

    ax.set_title(f"{team} | Aerial Duel Zones", fontsize=15, weight="bold")
    return fig


def plot_vulnerable_zones(match_events, match_id, opponent, selected_team):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == opponent) &
        (match_events["type"] == "Shot")
    ].copy()

    df = remove_penalty_shootout(df)
    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bin_statistic, ax=ax, cmap="Reds", alpha=0.78)
    pitch.label_heatmap(bin_statistic, color="#111827", fontsize=10, ax=ax)

    ax.set_title(f"{selected_team} | Vulnerable Zones - Opponent Shots", fontsize=15, weight="bold")
    return fig


def plot_pressure_height(team_events, team, team_color="#2563EB"):
    df = team_events[team_events["type"] == "Pressure"].copy()
    df = ensure_xy(df)
    df = df.dropna(subset=["x"])

    if len(df) == 0:
        return None

    low = len(df[df["x"] < 40])
    mid = len(df[(df["x"] >= 40) & (df["x"] < 80)])
    high = len(df[df["x"] >= 80])

    plot_df = pd.DataFrame({
        "Zone": ["Low Block", "Mid Block", "High Press"],
        "Pressures": [low, mid, high]
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.barh(plot_df["Zone"][::-1], plot_df["Pressures"][::-1], color=team_color, alpha=0.85)
    ax.set_title(f"{team} | Pressure Height", fontsize=15, weight="bold")
    ax.set_xlabel("Pressures")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_match_passing_network(match_events, match_id, team, team_color="#2563EB", min_passes=4):
    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Pass")
    ].copy()

    df = remove_penalty_shootout(df)

    if "pass_recipient" not in df.columns and "pass_recipient_name" in df.columns:
        df["pass_recipient"] = df["pass_recipient_name"]

    required_cols = ["player", "pass_recipient", "location", "pass_end_location"]

    if len(df) == 0 or not all(col in df.columns for col in required_cols):
        return None, pd.DataFrame()

    df = ensure_xy(df)

    pass_end = df["pass_end_location"].apply(parse_location)
    df["end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    df = df.dropna(subset=["player", "pass_recipient", "x", "y", "end_x", "end_y"])

    if len(df) == 0:
        return None, pd.DataFrame()

    lineup = get_lineup_df(match_events, match_id, team)
    starters = set(lineup["player"].tolist()) if len(lineup) > 0 else set()

    avg_positions = (
        df.groupby("player")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            passes=("type", "count")
        )
        .reset_index()
    )

    avg_positions = avg_positions.merge(
        lineup[["player", "position", "position_order"]],
        on="player",
        how="left"
    )

    avg_positions["status"] = avg_positions["player"].apply(
        lambda x: "Starter" if x in starters else "Substitute"
    )

    avg_positions["position"] = avg_positions["position"].fillna("Substitute")
    avg_positions["position_order"] = avg_positions["position_order"].fillna(99)

    starters_df_order = (
        avg_positions[avg_positions["status"] == "Starter"]
        .sort_values(["position_order", "y"], ascending=[True, True])
        .copy()
    )

    subs_df_order = (
        avg_positions[avg_positions["status"] == "Substitute"]
        .sort_values("passes", ascending=False)
        .copy()
    )

    avg_positions = pd.concat([starters_df_order, subs_df_order], ignore_index=True)
    avg_positions["number"] = range(1, len(avg_positions) + 1)

    pass_links = (
        df.groupby(["player", "pass_recipient"])
        .size()
        .reset_index(name="passes")
    )

    pass_links = pass_links[pass_links["passes"] >= min_passes]

    legend_df = avg_positions[["number", "player", "position", "status", "passes"]].copy()
    legend_df.columns = ["#", "Player", "Position", "Status", "Passes"]

    if len(pass_links) == 0:
        return None, legend_df

    pitch, fig, ax = draw_pitch()

    for _, row in pass_links.iterrows():
        passer = avg_positions[avg_positions["player"] == row["player"]]
        receiver = avg_positions[avg_positions["player"] == row["pass_recipient"]]

        if len(passer) == 0 or len(receiver) == 0:
            continue

        x1 = passer.iloc[0]["x"]
        y1 = passer.iloc[0]["y"]
        x2 = receiver.iloc[0]["x"]
        y2 = receiver.iloc[0]["y"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=max(1, row["passes"] * 0.18),
            color=team_color,
            alpha=0.35,
            zorder=2
        )

    starters_plot = avg_positions[avg_positions["status"] == "Starter"]
    subs_plot = avg_positions[avg_positions["status"] == "Substitute"]

    if len(starters_plot) > 0:
        pitch.scatter(
            starters_plot["x"],
            starters_plot["y"],
            s=starters_plot["passes"] * 14 + 160,
            ax=ax,
            color=team_color,
            edgecolors="#111827",
            linewidth=1.3,
            alpha=0.95,
            zorder=3,
            label="Starter"
        )

    if len(subs_plot) > 0:
        pitch.scatter(
            subs_plot["x"],
            subs_plot["y"],
            s=subs_plot["passes"] * 14 + 160,
            ax=ax,
            color="#6B7280",
            edgecolors="#111827",
            linewidth=1.3,
            alpha=0.95,
            zorder=3,
            label="Substitute"
        )

    for _, row in avg_positions.iterrows():
        ax.text(
            row["x"],
            row["y"],
            str(row["number"]),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
            zorder=4
        )

    ax.set_title(f"{team} | Passing Network", fontsize=15, weight="bold")
    ax.legend(loc="upper left")

    return fig, legend_df


def plot_barh(df, metric, title, color="#2563EB", top_n=8):
    if len(df) == 0 or metric not in df.columns:
        return None

    plot_df = df.sort_values(metric, ascending=False).head(top_n).copy()

    if len(plot_df) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(plot_df["player"][::-1], plot_df[metric][::-1], color=color, alpha=0.85)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(metric)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_event_volume_comparison(team_events, opponent_events, team, opponent, team_color, opponent_color):
    phases = {
        "Shots": "Shot",
        "Passes": "Pass",
        "Carries": "Carry",
        "Pressures": "Pressure",
        "Recoveries": ["Ball Recovery", "Interception"],
        "Fouls": "Foul Committed"
    }

    rows = []

    for label, event_type in phases.items():
        if isinstance(event_type, list):
            team_value = team_events["event_type"].isin(event_type).sum()
            opponent_value = opponent_events["event_type"].isin(event_type).sum()
        else:
            team_value = (team_events["event_type"] == event_type).sum()
            opponent_value = (opponent_events["event_type"] == event_type).sum()

        rows.append({
            "phase": label,
            team: team_value,
            opponent: opponent_value
        })

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(plot_df))
    width = 0.36

    ax.bar(x - width / 2, plot_df[team], width, label=team, color=team_color, alpha=0.85)
    ax.bar(x + width / 2, plot_df[opponent], width, label=opponent, color=opponent_color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["phase"], rotation=0)
    ax.set_ylabel("Events")
    ax.set_title("Event Volume Comparison", fontsize=15, weight="bold")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


st.sidebar.title("Filters")

tournament_id = st.sidebar.selectbox(
    "Tournament",
    sorted(matches["tournament_id"].dropna().unique())
)

events = load_events(tournament_id)

if events.empty or "match_id" not in events.columns:
    st.error(f"No events data available for tournament: {tournament_id}")
    st.stop()

if "event_type" not in events.columns and "type" in events.columns:
    events["event_type"] = events["type"]

if "type" not in events.columns and "event_type" in events.columns:
    events["type"] = events["event_type"]

tournament_matches = matches[matches["tournament_id"] == tournament_id].copy()

if "match_label" not in tournament_matches.columns:
    tournament_matches["match_label"] = build_match_label(tournament_matches)

match_label = st.sidebar.selectbox(
    "Match",
    tournament_matches["match_label"].tolist()
)

match_row = tournament_matches[tournament_matches["match_label"] == match_label].iloc[0]

match_id = match_row["match_id"]

home_team = match_row["home_team"]
away_team = match_row["away_team"]

team = st.sidebar.selectbox(
    "Team",
    [home_team, away_team]
)

opponent = away_team if team == home_team else home_team

team_flag, team_color = get_team_asset(team)
opponent_flag, opponent_color = get_team_asset(opponent)
home_flag, home_color = get_team_asset(home_team)
away_flag, away_color = get_team_asset(away_team)

match_events = events[events["match_id"] == match_id].copy()
match_play_events = remove_penalty_shootout(match_events)

match_events = ensure_xy(match_events)
match_play_events = ensure_xy(match_play_events)

team_events = match_play_events[match_play_events["team"] == team].copy()
opponent_events = match_play_events[match_play_events["team"] == opponent].copy()

shots = len(team_events[team_events["event_type"] == "Shot"])
opponent_shots = len(opponent_events[opponent_events["event_type"] == "Shot"])

passes = len(team_events[team_events["event_type"] == "Pass"])
pressures = len(team_events[team_events["event_type"] == "Pressure"])
recoveries = len(team_events[team_events["event_type"].isin(["Ball Recovery", "Interception"])])
carries = len(team_events[team_events["event_type"] == "Carry"])
duels = len(team_events[team_events["event_type"] == "Duel"])

xg = team_events["shot_statsbomb_xg"].fillna(0).sum() if "shot_statsbomb_xg" in team_events.columns else 0
opponent_xg = opponent_events["shot_statsbomb_xg"].fillna(0).sum() if "shot_statsbomb_xg" in opponent_events.columns else 0

if team == home_team:
    goals_for = match_row["home_score"]
    goals_against = match_row["away_score"]
else:
    goals_for = match_row["away_score"]
    goals_against = match_row["home_score"]

result = "Draw"

if goals_for > goals_against:
    result = "Win"
elif goals_for < goals_against:
    result = "Loss"

player_match = build_player_match_table(team_events)
player_match = add_progressive_metrics_to_players(team_events, player_match)

competition_name = get_row_value(match_row, ["competition_name", "competition"], "Unknown Competition")
season_name = get_row_value(match_row, ["season_name", "season"], "Unknown Season")
stage_name = get_row_value(match_row, ["competition_stage", "stage_name"], "")


header_left, header_center, header_right = st.columns([1, 6, 1])

with header_left:
    if home_flag:
        st.image(home_flag, width=90)

with header_center:
    st.markdown(
        f"""
        <h1 style='text-align:center; margin-bottom:0;'>
            {home_team} {match_row['home_score']} - {match_row['away_score']} {away_team}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='page-subtitle' style='text-align:center;'>{competition_name} | {season_name} | {stage_name}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='page-subtitle' style='text-align:center;'>Selected team: <b>{team}</b> vs {opponent}</div>",
        unsafe_allow_html=True
    )

with header_right:
    if away_flag:
        st.image(away_flag, width=90)


c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    metric_card("Result", result)

with c2:
    metric_card("Goals", int(goals_for))

with c3:
    metric_card("xG", round(xg, 2))

with c4:
    metric_card("Shots", shots)

with c5:
    metric_card("Passes", passes)

with c6:
    metric_card("Pressures", pressures)


tab_overview, tab_build, tab_attack, tab_defense, tab_players = st.tabs([
    "Overview",
    "Build-Up",
    "Attack",
    "Defense",
    "Players"
])


with tab_overview:

    st.markdown("## Match Overview")

    summary = (
        f"{team} finished the match with {shots} shots and {round(xg, 2)} xG, "
        f"compared with {opponent}'s {opponent_shots} shots and {round(opponent_xg, 2)} xG. "
        f"The selected team recorded {passes} passes, {pressures} pressures, "
        f"{recoveries} recoveries/interceptions and {duels} duels."
    )

    text_card("Match Summary", summary)

    st.markdown("### Starting XI")

    fig = plot_match_xi_comparison(
        match_events,
        match_id,
        home_team,
        away_team,
        home_color=home_color,
        away_color=away_color
    )

    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No Starting XI data available.")

    st.markdown("### Match Control")

    m1, m2 = st.columns(2)

    with m1:
        fig = plot_match_xg_flow(
            match_events,
            match_id,
            team,
            opponent,
            team_color=team_color,
            opponent_color=opponent_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No xG flow available.")

    with m2:
        fig = plot_event_volume_comparison(
            team_events,
            opponent_events,
            team,
            opponent,
            team_color,
            opponent_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Key Players")

    fig = plot_barh(
        player_match,
        "impact_score",
        f"{team} | Key Players Impact",
        color=team_color
    )

    if fig is not None:
        st.pyplot(fig, use_container_width=True)


with tab_build:

    st.markdown("## Build-Up & Control")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        metric_card("Passes", passes)

    with b2:
        metric_card("Carries", carries)

    with b3:
        metric_card("Pass / Carry Ratio", round(passes / carries, 2) if carries > 0 else 0)

    with b4:
        metric_card("Total On-Ball Events", passes + carries)

    st.markdown("### Passing Network")

    network_col, legend_col = st.columns([2.2, 1])

    with network_col:
        fig, network_legend = plot_match_passing_network(
            match_events,
            match_id,
            team,
            team_color=team_color,
            min_passes=4
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No passing network available.")

    with legend_col:
        st.markdown("#### Player Legend")

        if len(network_legend) > 0:
            st.dataframe(
                network_legend,
                use_container_width=True,
                hide_index=True,
                height=520
            )
        else:
            st.info("No legend available.")

    st.markdown("### Zones")

    z1, z2 = st.columns(2)

    with z1:
        fig = plot_match_zone_heatmap(
            match_events,
            match_id,
            team,
            ["Pass"],
            f"{team} | Pass Origin Zones",
            cmap="Blues"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with z2:
        fig = plot_match_zone_heatmap(
            match_events,
            match_id,
            team,
            ["Carry"],
            f"{team} | Carry Zones",
            cmap="Purples"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)


with tab_attack:

    st.markdown("## Offensive Analysis")

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        metric_card("Shots", shots)

    with o2:
        metric_card("xG", round(xg, 2))

    with o3:
        metric_card("Goals", int(goals_for))

    with o4:
        conversion = round((goals_for / shots) * 100, 1) if shots > 0 else 0
        metric_card("Shot Conversion", f"{conversion}%")

    s1, s2 = st.columns(2)

    with s1:
        fig = plot_match_shot_map(
            match_events,
            match_id,
            team,
            team_color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No shot map available.")

    with s2:
        fig = plot_box_entries(
            match_events,
            match_id,
            team,
            team_color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No box entries available.")

    st.markdown("### Crosses & Offensive Duels")

    a1, a2 = st.columns(2)

    with a1:
        fig = plot_cross_origin_zones(
            match_events,
            match_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No cross origin data available.")

    with a2:
        fig = plot_cross_target_zones(
            match_events,
            match_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No cross target data available.")

    st.markdown("### Offensive Duels")

    fig = plot_offensive_duel_zones(
        match_events,
        match_id,
        team
    )

    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No offensive duel data available.")


with tab_defense:

    st.markdown("## Defensive Analysis")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        metric_card("Pressures", pressures)

    with d2:
        metric_card("Recoveries + Interceptions", recoveries)

    with d3:
        metric_card("Duels", duels)

    with d4:
        metric_card("Opponent Shots", opponent_shots)

    r1, r2 = st.columns(2)

    with r1:
        fig = plot_match_zone_heatmap(
            match_events,
            match_id,
            team,
            ["Ball Recovery", "Interception"],
            f"{team} | Recovery Zones",
            cmap="Greens"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with r2:
        fig = plot_pressure_height(
            team_events,
            team,
            team_color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Defensive Risk")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        fig = plot_vulnerable_zones(
            match_events,
            match_id,
            opponent,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No vulnerable zone data available.")

    with dcol2:
        fig = plot_aerial_duel_zones(
            match_events,
            match_id,
            team
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No aerial duel data available.")


with tab_players:

    st.markdown("## Player Match Impact")

    p1, p2 = st.columns(2)

    with p1:
        fig = plot_barh(
            player_match,
            "impact_score",
            f"{team} | Key Players Impact",
            color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with p2:
        fig = plot_barh(
            player_match,
            "events",
            f"{team} | Most Involved Players",
            color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    p3, p4 = st.columns(2)

    with p3:
        fig = plot_barh(
            player_match,
            "pressures",
            f"{team} | Most Active Pressers",
            color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with p4:
        fig = plot_barh(
            player_match,
            "progressive_actions",
            f"{team} | Progressive Players",
            color=team_color
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.markdown("### Full Player Match Table")

    table_cols = [
        "player",
        "impact_score",
        "events",
        "shots",
        "xg",
        "passes",
        "carries",
        "progressive_actions",
        "pressures",
        "recoveries",
        "duels",
        "fouls"
    ]

    existing_cols = [col for col in table_cols if col in player_match.columns]

    st.dataframe(
        player_match[existing_cols],
        use_container_width=True,
        hide_index=True
    )


st.divider()
st.caption(
    "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"
)