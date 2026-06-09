import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mplsoccer import Pitch


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
        except:
            return None

    return None


def ensure_xy(df):
    df = df.copy()

    if "x" not in df.columns:
        df["x"] = np.nan

    if "y" not in df.columns:
        df["y"] = np.nan

    if "location" in df.columns:
        parsed_locations = df["location"].apply(parse_location)

        df["x"] = df["x"].fillna(
            parsed_locations.apply(
                lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan
            )
        )

        df["y"] = df["y"].fillna(
            parsed_locations.apply(
                lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan
            )
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

    if len(parts) == 1:
        return parts[0]

    return parts[-1]


# ============================================================
# XI SHAPE
# ============================================================

def plot_team_xi_pitch(most_frequent_xi, tournament_id, team, team_color="#2563EB"):

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

    df = most_frequent_xi[
        (most_frequent_xi["tournament_id"] == tournament_id) &
        (most_frequent_xi["team"] == team)
    ].copy()

    df = df[df["position_x"].notna() & df["position_y"].notna()]

    if len(df) == 0:
        return None

    df["position_order"] = df["position"].map(position_order).fillna(99)

    sort_cols = ["starts"]
    ascending_values = [False]

    if "importance_score" in df.columns:
        sort_cols.append("importance_score")
        ascending_values.append(False)

    df = df.sort_values(sort_cols, ascending=ascending_values).copy()

    selected_rows = []
    used_positions = set()
    used_players = set()

    for _, row in df.iterrows():
        if row["position"] not in used_positions and row["player"] not in used_players:
            selected_rows.append(row)
            used_positions.add(row["position"])
            used_players.add(row["player"])

        if len(selected_rows) == 11:
            break

    if len(selected_rows) < 11:
        for _, row in df.iterrows():
            if row["player"] not in used_players:
                selected_rows.append(row)
                used_players.add(row["player"])

            if len(selected_rows) == 11:
                break

    xi = pd.DataFrame(selected_rows)

    if len(xi) == 0:
        return None

    xi["position_order"] = xi["position"].map(position_order).fillna(99)
    xi = xi.sort_values("position_order").copy()
    xi["display_number"] = range(1, len(xi) + 1)

    pitch, fig, ax = draw_pitch()

    for _, row in xi.iterrows():
        pitch.scatter(
            row["position_x"],
            row["position_y"],
            s=850,
            ax=ax,
            color=team_color,
            edgecolors="#111827",
            linewidth=1.7,
            zorder=3
        )

        ax.text(
            row["position_x"],
            row["position_y"],
            str(row["display_number"]),
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="white",
            zorder=4
        )

    ax.set_title(f"{team} | Most Frequent XI", fontsize=16, weight="bold")

    return fig


# ============================================================
# TEAM SHOT MAP
# ============================================================

def plot_team_shot_map(events, tournament_id, team, team_color="#2563EB"):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Shot")
    ].copy()

    df = ensure_xy(df)
    df = df[df["x"].notna() & df["y"].notna()]

    if len(df) == 0:
        return None

    df["shot_xg"] = df["shot_statsbomb_xg"].fillna(0)

    pitch, fig, ax = draw_pitch()

    goals = df[df["shot_outcome"] == "Goal"]
    shots = df[df["shot_outcome"] != "Goal"]

    pitch.scatter(
        shots["x"],
        shots["y"],
        s=shots["shot_xg"] * 1300 + 45,
        ax=ax,
        color=team_color,
        alpha=0.45,
        label="Shot"
    )

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


# ============================================================
# GENERIC ZONE HEATMAP
# ============================================================

def plot_zone_heatmap(events, tournament_id, team, event_types, title, cmap="Blues"):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"].isin(event_types))
    ].copy()

    df = ensure_xy(df)
    df = df[df["x"].notna() & df["y"].notna()]

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(
        df["x"],
        df["y"],
        statistic="count",
        bins=(6, 4)
    )

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap=cmap,
        alpha=0.78
    )

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


def plot_shot_zones(events, tournament_id, team):
    return plot_zone_heatmap(events, tournament_id, team, ["Shot"], f"{team} | Shot Zones", cmap="Reds")


def plot_passing_zones(events, tournament_id, team):
    return plot_zone_heatmap(events, tournament_id, team, ["Pass"], f"{team} | Pass Origin Zones", cmap="Blues")


def plot_carry_zones(events, tournament_id, team):
    return plot_zone_heatmap(events, tournament_id, team, ["Carry"], f"{team} | Carry Zones", cmap="Purples")


def plot_recovery_zones(events, tournament_id, team):
    return plot_zone_heatmap(events, tournament_id, team, ["Ball Recovery", "Interception"], f"{team} | Recovery Zones", cmap="Greens")


def plot_pressure_zones(events, tournament_id, team):
    return plot_zone_heatmap(events, tournament_id, team, ["Pressure"], f"{team} | Pressure Zones", cmap="Oranges")


def plot_foul_zones(events, tournament_id, team):
    return plot_zone_heatmap(events, tournament_id, team, ["Foul Committed"], f"{team} | Foul Zones", cmap="Greys")


# ============================================================
# SIMILARITY BAR
# ============================================================

def plot_team_similarity_bar(team_similarity, tournament_id, team):

    df = team_similarity[
        (team_similarity["tournament_id"] == tournament_id) &
        (team_similarity["team"] == team)
    ].copy()

    df = df.sort_values("similarity_rank").head(8)

    if len(df) == 0:
        return None

    labels = df["similar_team"].astype(str)

    if "similar_tournament_id" in df.columns:
        labels = labels + " | " + df["similar_tournament_id"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(labels[::-1], df["similarity_score"][::-1])

    ax.set_title(f"Most Similar Teams to {team}", fontsize=15, weight="bold")
    ax.set_xlabel("Similarity Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# TEAM RADAR COMPARISON
# ============================================================

def plot_team_radar_comparison(team_master, team_similarity, tournament_id, team):

    sim = team_similarity[
        (team_similarity["tournament_id"] == tournament_id) &
        (team_similarity["team"] == team)
    ].sort_values("similarity_rank").head(3)

    teams_to_plot = [team] + sim["similar_team"].tolist()

    df = team_master[
        team_master["team"].isin(teams_to_plot)
    ].copy()

    metrics = [
        "attack_score",
        "build_up_score",
        "defense_score",
        "directness_score"
    ]

    labels = [
        "Attack",
        "Build-Up",
        "Defense",
        "Directness"
    ]

    df = df.dropna(subset=metrics)

    if len(df) == 0:
        return None

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for _, row in df.iterrows():
        values = row[metrics].astype(float).tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=f'{row["team"]} | {row["tournament_id"]}'
        )

        ax.fill(
            angles,
            values,
            alpha=0.08
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, weight="bold")
    ax.set_ylim(0, 100)

    ax.set_title(f"{team} vs Similar Teams", fontsize=15, weight="bold", pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    return fig


# ============================================================
# TEAM DNA BARS
# ============================================================

def plot_team_dna_bars(team_master, tournament_id, team, team_color="#2563EB"):

    df = team_master[
        (team_master["tournament_id"] == tournament_id) &
        (team_master["team"] == team)
    ].copy()

    if len(df) == 0:
        return None

    row = df.iloc[0]

    metrics = {
        "Attack": row.get("attack_score", 0),
        "Build-Up": row.get("build_up_score", 0),
        "Defense": row.get("defense_score", 0),
        "Directness": row.get("directness_score", 0)
    }

    fig, ax = plt.subplots(figsize=(8, 3.8))

    labels = list(metrics.keys())
    values = list(metrics.values())

    ax.barh(labels[::-1], values[::-1], color=team_color)
    ax.set_xlim(0, 100)

    ax.set_title(f"{team} | Team DNA", fontsize=15, weight="bold")
    ax.set_xlabel("Score")

    for i, value in enumerate(values[::-1]):
        ax.text(
            value + 1,
            i,
            f"{value:.1f}",
            va="center",
            fontsize=10,
            weight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# MATCH xG FLOW
# ============================================================

def plot_match_xg_flow(events, match_id, team):

    df = events[
        (events["match_id"] == match_id) &
        (events["team"] == team) &
        (events["type"] == "Shot")
    ].copy()

    if len(df) == 0 or "minute" not in df.columns:
        return None

    df["shot_xg"] = df["shot_statsbomb_xg"].fillna(0)
    df = df.sort_values("minute")
    df["cumulative_xg"] = df["shot_xg"].cumsum()

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.step(df["minute"], df["cumulative_xg"], where="post", linewidth=2)
    ax.scatter(df["minute"], df["cumulative_xg"], s=45)

    ax.set_title(f"{team} | xG Flow", fontsize=15, weight="bold")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Cumulative xG")
    ax.set_xlim(0, max(95, df["minute"].max() + 5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# MATCH EVENT VOLUME
# ============================================================

def plot_match_event_volume(events, match_id, team):

    df = events[
        (events["match_id"] == match_id) &
        (events["team"] == team)
    ].copy()

    if len(df) == 0:
        return None

    event_groups = {
        "Shots": ["Shot"],
        "Passes": ["Pass"],
        "Carries": ["Carry"],
        "Pressures": ["Pressure"],
        "Recoveries": ["Ball Recovery", "Interception"],
        "Fouls": ["Foul Committed"]
    }

    values = []

    for label, types in event_groups.items():
        values.append({
            "phase": label,
            "count": df[df["type"].isin(types)].shape[0]
        })

    plot_df = pd.DataFrame(values)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(plot_df["phase"], plot_df["count"])

    ax.set_title(f"{team} | Event Volume by Phase", fontsize=15, weight="bold")
    ax.set_ylabel("Events")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# BOX ENTRY ZONES MATCH
# ============================================================

def plot_box_entry_zones(events, match_id, team):

    df = events[
        (events["match_id"] == match_id) &
        (events["team"] == team) &
        (events["type"].isin(["Pass", "Carry"]))
    ].copy()

    df = ensure_xy(df)

    if "pass_end_location" in df.columns:
        pass_end = df["pass_end_location"].apply(parse_location)
        df["end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    if "carry_end_location" in df.columns:
        carry_end = df["carry_end_location"].apply(parse_location)
        df["carry_end_x"] = carry_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["carry_end_y"] = carry_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

        df["end_x"] = df["end_x"].fillna(df["carry_end_x"]) if "end_x" in df.columns else df["carry_end_x"]
        df["end_y"] = df["end_y"].fillna(df["carry_end_y"]) if "end_y" in df.columns else df["carry_end_y"]

    df = df[df["x"].notna() & df["y"].notna() & df["end_x"].notna() & df["end_y"].notna()]

    box_entries = df[
        (df["end_x"] >= 102) &
        (df["end_y"] >= 18) &
        (df["end_y"] <= 62) &
        ~(
            (df["x"] >= 102) &
            (df["y"] >= 18) &
            (df["y"] <= 62)
        )
    ].copy()

    if len(box_entries) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    pitch.arrows(
        box_entries["x"],
        box_entries["y"],
        box_entries["end_x"],
        box_entries["end_y"],
        ax=ax,
        width=2,
        headwidth=4,
        alpha=0.65
    )

    ax.set_title(f"{team} | Box Entries", fontsize=15, weight="bold")

    return fig


# ============================================================
# DUEL ZONES
# ============================================================

def plot_duel_zones(events, match_id, team):

    match_df = events[events["match_id"] == match_id].copy()

    if len(match_df) == 0 or "tournament_id" not in match_df.columns:
        return None

    return plot_zone_heatmap(
        match_df,
        match_df["tournament_id"].iloc[0],
        team,
        ["Duel"],
        f"{team} | Duel Zones",
        cmap="Purples"
    )


# ============================================================
# TOP MATCH PLAYERS BAR
# ============================================================

def plot_top_match_players_bar(player_match, metric, title):

    if len(player_match) == 0 or metric not in player_match.columns:
        return None

    df = player_match.sort_values(metric, ascending=False).head(8).copy()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(df["player"][::-1], df[metric][::-1])

    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(metric)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# BOX ENTRIES BY TYPE
# ============================================================

def plot_box_entry_zones_by_type(events, tournament_id, team, team_color="#2563EB"):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"].isin(["Pass", "Carry"]))
    ].copy()

    df = ensure_xy(df)

    if "pass_end_location" in df.columns:
        pass_end = df["pass_end_location"].apply(parse_location)
        df["pass_end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["pass_end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    if "carry_end_location" in df.columns:
        carry_end = df["carry_end_location"].apply(parse_location)
        df["carry_end_x"] = carry_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        df["carry_end_y"] = carry_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

    df["end_x"] = np.nan
    df["end_y"] = np.nan

    if "pass_end_x" in df.columns:
        df["end_x"] = df["end_x"].fillna(df["pass_end_x"])
        df["end_y"] = df["end_y"].fillna(df["pass_end_y"])

    if "carry_end_x" in df.columns:
        df["end_x"] = df["end_x"].fillna(df["carry_end_x"])
        df["end_y"] = df["end_y"].fillna(df["carry_end_y"])

    df = df.dropna(subset=["x", "y", "end_x", "end_y"])

    box_entries = df[
        (df["end_x"] >= 102) &
        (df["end_y"] >= 18) &
        (df["end_y"] <= 62) &
        ~(
            (df["x"] >= 102) &
            (df["y"] >= 18) &
            (df["y"] <= 62)
        )
    ].copy()

    if len(box_entries) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    passes = box_entries[box_entries["type"] == "Pass"]
    carries = box_entries[box_entries["type"] == "Carry"]

    if len(passes) > 0:
        pitch.arrows(
            passes["x"], passes["y"], passes["end_x"], passes["end_y"],
            ax=ax, width=2, headwidth=4, color=team_color, alpha=0.65, label="Pass into box"
        )

    if len(carries) > 0:
        pitch.arrows(
            carries["x"], carries["y"], carries["end_x"], carries["end_y"],
            ax=ax, width=2, headwidth=4, color="#111827", alpha=0.65, label="Carry into box"
        )

    ax.set_title(f"{team} | Box Entries by Type", fontsize=15, weight="bold")
    ax.legend(loc="upper left")

    return fig


# ============================================================
# CROSS MAP
# ============================================================

def plot_cross_map(events, tournament_id, team, team_color="#2563EB"):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Pass")
    ].copy()

    if "pass_cross" not in df.columns or "pass_end_location" not in df.columns:
        return None

    df = df[df["pass_cross"].fillna(False) == True].copy()

    df = ensure_xy(df)

    pass_end = df["pass_end_location"].apply(parse_location)
    df["end_x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["end_y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

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
        alpha=0.65
    )

    ax.set_title(f"{team} | Cross Map", fontsize=15, weight="bold")

    return fig


# ============================================================
# SET PIECE SHOT ZONES
# ============================================================

def plot_set_piece_shot_zones(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team)
    ].copy()

    if "play_pattern" not in df.columns:
        return None

    set_piece_events = df[
        df["play_pattern"].isin([
            "From Corner",
            "From Free Kick",
            "From Throw In",
            "From Goal Kick"
        ])
    ].copy()

    if len(set_piece_events) == 0:
        return None

    return plot_zone_heatmap(
        set_piece_events,
        tournament_id,
        team,
        ["Shot"],
        f"{team} | Set Piece Shot Zones",
        cmap="Reds"
    )


# ============================================================
# GOAL TIMING CHART
# ============================================================

def plot_goal_timing(events, tournament_id, team):

    shots = events[
        (events["tournament_id"] == tournament_id) &
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

    shots["period"] = pd.cut(
        shots["minute"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    plot_df = (
        shots
        .groupby(["period", "goal_type"], observed=False)
        .size()
        .reset_index(name="goals")
    )

    pivot = plot_df.pivot(
        index="period",
        columns="goal_type",
        values="goals"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 4))

    pivot.plot(kind="bar", ax=ax)

    ax.set_title(f"{team} | Goal Timing", fontsize=15, weight="bold")
    ax.set_xlabel("Match Period")
    ax.set_ylabel("Goals")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# GOALS CONCEDED ZONES
# ============================================================

def plot_goals_conceded_zones(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] != team) &
        (events["type"] == "Shot") &
        (events["shot_outcome"] == "Goal")
    ].copy()

    if len(df) == 0:
        return None

    df = ensure_xy(df)
    df = df[df["x"].notna() & df["y"].notna()]

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(
        df["x"],
        df["y"],
        statistic="count",
        bins=(6, 4)
    )

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Reds",
        alpha=0.78
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(f"{team} | Goals Conceded Zones", fontsize=15, weight="bold")

    return fig


# ============================================================
# CROSS TARGET ZONES
# ============================================================

def plot_cross_target_zones(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Pass")
    ].copy()

    if "pass_cross" not in df.columns or "pass_end_location" not in df.columns:
        return None

    df = df[df["pass_cross"].fillna(False) == True].copy()

    if len(df) == 0:
        return None

    pass_end = df["pass_end_location"].apply(parse_location)

    df["x"] = pass_end.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
    df["y"] = pass_end.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)

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

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Oranges",
        alpha=0.78
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(f"{team} | Cross Target Zones", fontsize=15, weight="bold")

    return fig


# ============================================================
# ATTACK TYPE PROFILE
# ============================================================

def plot_attack_type_profile(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team)
    ].copy()

    if "play_pattern" not in df.columns:
        return None

    attack_types = (
        df["play_pattern"]
        .value_counts()
        .head(8)
        .reset_index()
    )

    attack_types.columns = ["play_pattern", "events"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        attack_types["play_pattern"][::-1],
        attack_types["events"][::-1]
    )

    ax.set_title(f"{team} | Attack / Play Pattern Profile", fontsize=15, weight="bold")
    ax.set_xlabel("Events")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig
    # ============================================================
# NEW GRAPH — LONG PASS ORIGIN ZONES
# ============================================================

def plot_long_pass_origin_zones(events, tournament_id, team, min_length=30):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Pass")
    ].copy()

    if "pass_length" not in df.columns:
        return None

    df = df[df["pass_length"] >= min_length]

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

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Blues",
        alpha=0.8
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Long Pass Origin Zones",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW GRAPH — LONG PASS TARGET ZONES
# ============================================================

def plot_long_pass_target_zones(events, tournament_id, team, min_length=30):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Pass")
    ].copy()

    if "pass_length" not in df.columns:
        return None

    if "pass_end_location" not in df.columns:
        return None

    df = df[df["pass_length"] >= min_length]

    pass_end = df["pass_end_location"].apply(parse_location)

    df["x"] = pass_end.apply(
        lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan
    )

    df["y"] = pass_end.apply(
        lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan
    )

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

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Blues",
        alpha=0.8
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Long Pass Target Zones",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW GRAPH — PROGRESSIVE PASS MAP
# ============================================================

def plot_progressive_pass_zones(events, tournament_id, team, progression_threshold=15):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Pass")
    ].copy()

    if "pass_end_location" not in df.columns:
        return None

    df = ensure_xy(df)

    pass_end = df["pass_end_location"].apply(parse_location)

    df["end_x"] = pass_end.apply(
        lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan
    )

    df["end_y"] = pass_end.apply(
        lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan
    )

    df = df.dropna(subset=["x", "end_x"])

    df["progression"] = df["end_x"] - df["x"]

    df = df[df["progression"] >= progression_threshold]

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
        color="#2563EB",
        alpha=0.45
    )

    ax.set_title(
        f"{team} | Progressive Passes",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW GRAPH — PROGRESSIVE CARRIES
# ============================================================

def plot_progressive_carry_zones(events, tournament_id, team, progression_threshold=10):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Carry")
    ].copy()

    if "carry_end_location" not in df.columns:
        return None

    df = ensure_xy(df)

    carry_end = df["carry_end_location"].apply(parse_location)

    df["end_x"] = carry_end.apply(
        lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan
    )

    df["end_y"] = carry_end.apply(
        lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan
    )

    df = df.dropna(subset=["x", "end_x"])

    df["progression"] = df["end_x"] - df["x"]

    df = df[df["progression"] >= progression_threshold]

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
        color="#7C3AED",
        alpha=0.45
    )

    ax.set_title(
        f"{team} | Progressive Carries",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW GRAPH — CONCEDED SHOT ZONES
# ============================================================

def plot_conceded_shot_zones(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] != team) &
        (events["type"] == "Shot")
    ].copy()

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

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Reds",
        alpha=0.8
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Conceded Shot Zones",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW GRAPH — AERIAL DUEL ZONES
# ============================================================

def plot_aerial_duel_zones(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Duel")
    ].copy()

    if "duel_type" not in df.columns:
        return None

    df = df[
        df["duel_type"].astype(str).str.contains("Aerial", na=False)
    ]

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

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Greens",
        alpha=0.8
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Aerial Duel Zones",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW GRAPH — PASSES BEFORE SHOT
# ============================================================

def plot_passes_before_shot(events, tournament_id, team, team_color="#2563EB", max_previous_events=20):

    df = events[events["tournament_id"] == tournament_id].copy()

    if "index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "event_index"})
        idx_col = "event_index"
    else:
        idx_col = "index"

    sequences = []

    for match_id, match_df in df.groupby("match_id"):

        match_df = match_df.sort_values(idx_col)

        shots = match_df[
            (match_df["team"] == team) &
            (match_df["type"] == "Shot")
        ]

        for _, shot in shots.iterrows():

            previous = match_df[
                (match_df[idx_col] < shot[idx_col]) &
                (match_df[idx_col] >= shot[idx_col] - max_previous_events)
            ]

            passes = len(
                previous[
                    (previous["team"] == team) &
                    (previous["type"] == "Pass")
                ]
            )

            sequences.append(passes)

    if len(sequences) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = range(0, max(sequences) + 2)

    ax.hist(
        sequences,
        bins=bins,
        color=team_color,
        alpha=0.85
    )

    ax.set_title(f"{team} | Passes Before Shot", fontsize=15, weight="bold")
    ax.set_xlabel("Passes before shot")
    ax.set_ylabel("Shots")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# NEW GRAPH — RECOVERY TO SHOT SPEED
# ============================================================
def plot_transition_shot_speed(events, tournament_id, team, team_color="#2563EB", max_seconds=20):

    df = events[events["tournament_id"] == tournament_id].copy()

    if "minute" not in df.columns or "second" not in df.columns:
        return None

    if "index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "event_index"})
        idx_col = "event_index"
    else:
        idx_col = "index"

    df["event_seconds"] = df["minute"].fillna(0) * 60 + df["second"].fillna(0)

    transition_times = []

    for match_id, match_df in df.groupby("match_id"):

        match_df = match_df.sort_values(idx_col)

        recoveries = match_df[
            (match_df["team"] == team) &
            (match_df["type"].isin(["Ball Recovery", "Interception"]))
        ]

        shots = match_df[
            (match_df["team"] == team) &
            (match_df["type"] == "Shot")
        ]

        for _, recovery in recoveries.iterrows():

            next_shots = shots[shots[idx_col] > recovery[idx_col]]

            if len(next_shots) == 0:
                continue

            next_shot = next_shots.iloc[0]

            seconds_diff = next_shot["event_seconds"] - recovery["event_seconds"]

            if 0 <= seconds_diff <= max_seconds:
                transition_times.append(seconds_diff)

    if len(transition_times) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = range(0, max_seconds + 2, 2)

    ax.hist(
        transition_times,
        bins=bins,
        color=team_color,
        alpha=0.85
    )

    ax.set_title(f"{team} | Recovery to Shot Speed", fontsize=15, weight="bold")
    ax.set_xlabel("Seconds from recovery to shot")
    ax.set_ylabel("Sequences")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig

# ============================================================
# UPDATED — GOAL TIMING WITH TEAM COLOR
# ============================================================

def plot_goal_timing(events, tournament_id, team, team_color="#2563EB", against_color="#DC2626"):

    shots = events[
        (events["tournament_id"] == tournament_id) &
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

    shots["period"] = pd.cut(
        shots["minute"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    pivot = (
        shots
        .groupby(["period", "goal_type"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(labels, fill_value=0)
    )

    for col in ["Goals For", "Goals Against"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[["Goals For", "Goals Against"]]

    fig, ax = plt.subplots(figsize=(9, 4))

    pivot.plot(
        kind="bar",
        ax=ax,
        color=[team_color, against_color]
    )

    ax.set_title(f"{team} | Goal Timing", fontsize=15, weight="bold")
    ax.set_xlabel("Match Period")
    ax.set_ylabel("Goals")
    ax.legend(loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# UPDATED — SIMILARITY BAR WITH TEAM COLOR
# ============================================================

def plot_team_similarity_bar(team_similarity, tournament_id, team, team_color="#2563EB"):

    df = team_similarity[
        (team_similarity["tournament_id"] == tournament_id) &
        (team_similarity["team"] == team)
    ].copy()

    if len(df) == 0:
        return None

    df = df.sort_values("similarity_rank").head(8)

    labels = df["similar_team"].astype(str)

    if "similar_tournament_id" in df.columns:
        labels = labels + " | " + df["similar_tournament_id"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        labels[::-1],
        df["similarity_score"][::-1],
        color=team_color,
        alpha=0.85
    )

    ax.set_title(f"Most Similar Teams to {team}", fontsize=15, weight="bold")
    ax.set_xlabel("Similarity Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# NEW — SET PIECE PROFILE
# ============================================================

def plot_set_piece_profile(events, tournament_id, team):

    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team)
    ].copy()

    if "play_pattern" not in df.columns:
        return None

    df = df[
        df["play_pattern"].isin([
            "From Corner",
            "From Free Kick",
            "From Throw In",
            "From Goal Kick"
        ])
    ].copy()

    if len(df) == 0:
        return None

    profile = (
        df["play_pattern"]
        .value_counts()
        .reset_index()
    )

    profile.columns = ["set_piece_type", "events"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        profile["set_piece_type"][::-1],
        profile["events"][::-1],
        color="#F97316",
        alpha=0.85
    )

    ax.set_title(f"{team} | Set Piece Profile", fontsize=15, weight="bold")
    ax.set_xlabel("Events")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ============================================================
# NEW — CROSS ORIGIN ZONES
# ============================================================

def plot_cross_origin_zones(events, tournament_id, team):
    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Pass")
    ].copy()

    if "pass_cross" not in df.columns:
        return None

    df = df[df["pass_cross"].fillna(False) == True].copy()
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

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Blues",
        alpha=0.78
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Cross Origin Zones",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW — OFFENSIVE DUEL ZONES
# ============================================================

def plot_offensive_duel_zones(events, tournament_id, team):
    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Duel")
    ].copy()

    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])
    df = df[df["x"] >= 60].copy()

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(
        df["x"],
        df["y"],
        statistic="count",
        bins=(6, 4)
    )

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Oranges",
        alpha=0.78
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Offensive Duel Zones",
        fontsize=15,
        weight="bold"
    )

    return fig


# ============================================================
# NEW — DEFENSIVE DUEL ZONES
# ============================================================

def plot_defensive_duel_zones(events, tournament_id, team):
    df = events[
        (events["tournament_id"] == tournament_id) &
        (events["team"] == team) &
        (events["type"] == "Duel")
    ].copy()

    df = ensure_xy(df)
    df = df.dropna(subset=["x", "y"])
    df = df[df["x"] < 60].copy()

    if len(df) == 0:
        return None

    pitch, fig, ax = draw_pitch()

    bin_statistic = pitch.bin_statistic(
        df["x"],
        df["y"],
        statistic="count",
        bins=(6, 4)
    )

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="Purples",
        alpha=0.78
    )

    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        fontsize=10,
        color="#111827"
    )

    ax.set_title(
        f"{team} | Defensive Duel Zones",
        fontsize=15,
        weight="bold"
    )

    return fig




# ============================================================
# TEAM MATCH — MATCH XI COMPARISON
# ============================================================

def plot_match_xi_comparison(match_events, match_id, home_team, away_team, home_color="#2563EB", away_color="#DC2626"):

    if "type" not in match_events.columns:
        return None

    xi_events = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["type"] == "Starting XI")
    ].copy()

    if len(xi_events) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    teams = [home_team, away_team]
    colors = [home_color, away_color]
    x_positions = [0.25, 0.75]

    for team_name, color, x in zip(teams, colors, x_positions):

        team_xi = xi_events[xi_events["team"] == team_name]
        players = []

        if len(team_xi) > 0 and "tactics_lineup" in team_xi.columns:
            lineup = team_xi.iloc[0]["tactics_lineup"]

            try:
                parsed = ast.literal_eval(str(lineup))
            except:
                parsed = None

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        player_data = item.get("player", {})
                        position_data = item.get("position", {})

                        player_name = player_data.get("name", "N/A") if isinstance(player_data, dict) else "N/A"
                        position_name = position_data.get("name", "") if isinstance(position_data, dict) else ""

                        players.append(f"{clean_player_name(player_name)} · {position_name}")

        ax.text(
            x,
            0.95,
            team_name,
            ha="center",
            va="top",
            fontsize=15,
            weight="bold",
            color=color
        )

        if len(players) == 0:
            ax.text(
                x,
                0.50,
                "No Starting XI available",
                ha="center",
                va="center",
                fontsize=11
            )
        else:
            for i, p in enumerate(players[:11]):
                ax.text(
                    x,
                    0.86 - i * 0.07,
                    p,
                    ha="center",
                    va="center",
                    fontsize=10
                )

    ax.set_title("Starting XI Comparison", fontsize=16, weight="bold")

    return fig


# ============================================================
# TEAM MATCH — PASSING NETWORK
# ============================================================

def plot_match_passing_network(match_events, match_id, team, team_color="#2563EB", min_passes=2):

    df = match_events[
        (match_events["match_id"] == match_id) &
        (match_events["team"] == team) &
        (match_events["type"] == "Pass")
    ].copy()

    if len(df) == 0:
        return None

    required_cols = ["player", "pass_recipient", "location", "pass_end_location"]

    if not all(col in df.columns for col in required_cols):
        return None

    df = ensure_xy(df)

    pass_end = df["pass_end_location"].apply(parse_location)

    df["end_x"] = pass_end.apply(
        lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan
    )

    df["end_y"] = pass_end.apply(
        lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan
    )

    df = df.dropna(
        subset=["player", "pass_recipient", "x", "y", "end_x", "end_y"]
    )

    if len(df) == 0:
        return None

    avg_positions = (
        df.groupby("player")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            passes=("type", "count")
        )
        .reset_index()
    )

    pass_links = (
        df.groupby(["player", "pass_recipient"])
        .size()
        .reset_index(name="passes")
    )

    pass_links = pass_links[pass_links["passes"] >= min_passes]

    if len(pass_links) == 0:
        return None

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
            linewidth=max(1, row["passes"] * 0.45),
            color=team_color,
            alpha=0.35,
            zorder=2
        )

    pitch.scatter(
        avg_positions["x"],
        avg_positions["y"],
        s=avg_positions["passes"] * 18 + 120,
        ax=ax,
        color=team_color,
        edgecolors="#111827",
        linewidth=1.2,
        alpha=0.9,
        zorder=3
    )

    for _, row in avg_positions.iterrows():
        ax.text(
            row["x"],
            row["y"],
            clean_player_name(row["player"]),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            zorder=4
        )

    ax.set_title(f"{team} | Passing Network", fontsize=15, weight="bold")

    return fig


# ============================================================
# TEAM MATCH — KEY PLAYERS BAR
# ============================================================

def plot_match_key_players(player_match, team, team_color="#2563EB"):

    if len(player_match) == 0:
        return None

    df = player_match.copy()

    numeric_cols = [
        "shots",
        "passes",
        "carries",
        "recoveries",
        "pressures",
        "duels",
        "xg"
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    df["impact_score"] = (
        df["shots"] * 3 +
        df["xg"] * 8 +
        df["passes"] * 0.2 +
        df["carries"] * 0.3 +
        df["recoveries"] * 1.5 +
        df["pressures"] * 0.8 +
        df["duels"] * 0.8
    )

    df = df.sort_values("impact_score", ascending=False).head(8)

    if len(df) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        df["player"][::-1],
        df["impact_score"][::-1],
        color=team_color,
        alpha=0.85
    )

    ax.set_title(f"{team} | Key Players Impact", fontsize=15, weight="bold")
    ax.set_xlabel("Impact Score")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig
