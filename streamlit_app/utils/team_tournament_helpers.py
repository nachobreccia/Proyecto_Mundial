import ast
import pandas as pd
import streamlit as st

from utils.cards import metric_card


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


def parse_loc(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except:
            return None
    return None


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


def plot_goal_timing_safe(plot_goal_timing, events, tournament_id, team, primary_color):
    try:
        return plot_goal_timing(events, tournament_id, team, team_color=primary_color)
    except TypeError:
        return plot_goal_timing(events, tournament_id, team)


def plot_similarity_bar_safe(plot_team_similarity_bar, team_similarity, tournament_id, team, primary_color):
    try:
        return plot_team_similarity_bar(team_similarity, tournament_id, team, team_color=primary_color)
    except TypeError:
        return plot_team_similarity_bar(team_similarity, tournament_id, team)