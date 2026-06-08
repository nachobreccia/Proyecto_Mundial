from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "streamlit_app" / "data"

EVENTS_COLUMNS = [
    "match_id", "tournament_id", "team", "player", "type", "event_type",
    "minute", "second", "period", "location", "shot_statsbomb_xg",
    "shot_outcome", "pass_outcome", "pass_end_location", "carry_end_location",
    "pass_cross", "pass_recipient", "duel_type", "duel_outcome"
]


@st.cache_data
def load_csv(file_name):
    for path in [BASE_DIR / file_name, DATA_DIR / file_name]:
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_events():
    full_events = load_csv("events.csv")

    if not full_events.empty:
        if "event_type" not in full_events.columns and "type" in full_events.columns:
            full_events["event_type"] = full_events["type"]
        return full_events

    event_parts = sorted(DATA_DIR.glob("events_part_*.csv"))

    if len(event_parts) == 0:
        return pd.DataFrame(columns=EVENTS_COLUMNS)

    df = pd.concat(
        [pd.read_csv(path) for path in event_parts],
        ignore_index=True
    )

    if "event_type" not in df.columns and "type" in df.columns:
        df["event_type"] = df["type"]

    return df


@st.cache_data
def load_all_data():
    return {
        "matches": load_csv("matches.csv"),
        "events": load_events(),
        "assets": load_csv("assets.csv"),
        "team_master": load_csv("team_master.csv"),
        "team_match_stats": load_csv("team_match_stats.csv"),
        "team_similarity": load_csv("team_similarity.csv"),
        "player_master": load_csv("player_master.csv"),
        "player_match_stats": load_csv("player_match_stats.csv"),
        "player_match_stats_with_positions": load_csv("player_match_stats_with_positions.csv"),
        "player_role_fit": load_csv("player_role_fit.csv"),
        "player_similarity": load_csv("player_similarity.csv"),
        "positions": load_csv("positions.csv"),
        "highlighted_matches": load_csv("highlighted_matches.csv"),
        "most_frequent_xi": load_csv("most_frequent_xi.csv"),
    }