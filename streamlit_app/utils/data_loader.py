from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "streamlit_app" / "data"


@st.cache_data(show_spinner=False)
def load_csv(file_name):
    path = DATA_DIR / file_name

    if path.exists():
        return pd.read_csv(path, low_memory=False)

    return pd.DataFrame()


@st.cache_data(show_spinner="Loading events data...")
def load_events():
    event_parts = sorted(DATA_DIR.glob("events_part_*.csv"))

    if len(event_parts) == 0:
        return pd.DataFrame()

    events = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in event_parts],
        ignore_index=True
    )

    if "event_type" not in events.columns and "type" in events.columns:
        events["event_type"] = events["type"]

    return events


@st.cache_data(show_spinner="Loading base data...")
def load_all_data(include_events=False):
    data = {
        "matches": load_csv("matches.csv"),
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

    if include_events:
        data["events"] = load_events()
    else:
        data["events"] = pd.DataFrame()

    return data