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


@st.cache_data(show_spinner="Loading tournament events...")
def load_events(tournament_id=None):
    if tournament_id is None:
        return pd.DataFrame()

    safe_name = str(tournament_id).replace("/", "_").replace(" ", "_")
    tournament_path = DATA_DIR / f"events_{safe_name}.parquet"

    if tournament_path.exists():
        events = pd.read_parquet(tournament_path)

        if "event_type" not in events.columns and "type" in events.columns:
            events["event_type"] = events["type"]

        return events

    full_path = DATA_DIR / "events.parquet"

    if full_path.exists():
        events = pd.read_parquet(full_path)

        if "event_type" not in events.columns and "type" in events.columns:
            events["event_type"] = events["type"]

        if "tournament_id" in events.columns:
            events = events[events["tournament_id"] == tournament_id].copy()

        return events

    return pd.DataFrame()


@st.cache_data(show_spinner="Loading base data...")
def load_all_data():
    return {
        "matches": load_csv("matches.csv"),
        "events": pd.DataFrame(),
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