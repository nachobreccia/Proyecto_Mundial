from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "streamlit_app" / "data"

@st.cache_data
def load_csv(file_name):
    for path in [BASE_DIR / file_name, DATA_DIR / file_name]:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_events():
    full = load_csv("events.csv")
    if len(full) > 0:
        return full

    parts = sorted(DATA_DIR.glob("events_*.csv"))
    if not parts:
        parts = sorted(DATA_DIR.glob("events_part_*.csv"))

    if not parts:
        return pd.DataFrame()

    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

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