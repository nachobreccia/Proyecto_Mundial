import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

@st.cache_data
def load_csv(file_name):
    return pd.read_csv(BASE_DIR / file_name)

@st.cache_data
def load_all_data():
    data = {
        "matches": load_csv("matches.csv"),
        "events": load_csv("events.csv"),
        "team_match_stats": load_csv("team_match_stats.csv"),
        "player_match_stats": load_csv("player_match_stats.csv"),
        "player_match_stats_with_positions": load_csv("player_match_stats_with_positions.csv"),
        "positions": load_csv("positions.csv"),
        "team_master": load_csv("team_master.csv"),
        "player_master": load_csv("player_master.csv"),
        "team_similarity": load_csv("team_similarity.csv"),
        "player_similarity": load_csv("player_similarity.csv"),
        "highlighted_matches": load_csv("highlighted_matches.csv"),
        "most_frequent_xi": load_csv("most_frequent_xi.csv"),
        "player_role_fit": load_csv("player_role_fit.csv"),
        "assets": load_csv("assets.csv")
    }

    return data