import streamlit as st

from utils.data_loader import load_all_data
from utils.style import apply_global_style


st.set_page_config(
    page_title="Home",
    layout="wide"
)

apply_global_style()

data = load_all_data()

matches = data["matches"]
events = data["events"]
team_master = data["team_master"]
player_master = data["player_master"]
team_similarity = data["team_similarity"]
player_similarity = data["player_similarity"]
highlighted_matches = data["highlighted_matches"]
most_frequent_xi = data["most_frequent_xi"]


st.markdown(
    """
    <h1 style='margin-bottom:0;'>Football Intelligence Platform</h1>
    <div class='page-subtitle'>
        Tactical analysis, scouting and performance intelligence powered by StatsBomb Open Data.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    This project transforms open event data into an interactive football analytics platform.
    It combines team analysis, player profiling, match analysis, similarity models, role fit logic,
    highlighted performances and tactical visualizations across selected international tournaments.
    """
)

st.divider()


# ============================================================
# MAIN KPIs
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Competitions", matches["competition_name"].nunique())

with c2:
    st.metric("Tournaments", matches["tournament_id"].nunique())

with c3:
    st.metric("Matches", matches["match_id"].nunique())

with c4:
    st.metric("Teams", team_master["team"].nunique())

with c5:
    st.metric("Players", player_master["player"].nunique())

c6, c7, c8, c9 = st.columns(4)

with c6:
    st.metric("Events", f"{len(events):,}")

with c7:
    st.metric("Team Profiles", f"{len(team_master):,}")

with c8:
    st.metric("Player Profiles", f"{len(player_master):,}")

with c9:
    st.metric("Highlighted Matches", f"{len(highlighted_matches):,}")

st.divider()


# ============================================================
# ABOUT
# ============================================================

st.subheader("Project Overview")

text_col, module_col = st.columns([1.25, 1])

with text_col:
    st.markdown(
        """
        The platform was built to explore international football through event data.
        It is designed for tactical analysis, player scouting, match review and profile comparison.

        The objective is to make StatsBomb Open Data easier to understand from a football
        decision-making perspective: how teams play, which players stand out, what roles they fit,
        and which profiles are similar across competitions.
        """
    )

with module_col:
    st.info(
        """
        Main analytical layers:

        • Team DNA and tactical identity  
        • Player importance and role profiles  
        • Match impact scores  
        • Similar teams and players  
        • Most frequent XI  
        • Strengths and weaknesses  
        """
    )

st.divider()


# ============================================================
# ANALYSIS MODULES
# ============================================================

st.subheader("Analysis Modules")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        """
        #### Team Analysis

        **Team Tournament**  
        Full tactical profile across a tournament.

        **Team Match**  
        Match-specific team analysis.

        **Team Comparison**  
        Compare teams across different tournaments.
        """
    )

with m2:
    st.markdown(
        """
        #### Player Analysis

        **Player Tournament**  
        Player profile, strengths, weaknesses and role fit.

        **Player Match**  
        Match-level action maps and performance indicators.

        **Player Comparison**  
        Compare players across teams and competitions.
        """
    )

with m3:
    st.markdown(
        """
        #### Intelligence Layer

        **Similarity Models**  
        Find comparable players and teams.

        **Role Fit**  
        Evaluate player suitability for tactical roles.

        **Highlighted Matches**  
        Identify strongest team and player performances.
        """
    )

st.divider()


# ============================================================
# MODEL OUTPUTS
# ============================================================

st.subheader("Model Outputs")

o1, o2, o3 = st.columns(3)

with o1:
    st.metric("Team Similarity Rows", f"{len(team_similarity):,}")

with o2:
    st.metric("Player Similarity Rows", f"{len(player_similarity):,}")

with o3:
    st.metric("Most Frequent XI Rows", f"{len(most_frequent_xi):,}")

st.divider()


# ============================================================
# AVAILABLE COMPETITIONS
# ============================================================

st.subheader("Available Competitions")

tournaments = (
    matches[
        [
            "tournament_id",
            "competition_name",
            "season_name"
        ]
    ]
    .drop_duplicates()
    .sort_values(["competition_name", "season_name"])
)

st.dataframe(
    tournaments,
    use_container_width=True,
    hide_index=True
)

st.divider()


# ============================================================
# COMPETITION BREAKDOWN
# ============================================================

st.subheader("Competition Breakdown")

comp_summary = (
    matches
    .groupby("competition_name")
    .agg(
        tournaments=("tournament_id", "nunique"),
        matches=("match_id", "nunique")
    )
    .reset_index()
    .sort_values("matches", ascending=False)
)

st.dataframe(
    comp_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()


# ============================================================
# METHODOLOGY
# ============================================================

st.subheader("Methodology")

method_col1, method_col2 = st.columns(2)

with method_col1:
    st.markdown(
        """
        The project uses StatsBomb event data to create indicators around:

        • Attacking volume  
        • Build-up and progression  
        • Defensive activity  
        • Pressing and recoveries  
        • Shot generation  
        • Passing and carrying behavior  
        • Set-piece activity  
        """
    )

with method_col2:
    st.markdown(
        """
        The derived models include:

        • Team DNA scores  
        • Player importance scores  
        • Position-aware goalkeeper scoring  
        • Role classification  
        • Role fit ranking  
        • Similarity models  
        • Match impact scores  
        """
    )

st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Developed by Juan Ignacio Breccia | Football Analytics · Scouting · Data Science | Powered by StatsBomb Open Data"
)