@st.cache_data(show_spinner="Loading tournament events...")
def load_events(tournament_id=None):
    if tournament_id is None:
        return pd.DataFrame()

    # 1) Buscar parquet por tournament_id
    safe_name = str(tournament_id).replace("/", "_").replace(" ", "_")
    path = DATA_DIR / f"events_{safe_name}.parquet"

    if path.exists():
        events = pd.read_parquet(path)

        if "event_type" not in events.columns and "type" in events.columns:
            events["event_type"] = events["type"]

        return events

    # 2) Fallback: si existe events.parquet completo
    full_path = DATA_DIR / "events.parquet"

    if full_path.exists():
        events = pd.read_parquet(full_path)

        if "event_type" not in events.columns and "type" in events.columns:
            events["event_type"] = events["type"]

        if "tournament_id" in events.columns:
            events = events[events["tournament_id"] == tournament_id].copy()

        return events

    return pd.DataFrame()