def filter_tournament(df, tournament_id):
    return df[df["tournament_id"] == tournament_id].copy()


def filter_team(df, tournament_id, team):
    return df[
        (df["tournament_id"] == tournament_id) &
        (df["team"] == team)
    ].copy()


def filter_player(df, tournament_id, team, player):
    return df[
        (df["tournament_id"] == tournament_id) &
        (df["team"] == team) &
        (df["player"] == player)
    ].copy()


def get_tournament_options(df):
    return sorted(df["tournament_id"].dropna().unique())


def get_team_options(df, tournament_id):
    return sorted(
        df[df["tournament_id"] == tournament_id]["team"]
        .dropna()
        .unique()
    )


def get_player_options(df, tournament_id, team):
    return sorted(
        df[
            (df["tournament_id"] == tournament_id) &
            (df["team"] == team)
        ]["player"]
        .dropna()
        .unique()
    )


def get_all_team_options(df):
    return sorted(df["team"].dropna().unique())


def get_all_player_options(df):
    return sorted(df["player"].dropna().unique())