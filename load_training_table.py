import pandas as pd

def load_training_table(filepath="results/features_engineering_manual/feature_engineered_transfers.csv", category_encode=False):
    dtype = {
        # Identifiers
        "player_id": int,
        "from_club_id": int,
        "to_club_id": int,
        # Dates/seasons
        "transfer_date": "string",
        "transfer_season": int,
        # Target
        "transfer_fee": float,
        # Player features
        "player_country_of_birth": "category",
        "player_position": "category",
        "player_sub_position": "category",
        "player_foot": "category",
        "player_height_cm": float,
        "player_date_of_birth": "string",
        "player_age_at_transfer": int,
        # Player appearances
        "player_appearances_total": int,
        "player_appearances_last_season": int,
        "player_appearances_last_3_seasons": int,
        "player_avg_yellow_cards_total": float,
        "player_avg_yellow_cards_last_season": float,
        "player_avg_yellow_cards_last_3_seasons": float,
        "player_avg_red_cards_total": float,
        "player_avg_red_cards_last_season": float,
        "player_avg_red_cards_last_3_seasons": float,
        "player_avg_goals_total": float,
        "player_avg_goals_last_season": float,
        "player_avg_goals_last_3_seasons": float,
        "player_avg_assists_total": float,
        "player_avg_assists_last_season": float,
        "player_avg_assists_last_3_seasons": float,
        "player_avg_minutes_played_total": float,
        "player_avg_minutes_played_last_season": float,
        "player_avg_minutes_played_last_3_seasons": float,
        "player_avg_attendance_total": float,
        "player_avg_attendance_last_season": float,
        "player_avg_attendance_last_3_seasons": float,
        "player_appearances_top_national_total": int,
        "player_appearances_top_national_last_season": int,
        "player_appearances_top_national_last_3_seasons": int,
        "player_appearances_top_continental_total": int,
        "player_appearances_top_continental_last_season": int,
        "player_appearances_top_continental_last_3_seasons": int,
        # Player valuations
        "player_last_valuation": float,
        "player_highest_valuation": float,
        "player_highest_valuation_last_year": float,
        "player_highest_valuation_last_3_years": float,
        "player_avg_valuation": float,
        "player_avg_valuation_last_year": float,
        "player_avg_valuation_last_3_years": float,
        # Club features
        "from_club_domestic_competition_id": "category",
        "to_club_domestic_competition_id": "category",
        "from_club_national_team_players": int,
        "to_club_national_team_players": int,
        "from_club_average_age": float,
        "to_club_average_age": float,
        "from_club_stadium_seats": int,
        "to_club_stadium_seats": int,
        "from_club_squad_size": int,
        "to_club_squad_size": int,
        "from_club_net_transfer_record": float,
        "to_club_net_transfer_record": float,
        "from_club_is_top_national": int,
        "to_club_is_top_national": int,
        # Club transfer features
        "from_club_to_club_transfers": int,
        "from_club_avg_sell_value": float,
        "to_club_avg_buy_value": float,
        "from_club_avg_sell_value_same_position": float,
        "to_club_avg_buy_value_same_position": float,
    }
    # Read CSV with explicit dtypes, do NOT parse dates
    df = pd.read_csv(filepath, dtype=dtype)

    # Transform date columns to integer YYYYMMDD
    for col in ["transfer_date", "player_date_of_birth"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y%m%d").astype("Int64")

    # Set categorical columns and encode if requested
    category_columns = [col for col, typ in dtype.items() if typ == "category"]
    # Ensure from_club_domestic_competition_id and to_club_domestic_competition_id share categories
    if "from_club_domestic_competition_id" in df.columns and "to_club_domestic_competition_id" in df.columns:
        all_cats = pd.Categorical(
            pd.concat([
                df["from_club_domestic_competition_id"].astype(str),
                df["to_club_domestic_competition_id"].astype(str)
            ]).unique()
        ).categories
        df["from_club_domestic_competition_id"] = df["from_club_domestic_competition_id"].astype(pd.CategoricalDtype(categories=all_cats))
        df["to_club_domestic_competition_id"] = df["to_club_domestic_competition_id"].astype(pd.CategoricalDtype(categories=all_cats))
        if category_encode:
            df["from_club_domestic_competition_id"] = df["from_club_domestic_competition_id"].cat.codes
            df["to_club_domestic_competition_id"] = df["to_club_domestic_competition_id"].cat.codes
    # All other categorical columns
    for col in category_columns:
        if col in df.columns and col not in ["from_club_domestic_competition_id", "to_club_domestic_competition_id"]:
            df[col] = df[col].astype("category")
            if category_encode:
                df[col] = df[col].cat.codes

    return df
