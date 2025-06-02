import os
import kagglehub
import pandas as pd

def to_category(series, categories, missing_value_category="Missing"):
  """
  Convert a pandas Series to a categorical type with specified categories.
  Missing values and values not in categories are replaced with a specified missing value category.
  """
  series = series.fillna(missing_value_category)
  series = series.apply(lambda x: x if x in categories else missing_value_category)
  series = series.astype(pd.CategoricalDtype(categories=categories, ordered=True))
  return series

def load_transfers(dataset_path):
  transfers_df = pd.read_csv(
    os.path.join(dataset_path, "transfers.csv"),
    parse_dates=["transfer_date"],
    usecols=[
      "player_id",
      "from_club_id",
      "to_club_id",
      "transfer_date",
      "transfer_season",
      "transfer_fee",
      "market_value_in_eur",
      # "from_club_name", # Not used in the analysis, but can be useful for display
      # "to_club_name", # Not used in the analysis, but can be useful for display
      # "player_name" # Not used in the analysis, but can be useful for display
    ],
    dtype={
      # Identifiers
      "player_id": int,
      # Foreign keys
      "from_club_id": int,
      "to_club_id": int,
      # Other fields
      "transfer_season": str,
      "from_club_name": str,
      "to_club_name": str,
      "transfer_fee": float,
      "market_value_in_eur": float,
      "player_name": str
    }
  )

  # Filter only transfers with transfer_fee and market_value_in_eur not null and greater than 0
  transfers_df = transfers_df[
    transfers_df["transfer_fee"].notnull() & 
    (transfers_df["transfer_fee"] > 0) &
    transfers_df["market_value_in_eur"].notnull() &
    (transfers_df["market_value_in_eur"] > 0)
  ]

  return transfers_df

def load_players(dataset_path, transfers_df=None):
  players_df = pd.read_csv(
    os.path.join(dataset_path, "players.csv"),
    parse_dates=["date_of_birth", "contract_expiration_date"],
    usecols=[
      "player_id",
      "last_season",
      "current_club_id",
      "country_of_birth",
      "city_of_birth",
      "country_of_citizenship",
      "date_of_birth",
      "sub_position",
      "position",
      "foot",
      "height_in_cm",
      "contract_expiration_date",
      "agent_name",
      "current_club_domestic_competition_id",
      "current_club_name",
      "market_value_in_eur",
      "highest_market_value_in_eur"
    ],
    dtype={
      "player_id": int,
      "current_club_id": int,
      "current_club_domestic_competition_id": str,
      "name": str,
      "first_name": str,
      "last_name": str,
      "last_season": str,
      "player_code": str,
      "country_of_birth": str,
      "city_of_birth": str,
      "country_of_citizenship": str,
      "sub_position": str,
      "position": str,
      "foot": str,
      "height_in_cm": 'Int64',
      "agent_name": str,
      "image_url": str,
      "url": str,
      "current_club_name": str,
      "market_value_in_eur": float,
      "highest_market_value_in_eur": float
    }
  )

  # Fill missing values on categorical columns with "Missing"
  players_df["city_of_birth"] = players_df["city_of_birth"].fillna("Missing")
  players_df["country_of_birth"] = players_df["country_of_birth"].fillna("Missing")
  players_df["country_of_citizenship"] = players_df["country_of_citizenship"].fillna("Missing")
  players_df["agent_name"] = players_df["agent_name"].fillna("Missing")

  # Fill heigh based on position median height
  position_median_height = players_df.groupby('position')['height_in_cm'].transform('median')
  players_df['height_in_cm'] = players_df['height_in_cm'].fillna(position_median_height)
  
  # Fill contract expiration date with a median by club and position, otherwise with the median of the whole position
  contract_expiration_median_per_club = players_df.groupby(['current_club_id', 'position'])['contract_expiration_date'].transform('median')
  contract_expiration_median = players_df.groupby(['position'])['contract_expiration_date'].transform('median')
  players_df['contract_expiration_date'] = players_df['contract_expiration_date'].fillna(contract_expiration_median_per_club).fillna(contract_expiration_median)

  # Drop null players birth date (only one in needed dataset)
  players_df = players_df[players_df["date_of_birth"].notnull()]

  # Convert position to categorical type
  players_df["position"] = to_category(players_df["position"], missing_value_category="Missing", categories=["Goalkeeper", "Defender", "Midfield", "Attack", "Missing"])
  # Convert sub_position to categorical type
  players_df["sub_position"] = to_category(players_df["sub_position"], missing_value_category="Missing", categories=["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",  "Left Midfield", "Right Midfield",  "Attacking Midfield", "Left Winger", "Right Winger", "Second Striker", "Centre-Forward", "Missing"])
  # Convert foot to categorical type, transforming values to "Left", "Right", "Both" and "Missing"
  players_df["foot"] = players_df["foot"].replace({"left": "Left", "right": "Right", "both": "Both"})
  players_df["foot"] = to_category(players_df["foot"], missing_value_category="Missing", categories=["Left", "Right", "Both", "Missing"])

  if transfers_df is not None:
    transferred_player_ids = transfers_df["player_id"].unique()
    players_df = players_df[players_df["player_id"].isin(transferred_player_ids)]

  return players_df

def load_appearances(dataset_path, players_df=None):
  appearances_df = pd.read_csv(
    os.path.join(dataset_path, "appearances.csv"),
    parse_dates=["date"],
    usecols=[
      "appearance_id",
      "game_id",
      "player_id",
      "player_club_id",
      "player_current_club_id",
      "competition_id",
      "date",
      "yellow_cards",
      "red_cards",
      "goals",
      "assists",
      "minutes_played"
    ],
    dtype={
      # Identifiers
      "appearance_id": str,
      # Foreign keys
      "game_id": int,
      "player_id": int,
      "player_club_id": int,
      "player_current_club_id": int,
      "competition_id": str,
      # Other fields
      "yellow_cards": int,
      "red_cards": int,
      "goals": int,
      "assists": int,
      "minutes_played": int
    }
  )

  if players_df is not None:
    current_player_ids = players_df["player_id"].unique()
    appearances_df = appearances_df[appearances_df["player_id"].isin(current_player_ids)]

  return appearances_df

def load_clubs(dataset_path):
  clubs_df = pd.read_csv(
    os.path.join(dataset_path, "clubs.csv"),
    usecols=[
      "club_id",
      "domestic_competition_id",
      "squad_size",
      "average_age",
      "foreigners_number",
      "foreigners_percentage",
      "national_team_players",
      "stadium_seats",
      "net_transfer_record",
      "coach_name",
      "last_season",
      # "club_code", # Not used in the analysis, we already have the club_id
      # "name", # Not used in the analysis, but can be useful for display
      # "stadium_name", # Not used in the analysis, not relevat for the problem
      # "total_market_value", # Not used in the analysis, all values are null
      # "filename", # Not used in the analysis, but can be useful for display
      # "url" # Not used in the analysis, but can be useful for display
    ],
    dtype={
      # Identifiers
      "club_id": int,
      # Foreign keys
      "domestic_competition_id": str,
      # Other fields
      "club_code": str,
      "name": str,
      "total_market_value": float,
      "squad_size": int,
      "average_age": float,
      "foreigners_number": int,
      "foreigners_percentage": float,
      "national_team_players": int,
      "stadium_name": str,
      "stadium_seats": int,
      "net_transfer_record": str,
      "coach_name": str,
      "last_season": str,
      "filename": str,
      "url": str
    }
  )

  # Fill missing values in categorical columns with "Missing"
  clubs_df["coach_name"] = clubs_df["coach_name"].fillna("Missing")
  # Fill average age with median by competition
  competition_median_age = clubs_df.groupby('domestic_competition_id')['average_age'].transform('median')
  clubs_df['average_age'] = clubs_df['average_age'].fillna(competition_median_age)

  # Fill foreigners percentage with squad size and foreigners number, otherwise set as 0
  clubs_df['foreigners_percentage'] = clubs_df['foreigners_percentage'].fillna(
    clubs_df['foreigners_number'] / clubs_df['squad_size'] * 100
  ).fillna(0)

  return clubs_df

def load_competitions(dataset_path):
  competitions_df = pd.read_csv(
    os.path.join(dataset_path, "competitions.csv"),
    usecols=[
      "competition_id",
      "sub_type",
      "type",
      "country_id",
      "domestic_league_code",
      "confederation",
      "is_major_national_league"
      # "url", # Not used in the analysis, but can be useful for display
      # "country_name", # Not used in the analysis, but can be useful for display
      # "name", # Not used in the analysis, but can be useful for display
      # "competition_code", # Not used in the analysis, we already have the competition_id
    ],
    dtype={
      # Identifiers
      "competition_id": str,
      # Foreign keys
      "country_id": str,
      # Other fields
      "competition_code": str,
      "name": str,
      "sub_type": str,
      "type": str,
      "country_name": str,
      "domestic_league_code": str,
      "confederation": str,
      "url": str,
      "is_major_national_league": bool
    }
  )

  # Fill missing values in categorical columns with "Missing"
  competitions_df["domestic_league_code"] = competitions_df["domestic_league_code"].fillna("Missing")

  # Convert sub_type to categorical type
  competitions_df["sub_type"] = to_category(
    competitions_df["sub_type"],
    missing_value_category="Missing",
    categories=[
      "domestic_cup", "domestic_super_cup", "uefa_super_cup", "first_tier",
      "europa_league", "uefa_europa_conference_league", "europa_league_qualifying",
      "league_cup", "uefa_europa_conference_league_qualifiers",
      "uefa_champions_league", "fifa_club_world_cup", "uefa_champions_league_qualifying",
      "Missing"
    ]
  )
  # Convert type to categorical type
  competitions_df["type"] = to_category(
    competitions_df["type"],
    missing_value_category="Missing",
    categories=["domestic_cup", "other", "international_cup", "domestic_league", "Missing"]
  )

  return competitions_df

def load_game_events(dataset_path, players_df=None):
  game_events_df = pd.read_csv(
    os.path.join(dataset_path, "game_events.csv"),
    parse_dates=["date"],
    usecols=[
      "game_event_id",
      "game_id",
      "date",
      "minute",
      "type",
      "club_id",
      "player_id",
      "player_in_id",
      "player_assist_id"
      # "description", # Not used in the analysis, but can be useful for display
    ],
    dtype={
      # Identifiers
      "game_event_id": str,
      # Foreign keys
      "game_id": int,
      "club_id": int,
      "player_id": int,
      "player_in_id": 'Int64',  # Using 'Int64' to allow NaN values
      "player_assist_id": 'Int64',  # Using 'Int64' to allow NaN values
      # Other fields
      "minute": int,
      "type": str,
      "description": str,
    }
  )

  if players_df is not None:
    current_player_ids = players_df["player_id"].unique()
    game_events_df = game_events_df[
      game_events_df["player_id"].isin(current_player_ids) |
      game_events_df["player_in_id"].isin(current_player_ids) |
      game_events_df["player_assist_id"].isin(current_player_ids)
    ]

  # Fill missing values in 'player_in_id' and 'player_assist_id' with -1
  game_events_df["player_in_id"] = game_events_df["player_in_id"].fillna(-1).astype(int)
  game_events_df["player_assist_id"] = game_events_df["player_assist_id"].fillna(-1).astype(int)

  return game_events_df

def load_games(dataset_path):
  games_df = pd.read_csv(
    os.path.join(dataset_path, "games.csv"),
    parse_dates=["date"],
    usecols=[
      "game_id",
      "competition_id",
      "season",
      "date",
      "home_club_id",
      "away_club_id",
      "home_club_goals",
      "away_club_goals",
      "home_club_position",
      "away_club_position",
      "home_club_manager_name",
      "away_club_manager_name",
      "attendance",
      "competition_type"
      # "aggregate", # Not used in the analysis, not relevant for the problem
      # "round", # Not used in the analysis, not relevant for the problem
      # "stadium", # Not used in the analysis, not relevant for the problem
      # "referee", # Not used in the analysis, not relevant for the problem
      # "url", # Not used in the analysis, but can be useful for display
      # "home_club_formation", # Not used in the analysis, not relevant for the problem
      # "away_club_formation", # Not used in the analysis, not relevant for the problem
      # "home_club_name", # Not used in the analysis, but can be useful for display
      # "away_club_name", # Not used in the analysis, but can be useful for display
    ],
    dtype={
      "game_id": int,
      "competition_id": str,
      "season": int,
      "round": str,
      "home_club_id": 'Int64',
      "away_club_id": 'Int64',
      "home_club_goals": 'Int64',
      "away_club_goals": 'Int64',
      "home_club_position": 'Int64',
      "away_club_position": 'Int64',
      "home_club_manager_name": str,
      "away_club_manager_name": str,
      "stadium": str,
      "attendance": 'Int64',
      "referee": str,
      "url": str,
      "home_club_formation": str,
      "away_club_formation": str,
      "home_club_name": str,
      "away_club_name": str,
      "aggregate": str,
      "competition_type": str
    }
  )

  # Fill categorical columns with "Missing"
  games_df["home_club_manager_name"] = games_df["home_club_manager_name"].fillna("Missing")
  games_df["away_club_manager_name"] = games_df["away_club_manager_name"].fillna("Missing")
  games_df["home_club_position"] = games_df["home_club_position"].fillna(-1)
  games_df["away_club_position"] = games_df["away_club_position"].fillna(-1)

  # Fill missing attendance values with the median attendance per home_club_id, otherwise with the overall median
  attendance_median_per_club = games_df.groupby("home_club_id")["attendance"].transform("median").astype('Int64')
  games_df["attendance"] = games_df["attendance"].fillna(attendance_median_per_club).fillna(-1)

  # Convert competition_type to categorical type
  games_df["competition_type"] = to_category(
    games_df["competition_type"],
    missing_value_category="Missing",
    categories=["domestic_league", "domestic_cup", "international_cup", "Missing"]
  )

  # Filter out games with null home_club_id, away_club_id, home_club_goals or away_club_goals
  games_df = games_df[
    games_df["home_club_id"].notnull() & 
    games_df["away_club_id"].notnull() &
    games_df["home_club_goals"].notnull() &
    games_df["away_club_goals"].notnull()
  ]

  return games_df

def load_player_valuations(dataset_path, players_df=None):
  player_valuations_df = pd.read_csv(
    os.path.join(dataset_path, "player_valuations.csv"),
    parse_dates=["date"],
    usecols=[
      "player_id",
      "date",
      "market_value_in_eur",
      "current_club_id",
      "player_club_domestic_competition_id"
    ],
    dtype={
      "player_id": int,
      "market_value_in_eur": float,
      "current_club_id": int,
      "player_club_domestic_competition_id": str
    }
  )

  if players_df is not None:
    current_player_ids = players_df["player_id"].unique()
    player_valuations_df = player_valuations_df[player_valuations_df["player_id"].isin(current_player_ids)]

  return player_valuations_df

def load_dataset() :
  path = kagglehub.dataset_download("davidcariboo/player-scores")

  print("Path to dataset files:", path)

  files = os.listdir(path)
  print("Files in dataset:", files)

  transfers_df = load_transfers(path) # Target table for the job (transfer_fee column)

  clubs_df = load_clubs(path)
  competitions_df = load_competitions(path)
  games_df = load_games(path)

  players_df = load_players(path, transfers_df=transfers_df)
  appearances_df = load_appearances(path, players_df=players_df)
  game_events_df = load_game_events(path, players_df=players_df)
  player_valuations_df = load_player_valuations(path, players_df=players_df)

  return {
    "transfers": transfers_df,
    "appearances": appearances_df,
    "players": players_df,
    "clubs": clubs_df,
    "competitions": competitions_df,
    "game_events": game_events_df,
    "games": games_df,
    "player_valuations": player_valuations_df,
  }