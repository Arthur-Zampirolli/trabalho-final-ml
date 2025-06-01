import os
import kagglehub
import pandas as pd

# Download latest version
def load_dataset() :
  path = kagglehub.dataset_download("davidcariboo/player-scores")

  print("Path to dataset files:", path)

  files = os.listdir(path)
  print("Files in dataset:", files)

  appearances_df = pd.read_csv(os.path.join(path, "appearances.csv"), parse_dates=["date"])
  club_games_df = pd.read_csv(os.path.join(path, "club_games.csv"))
  clubs_df = pd.read_csv(os.path.join(path, "clubs.csv"))
  competitions_df = pd.read_csv(os.path.join(path, "competitions.csv"))
  game_events_df = pd.read_csv(os.path.join(path, "game_events.csv"), parse_dates=["date"])
  game_lineups_df = pd.read_csv(os.path.join(path, "game_lineups.csv"), parse_dates=["date"])
  games_df = pd.read_csv(os.path.join(path, "games.csv"), parse_dates=["date"])
  player_valuations_df = pd.read_csv(os.path.join(path, "player_valuations.csv"), parse_dates=["date"])
  players_df = pd.read_csv(os.path.join(path, "players.csv"), parse_dates=["date_of_birth"])
  transfers_df = pd.read_csv(os.path.join(path, "transfers.csv"), parse_dates=["transfer_date"])

  return {
    "appearances": appearances_df,
    "club_games": club_games_df,
    "clubs": clubs_df,
    "competitions": competitions_df,
    "game_events": game_events_df,
    "game_lineups": game_lineups_df,
    "games": games_df,
    "player_valuations": player_valuations_df,
    "players": players_df,
    "transfers": transfers_df,
  }