import warnings
import click

import pandas as pd
import featuretools as ft
import woodwork.logical_types as ww

from featuretools.selection import (
    remove_low_information_features,
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features
)

from load_dataset import load_dataset

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`."
)

def run_feature_engineering(output_dir):
  dataset = load_dataset()

  # Extract DataFrames
  transfers_df = dataset['transfers'].copy()
  players_df = dataset['players'].copy()
  clubs_df = dataset['clubs'].copy()
  competitions_df = dataset['competitions'].copy()
  appearances_df = dataset['appearances'].copy()
  games_df = dataset['games'].copy()
  game_events_df = dataset['game_events'].copy()
  player_valuations_df = dataset['player_valuations'].copy()

  # Convert all date columns to datetime
  transfers_df['transfer_date'] = pd.to_datetime(transfers_df['transfer_date'])
  players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'])
  players_df['contract_expiration_date'] = pd.to_datetime(players_df['contract_expiration_date'])
  appearances_df['date'] = pd.to_datetime(appearances_df['date'])
  games_df['date'] = pd.to_datetime(games_df['date'])
  game_events_df['date'] = pd.to_datetime(game_events_df['date'])
  player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'])

  # Create EntitySet
  es = ft.EntitySet(id='transfer_fee_prediction')

  # Add entities to EntitySet with corrected logical types
  es = es.add_dataframe(
    dataframe_name='transfers',
    dataframe=transfers_df,
    make_index=True,
    index='transfer_id',
    time_index='transfer_date',
    logical_types={
        'player_id': ww.Categorical,
        'from_club_id': ww.Categorical,
        'to_club_id': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='players',
    dataframe=players_df,
    index='player_id',
    logical_types={
        'player_id': ww.Categorical,
        'position': ww.Categorical,
        'sub_position': ww.Categorical,
        'foot': ww.Categorical,
        'current_club_id': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='clubs',
    dataframe=clubs_df,
    index='club_id',
    logical_types={
        'club_id': ww.Categorical,
        'domestic_competition_id': ww.Categorical,
        'coach_name': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='competitions',
    dataframe=competitions_df,
    index='competition_id',
    logical_types={
        'competition_id': ww.Categorical,
        'type': ww.Categorical,
        'sub_type': ww.Categorical,
        'confederation': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='appearances',
    dataframe=appearances_df,
    index='appearance_id',
    time_index='date',
    logical_types={
        'player_id': ww.Categorical,
        'competition_id': ww.Categorical,
        'player_club_id': ww.Categorical,
        'game_id': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='games',
    dataframe=games_df,
    index='game_id',
    time_index='date',
    logical_types={
        'game_id': ww.Categorical,
        'competition_id': ww.Categorical,
        'home_club_id': ww.Categorical,
        'away_club_id': ww.Categorical,
        'competition_type': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='game_events',
    dataframe=game_events_df,
    index='game_event_id',
    time_index='date',
    logical_types={
        'game_id': ww.Categorical,
        'club_id': ww.Categorical,
        'player_id': ww.Categorical,
        'player_in_id': ww.Categorical,
        'player_assist_id': ww.Categorical,
        'type': ww.Categorical
    }
  )

  es = es.add_dataframe(
    dataframe_name='player_valuations',
    dataframe=player_valuations_df,
    make_index=True,
    index='valuation_id',
    time_index='date',
    logical_types={
        'player_id': ww.Categorical,
        'current_club_id': ww.Categorical
    }
  )

  relationships = [
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'transfers', 'child_column_name': 'from_club_id'},
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'transfers', 'child_column_name': 'to_club_id'},
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'players', 'child_column_name': 'current_club_id'},
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'appearances', 'child_column_name': 'player_club_id'},
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'games', 'child_column_name': 'home_club_id'},
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'games', 'child_column_name': 'away_club_id'},
    {'parent_dataframe_name': 'clubs', 'parent_column_name': 'club_id', 'child_dataframe_name': 'game_events', 'child_column_name': 'club_id'},
    {'parent_dataframe_name': 'competitions', 'parent_column_name': 'competition_id', 'child_dataframe_name': 'clubs', 'child_column_name': 'domestic_competition_id'},
    {'parent_dataframe_name': 'competitions', 'parent_column_name': 'competition_id', 'child_dataframe_name': 'appearances', 'child_column_name': 'competition_id'},
    {'parent_dataframe_name': 'competitions', 'parent_column_name': 'competition_id', 'child_dataframe_name': 'games', 'child_column_name': 'competition_id'},
    {'parent_dataframe_name': 'games', 'parent_column_name': 'game_id', 'child_dataframe_name': 'appearances', 'child_column_name': 'game_id'},
    {'parent_dataframe_name': 'games', 'parent_column_name': 'game_id', 'child_dataframe_name': 'game_events', 'child_column_name': 'game_id'},
    {'parent_dataframe_name': 'players', 'parent_column_name': 'player_id', 'child_dataframe_name': 'transfers', 'child_column_name': 'player_id'},
    {'parent_dataframe_name': 'players', 'parent_column_name': 'player_id', 'child_dataframe_name': 'appearances', 'child_column_name': 'player_id'},
    {'parent_dataframe_name': 'players', 'parent_column_name': 'player_id', 'child_dataframe_name': 'player_valuations', 'child_column_name': 'player_id'},
    {'parent_dataframe_name': 'players', 'parent_column_name': 'player_id', 'child_dataframe_name': 'game_events', 'child_column_name': 'player_id'}
  ]

  # Prepare cutoff time dataframe
  cutoff_df = transfers_df[['transfer_id', 'transfer_date']].copy()
  cutoff_df.rename(columns={'transfer_id': 'instance_id', 'transfer_date': 'time'}, inplace=True)

  # Add all relationships to the entityset
  for rel in relationships:
    try:
      es = es.add_relationship(**rel)
    except Exception as e:
      print(f"Erro ao adicionar relacionamento: {rel} -> {e}")

  # Compute last_time_indexes
  es.add_last_time_indexes([
    'transfers',
    'games',
    'game_events',
    'appearances',
    'player_valuations',
  ])

  # Generate features using Deep Feature Synthesis
  feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='transfers',
    cutoff_time=cutoff_df,
    cutoff_time_in_index=True,
    agg_primitives=[
        'sum', 'mean', 'max', 'min', 'count', 'num_unique', 
    ],
    trans_primitives=['year', 'month', 'day', 'weekday'],
    max_depth=2,
    features_only=False,
    verbose=True,
    training_window=ft.Timedelta(365*10, 'days')  # 10 years historical data
  )

  # Reset index to access transfer_id and transfer_date
  feature_matrix = feature_matrix.reset_index()
  feature_matrix.rename(columns={'level_0': 'transfer_id'}, inplace=True)

  # Feature selection/cleaning
  feature_matrix = remove_low_information_features(feature_matrix)
  feature_matrix = remove_highly_correlated_features(feature_matrix)
  feature_matrix = remove_highly_null_features(feature_matrix)
  feature_matrix = remove_single_value_features(feature_matrix)

  # Save results
  feature_matrix.to_csv(f'{output_dir}/transfer_features.csv', index=False)

  print("Feature engineering completed!")
  print(f"Generated {len(feature_defs)} features")
  print(f"Feature matrix shape: {feature_matrix.shape}")

@click.command()
@click.option('--output-dir', default='.', help='Directory to save output CSV files')
def main(output_dir):
  run_feature_engineering(output_dir)

if __name__ == '__main__':
    main()