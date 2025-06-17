import click
import warnings

import pandas as pd
import featuretools as ft
import woodwork.logical_types as ww
from featuretools.primitives import Count, Equal

from load_dataset import load_dataset
from custom_primitives.ageAt import AgeAt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")

def run_feature_engineering(output_dir):
    # Load dataset
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
    transfers_df['transfer_date'] = pd.to_datetime(transfers_df['transfer_date'], format='%Y-%m-%d')
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], format='%Y-%m-%d')
    players_df['contract_expiration_date'] = pd.to_datetime(players_df['contract_expiration_date'], format='%Y-%m-%d')
    appearances_df['date'] = pd.to_datetime(appearances_df['date'], format='%Y-%m-%d')
    games_df['date'] = pd.to_datetime(games_df['date'], format='%Y-%m-%d')
    game_events_df['date'] = pd.to_datetime(game_events_df['date'], format='%Y-%m-%d')
    player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'], format='%Y-%m-%d')

    es = ft.EntitySet(id='transfer_fee_prediction')
    
    # Add entities to EntitySet with corrected logical types
    es.add_dataframe(
        dataframe_name='transfers',
        dataframe=transfers_df,
        make_index=True,
        index='transfer_id',
        time_index='transfer_date',
        logical_types={
            'player_id': ww.Categorical,
            'from_club_id': ww.Categorical,
            'to_club_id': ww.Categorical,
            'transfer_date': ww.Datetime,
            'transfer_season': ww.Categorical,
        }
    )

    es.add_dataframe(
        dataframe_name='players',
        dataframe=players_df,
        index='player_id',
        logical_types={
            'player_id': ww.Categorical,
            'position': ww.Categorical,
            'sub_position': ww.Categorical,
            'foot': ww.Categorical,
            'current_club_id': ww.Categorical,
            'date_of_birth': ww.Datetime,
            'contract_expiration_date': ww.Datetime
        }
    )

    es.add_dataframe(
        dataframe_name='clubs',
        dataframe=clubs_df,
        index='club_id',
        logical_types={
            'club_id': ww.Categorical,
            'domestic_competition_id': ww.Categorical,
            'coach_name': ww.Categorical
        }
    )

    es.add_dataframe(
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

    es.add_dataframe(
        dataframe_name='appearances',
        dataframe=appearances_df,
        index='appearance_id',
        time_index='date',
        logical_types={
            'player_id': ww.Categorical,
            'competition_id': ww.Categorical,
            'player_club_id': ww.Categorical,
            'game_id': ww.Categorical,
            'date': ww.Datetime,
            'season': ww.Categorical
        }
    )

    es.add_dataframe(
        dataframe_name='games',
        dataframe=games_df,
        index='game_id',
        time_index='date',
        logical_types={
            'game_id': ww.Categorical,
            'competition_id': ww.Categorical,
            'home_club_id': ww.Categorical,
            'away_club_id': ww.Categorical,
            'competition_type': ww.Categorical,
            'date': ww.Datetime,
            'season': ww.Categorical
        }
    )

    es.add_dataframe(
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
            'type': ww.Categorical,
            'date': ww.Datetime
        }
    )

    es.add_dataframe(
        dataframe_name='player_valuations',
        dataframe=player_valuations_df,
        make_index=True,
        index='valuation_id',
        time_index='date',
        logical_types={
            'player_id': ww.Categorical,
            'current_club_id': ww.Categorical,
            'date': ww.Datetime
        }
    )

    # Add relationships (directly without loops)
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
        {'parent_dataframe_name': 'players', 'parent_column_name': 'player_id', 'child_dataframe_name': 'game_events', 'child_column_name': 'player_id'},
    ]

    for rel in relationships:
        try:
            es.add_relationship(**rel)
        except Exception as e:
            print(f"Error adding relationship: {rel} -> {e}")

    # Prepare cutoff time
    cutoff_df = transfers_df[['transfer_id', 'transfer_date']].copy()
    cutoff_df.rename(columns={'transfer_id': 'instance_id', 'transfer_date': 'time'}, inplace=True)

    es.add_last_time_indexes([
        'transfers',
        'games',
        'game_events',
        'appearances',
        'player_valuations',
    ])

    features = [
        # Transfers
        ft.Feature(es['transfers'].ww['transfer_fee']),
        ft.Feature(es['transfers'].ww['transfer_date']),
        ft.Feature(es['transfers'].ww['transfer_season']),
        ft.Feature(es['transfers'].ww['from_club_id']),
        ft.Feature(es['transfers'].ww['to_club_id']),

        # Players
        ft.DirectFeature(es['players'].ww['player_id'], 'transfers'),
        ft.DirectFeature(es['players'].ww['country_of_birth'], 'transfers'),
        ft.DirectFeature(es['players'].ww['position'], 'transfers'),
        ft.DirectFeature(es['players'].ww['sub_position'], 'transfers'),
        ft.DirectFeature(es['players'].ww['foot'], 'transfers'),
        ft.DirectFeature(es['players'].ww['height_in_cm'], 'transfers'),
        ft.DirectFeature(es['players'].ww['date_of_birth'], 'transfers'),
        ft.Feature([
            ft.Feature(es['players'].ww['date_of_birth'], 'transfers'),
            ft.Feature(es['transfers'].ww['transfer_date'])
        ], primitive=AgeAt()).rename('players.age_at_transfer'),

        ## Player + Appearances + Games
        # Total appearances count for the player up to the transfer
        ft.DirectFeature(
            ft.Feature(
                es['appearances'].ww['appearance_id'],
                parent_dataframe_name='players',
                primitive=Count()
            ),
            'transfers'
        ).rename('players.total_appearances_count'),

        # ft.DirectFeature(
        #     ft.Feature(
        #         es['appearances'].ww['appearance_id'],
        #         parent_dataframe_name='players',
        #         primitive=Count()
        #         where=ft.Feature(
        #             es['appearances'].ww['date'],
        #             parent_dataframe_name='transfers',
        #             primitive=Equal()
        #         )
        #     ),
        #     'transfers'
        # ).rename('players.total_appearances_count'),

    ]

    # Generate features using DFS
    print("Generating features...")
    feature_matrix = ft.calculate_feature_matrix(
        features=features,
        entityset=es,
        cutoff_time=cutoff_df,
        verbose=True,
        n_jobs=1,
        chunk_size=1000
    )
    print("Feature generation completed.")
    
    # Process feature matrix
    feature_matrix = feature_matrix.reset_index()
    feature_matrix.rename(columns={'level_0': 'transfer_id'}, inplace=True)
    
    # # Feature selection
    # feature_matrix, feature_defs = ft.selection.remove_low_information_features(feature_matrix, features=feature_defs)
    # feature_matrix, feature_defs = ft.selection.remove_highly_correlated_features(feature_matrix, features=feature_defs)
    # feature_matrix, feature_defs = ft.selection.remove_highly_null_features(feature_matrix, features=feature_defs)
    # feature_matrix, feature_defs = ft.selection.remove_single_value_features(feature_matrix, features=feature_defs)
    
    # Save results
    # ft.save_features(feature_defs, f'{output_dir}/transfer_features.json')
    feature_matrix.to_csv(f'{output_dir}/transfer_features.csv', index=False)
    
    # print("Feature engineering completed!")
    # print(f"Generated {len(feature_defs)} features")
    # print(f"Feature matrix shape: {feature_matrix.shape}")

@click.command()
@click.option('--output-dir', default='.', help='Directory to save output files')
def main(output_dir):
    run_feature_engineering(output_dir)

if __name__ == '__main__':
    main()