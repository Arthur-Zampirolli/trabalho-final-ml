import os
import re
import click
import warnings

import pandas as pd

from load_dataset import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")


def run_feature_engineering(output_dir, current_file=None):
    def validate_and_add_feature(df, column_name, calculation_function):
        if column_name not in df.columns:
            print(f"Adding missing feature: {column_name}")
            df[column_name] = calculation_function(df)
            # Save intermediate result after adding each feature
            sv_path = os.path.join(output_dir, "feature_engineered_transfers_sv.csv")
            df.to_csv(sv_path, index=False)
        else:
            print(f"Feature {column_name} already exists. Skipping.")

    # Load dataset
    dataset = load_dataset()

    # Extract DataFrames
    transfers_df = dataset['transfers'].copy()
    players_df = dataset['players'].copy()
    appearances_df = dataset['appearances'].copy()
    player_valuations_df = dataset['player_valuations'].copy()
    clubs_df = dataset['clubs'].copy()

    # Convert all date columns to datetime
    transfers_df['transfer_date'] = pd.to_datetime(transfers_df['transfer_date'], format='%Y-%m-%d')
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], format='%Y-%m-%d')
    players_df['contract_expiration_date'] = pd.to_datetime(players_df['contract_expiration_date'], format='%Y-%m-%d')
    appearances_df['date'] = pd.to_datetime(appearances_df['date'], format='%Y-%m-%d')
    player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'], format='%Y-%m-%d')

    # If current_file is provided, load it
    if current_file and os.path.exists(current_file):
        print(f"Loading existing feature-engineered data from {current_file}.")
        feature_engineered_transfers = pd.read_csv(current_file)
        print("Current columns in the loaded feature-engineered data:")
        print(feature_engineered_transfers.columns.tolist())
    else:
        print(f"Creating new dataframe.")
        feature_engineered_transfers = pd.DataFrame(transfers_df.drop(columns=['market_value_in_eur'], errors='ignore'))

     # Player direct features
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_country_of_birth',
        lambda df: df['player_id'].map(
            players_df.set_index('player_id')['country_of_birth']
        )
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_position',
        lambda df: df['player_id'].map(
            players_df.set_index('player_id')['position']
        )
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_sub_position',
        lambda df: df['player_id'].map(
            players_df.set_index('player_id')['sub_position']
        )
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_foot',
        lambda df: df['player_id'].map(
            players_df.set_index('player_id')['foot']
        )
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_height_cm',
        lambda df: df['player_id'].map(
            players_df.set_index('player_id')['height_in_cm']
        )
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_date_of_birth',
        lambda df: df['player_id'].map(
            players_df.set_index('player_id')['date_of_birth']
        )
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_age_at_transfer',
        lambda df: (
            (
                pd.to_datetime(df['transfer_date']) - 
                pd.to_datetime(df['player_date_of_birth'])
            ).dt.days // 365
        ).fillna(-1).astype(int)
    )

    # Player appearances features
    ## Average of yellow cards (appearances.yellow_cards)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_yellow_cards_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['yellow_cards'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_yellow_cards_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ]['yellow_cards'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_yellow_cards_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ]['yellow_cards'].mean(),
            axis=1
        ).fillna(-1)
    )
    ## Average of red cards (appearances.red_cards)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_red_cards_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['red_cards'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_red_cards_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ]['red_cards'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_red_cards_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ]['red_cards'].mean(),
            axis=1
        ).fillna(-1)
    )
    ## Average of goals (appearances.goals)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_goals_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['goals'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_goals_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ]['goals'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_goals_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ]['goals'].mean(),
            axis=1
        ).fillna(-1)
    )
    ## Average of assists (appearances.assists)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_assists_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['assists'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_assists_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ]['assists'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_assists_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ]['assists'].mean(),
            axis=1
        ).fillna(-1)
    )
    ## Average of minutes_played (appearances.minutes_played)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_minutes_played_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['minutes_played'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_minutes_played_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ]['minutes_played'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_minutes_played_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ]['minutes_played'].mean(),
            axis=1
        ).fillna(-1)
    )
    ## Average of attendance (appearances.attendance)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_attendance_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['attendance'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_attendance_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ]['attendance'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_attendance_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ]['attendance'].mean(),
            axis=1
        ).fillna(-1)
    )
    # Number of appearances
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date']))
            ].shape[0],
            axis=1
        ).fillna(-1)
    )
    # Drop rows where player_appearances_total < 0
    feature_engineered_transfers = feature_engineered_transfers[feature_engineered_transfers['player_appearances_total'] > 0].reset_index(drop=True)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1)
            ].shape[0],
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3)
            ].shape[0],
            axis=1
        ).fillna(-1)
    )
    ## Appearances on top national leagues and cups: LaLiga, Premiere, Ligue 1, Serie A e Bundesliga (appearances.competition_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_top_national_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['competition_id'].isin([
                    'ES1', 'FR1', 'IT1', 'GB1', 'L1', # Leagues
                    'CIT', 'FAC', 'CDR', 'DFB' # Cups
                ]))
            ].shape[0],
            axis=1
        ).fillna(0)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_top_national_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1) &
                (appearances_df['competition_id'].isin([
                    'ES1', 'FR1', 'IT1', 'GB1', 'L1', # Leagues
                    'CIT', 'FAC', 'CDR', 'DFB' # Cups
                ]))
            ].shape[0],
            axis=1
        ).fillna(0)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_top_national_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3) &
                (appearances_df['competition_id'].isin([
                    'ES1', 'FR1', 'IT1', 'GB1', 'L1', # Leagues
                    'CIT', 'FAC', 'CDR', 'DFB' # Cups
                ]))
            ].shape[0],
            axis=1
        ).fillna(0)
    )
    ## Appearances on top continental leagues: Champions and Europa League (appearances.competition_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_top_continental_total',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['competition_id'].isin([
                    'USC', 'EL', 'CL', 'CLQ',
                ]))
            ].shape[0],
            axis=1
        ).fillna(0)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_top_continental_last_season',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] == row['transfer_season'] - 1) &
                (appearances_df['competition_id'].isin([
                    'USC', 'EL', 'CL', 'CLQ',
                ]))
            ].shape[0],
            axis=1
        ).fillna(0)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_appearances_top_continental_last_3_seasons',
        lambda df: df.apply(
            lambda row: appearances_df[
                (appearances_df['player_id'] == row['player_id']) &
                (appearances_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (appearances_df['season'] >= row['transfer_season'] - 3) &
                (appearances_df['competition_id'].isin([
                    'USC', 'EL', 'CL', 'CLQ',
                ]))
            ].shape[0],
            axis=1
        ).fillna(0)
    )

    # Player valuation features
    ## Last valuation before transfer (player_valuations.market_value_in_eur)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_last_valuation',
        lambda df: df.apply(
            lambda row: (
                player_valuations_df[
                    (player_valuations_df['player_id'] == row['player_id']) &
                    (player_valuations_df['date'] < pd.to_datetime(row['transfer_date']))
                ].sort_values(by='date', ascending=False)['market_value_in_eur'].iloc[0]
                if not player_valuations_df[
                    (player_valuations_df['player_id'] == row['player_id']) &
                    (player_valuations_df['date'] < pd.to_datetime(row['transfer_date']))
                ].empty else -1
            ),
            axis=1
        ).fillna(-1)
    )
    ## Highest valuation before transfer (player_valuations.market_value_in_eur)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_highest_valuation',
        lambda df: df.apply(
            lambda row: player_valuations_df[
                (player_valuations_df['player_id'] == row['player_id']) &
                (player_valuations_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['market_value_in_eur'].max(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_highest_valuation_last_year',
        lambda df: df.apply(
            lambda row: player_valuations_df[
                (player_valuations_df['player_id'] == row['player_id']) &
                (player_valuations_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (player_valuations_df['date'] >= pd.to_datetime(row['transfer_date']) - pd.DateOffset(years=1))
            ]['market_value_in_eur'].max(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_highest_valuation_last_3_years',
        lambda df: df.apply(
            lambda row: player_valuations_df[
                (player_valuations_df['player_id'] == row['player_id']) &
                (player_valuations_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (player_valuations_df['date'] >= pd.to_datetime(row['transfer_date']) - pd.DateOffset(years=3))
            ]['market_value_in_eur'].max(),
            axis=1
        ).fillna(-1)
    )
    ## Highest valuation before transfer (player_valuations.market_value_in_eur)
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_valuation',
        lambda df: df.apply(
            lambda row: player_valuations_df[
                (player_valuations_df['player_id'] == row['player_id']) &
                (player_valuations_df['date'] < pd.to_datetime(row['transfer_date']))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_valuation_last_year',
        lambda df: df.apply(
            lambda row: player_valuations_df[
                (player_valuations_df['player_id'] == row['player_id']) &
                (player_valuations_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (player_valuations_df['date'] >= pd.to_datetime(row['transfer_date']) - pd.DateOffset(years=1))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_avg_valuation_last_3_years',
        lambda df: df.apply(
            lambda row: player_valuations_df[
                (player_valuations_df['player_id'] == row['player_id']) &
                (player_valuations_df['date'] < pd.to_datetime(row['transfer_date'])) &
                (player_valuations_df['date'] >= pd.to_datetime(row['transfer_date']) - pd.DateOffset(years=3))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )

    # Join club
    ## Club domestic_competition_id (transfers.from_club_id == clubs.club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_domestic_competition_id',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['from_club_id']
            ]['domestic_competition_id'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_domestic_competition_id',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['to_club_id']
            ]['domestic_competition_id'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    ## Club national_team_players (transfers.from_club_id == clubs.club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_national_team_players',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['from_club_id']
            ]['national_team_players'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_national_team_players',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['to_club_id']
            ]['national_team_players'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    ## Club net_transfer_record (transfers.from_club_id == clubs.club_id)
    def parse_net_transfer_record(value):
        if pd.isna(value):
            return 0.0
        if value in ['+-0', '+-0.0', '+-0']:
            return 0.0
        # Remove spaces
        value = str(value).replace(' ', '')
        # Regex to extract sign, euro symbol, number, and multiplier
        match = re.match(r'([+-]?)€?([+-]?)?([\d.,]+)([mk]?)', value, re.IGNORECASE)
        if not match:
            return 0.0
        sign1, sign2, number, multiplier = match.groups()
        sign = -1 if '-' in (sign1 + sign2) else 1
        number = float(number.replace('.', '').replace(',', '.'))
        if multiplier.lower() == 'm':
            number *= 1_000_000
        elif multiplier.lower() == 'k':
            number *= 1_000
        return sign * number
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_net_transfer_record',
        lambda df: df.apply(
            lambda row: parse_net_transfer_record(
                clubs_df[clubs_df['club_id'] == row['from_club_id']]['net_transfer_record'].values[0]
            ) if not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty else 0.0,
            axis=1
        ).fillna(0.0)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_net_transfer_record',
        lambda df: df.apply(
            lambda row: parse_net_transfer_record(
                clubs_df[clubs_df['club_id'] == row['to_club_id']]['net_transfer_record'].values[0]
            ) if not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty else 0.0,
            axis=1
        ).fillna(0.0)
    )
    ## Club average_age (transfers.from_club_id == clubs.club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_average_age',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['from_club_id']
            ]['average_age'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_average_age',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['to_club_id']
            ]['average_age'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    ## Club stadium_seats (transfers.from_club_id == clubs.club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_stadium_seats',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['from_club_id']
            ]['stadium_seats'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_stadium_seats',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['to_club_id']
            ]['stadium_seats'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    ## Club squad_size (transfers.from_club_id == clubs.club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_squad_size',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['from_club_id']
            ]['squad_size'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_squad_size',
        lambda df: df.apply(
            lambda row: clubs_df[
                clubs_df['club_id'] == row['to_club_id']
            ]['squad_size'].values[0]
            if not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty else -1,
            axis=1
        ).fillna(-1)
    )
    ## Club is top national leagues and cups: LaLiga, Premiere, Ligue 1, Serie A e Bundesliga (transfers.from_club_id == clubs.club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_is_top_national',
        lambda df: df.apply(
            lambda row: (
                1 if (
                    not clubs_df[clubs_df['club_id'] == row['from_club_id']].empty and
                    clubs_df[clubs_df['club_id'] == row['from_club_id']]['domestic_competition_id'].values[0] in [
                        'ES1', 'FR1', 'IT1', 'GB1', 'L1' # Leagues
                    ]
                ) else 0
            ),
            axis=1
        ).fillna(0)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_is_top_national',
        lambda df: df.apply(
            lambda row: (
                1 if (
                    not clubs_df[clubs_df['club_id'] == row['to_club_id']].empty and
                    clubs_df[clubs_df['club_id'] == row['to_club_id']]['domestic_competition_id'].values[0] in [
                        'ES1', 'FR1', 'IT1', 'GB1', 'L1' # Leagues
                    ]
                ) else 0
            ),
            axis=1
        ).fillna(0)
    )

    # Club transfers features
    ## Number of transfers from club to club (transfers.from_club_id == transfers.to_club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_to_club_transfers',
        lambda df: df.apply(
            lambda row: transfers_df[
                (transfers_df['from_club_id'] == row['from_club_id']) &
                (transfers_df['to_club_id'] == row['to_club_id']) &
                (transfers_df['transfer_date'] < pd.to_datetime(row['transfer_date']))
            ].shape[0],
            axis=1
        ).fillna(0)
    )
    ## Average transfer(transfers.from_club_id == transfers.to_club_id)
    ## Average transfer value from club to club (transfers.from_club_id == transfers.to_club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_avg_sell_value',
        lambda df: df.apply(
            lambda row: transfers_df[
                (transfers_df['from_club_id'] == row['from_club_id']) &
                (transfers_df['transfer_date'] < pd.to_datetime(row['transfer_date']))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_avg_buy_value',
        lambda df: df.apply(
            lambda row: transfers_df[
                (transfers_df['to_club_id'] == row['to_club_id']) &
                (transfers_df['transfer_date'] < pd.to_datetime(row['transfer_date']))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )
    ## Average transfer value per player position from club to club (transfers.from_club_id == transfers.to_club_id)
    validate_and_add_feature(
        feature_engineered_transfers,
        'from_club_avg_sell_value_same_position',
        lambda df: df.apply(
            lambda row: transfers_df[
                (transfers_df['from_club_id'] == row['from_club_id']) &
                (transfers_df['transfer_date'] < pd.to_datetime(row['transfer_date'])) &
                (transfers_df['player_id'].isin(
                    players_df[players_df['position'] == row['player_position']]['player_id']
                ))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'to_club_avg_buy_value_same_position',
        lambda df: df.apply(
            lambda row: transfers_df[
                (transfers_df['to_club_id'] == row['to_club_id']) &
                (transfers_df['transfer_date'] < pd.to_datetime(row['transfer_date'])) &
                (transfers_df['player_id'].isin(
                    players_df[players_df['position'] == row['player_position']]['player_id']
                ))
            ]['market_value_in_eur'].mean(),
            axis=1
        ).fillna(-1)
    )

    # Define the output file path
    output_file_path = f"{output_dir}/feature_engineered_transfers.csv"
    # Save the updated DataFrame to the CSV file
    feature_engineered_transfers.to_csv(output_file_path, index=False)
    print(f"Updated feature-engineered data saved to {output_file_path}")
    
@click.command()
@click.option('--output-dir', default='.', help='Directory to save output files')
@click.option('--current-file', default=None, help='Path to an existing feature-engineered file')
def main(output_dir, current_file):
    run_feature_engineering(output_dir, current_file)

if __name__ == '__main__':
    main()