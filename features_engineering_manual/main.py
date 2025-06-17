import os
import click
import warnings

import pandas as pd

from load_dataset import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")

def validate_and_add_feature(df, column_name, calculation_function):
    if column_name not in df.columns:
        print(f"Adding missing feature: {column_name}")
        df[column_name] = calculation_function(df)
    else:
        print(f"Feature {column_name} already exists. Skipping.")

def run_feature_engineering(output_dir, current_file=None):
    # Load dataset
    dataset = load_dataset()

    # Extract DataFrames
    transfers_df = dataset['transfers'].copy()
    players_df = dataset['players'].copy()
    appearances_df = dataset['appearances'].copy()
    player_valuations_df = dataset['player_valuations'].copy()

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
        lambda df: df.merge(
            players_df[['player_id', 'country_of_birth']].rename(
                columns={'country_of_birth': 'player_country_of_birth'}
            ),
            on='player_id',
            how='left'
        )['player_country_of_birth']
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_position',
        lambda df: df.merge(
            players_df[['player_id', 'position']].rename(
                columns={'position': 'player_position'}
            ),
            on='player_id',
            how='left'
        )['player_position']
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_sub_position',
        lambda df: df.merge(
            players_df[['player_id', 'sub_position']].rename(
                columns={'sub_position': 'player_sub_position'}
            ),
            on='player_id',
            how='left'
        )['player_sub_position']
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_foot',
        lambda df: df.merge(
            players_df[['player_id', 'foot']].rename(
                columns={'foot': 'player_foot'}
            ),
            on='player_id',
            how='left'
        )['player_foot']
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_height_cm',
        lambda df: df.merge(
            players_df[['player_id', 'height_in_cm']].rename(
                columns={'height_in_cm': 'player_height_cm'}
            ),
            on='player_id',
            how='left'
        )['player_height_cm']
    )
    validate_and_add_feature(
        feature_engineered_transfers,
        'player_date_of_birth',
        lambda df: df.merge(
            players_df[['player_id', 'date_of_birth']].rename(
                columns={'date_of_birth': 'player_date_of_birth'}
            ),
            on='player_id',
            how='left'
        )['player_date_of_birth']
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
        )
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
        )
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
        )
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
        )
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
    ## Number of appearances
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