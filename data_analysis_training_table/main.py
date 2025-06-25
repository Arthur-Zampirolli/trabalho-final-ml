import os
import click

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from load_training_table import load_training_table
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def get_basic_metrics(table, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    transfer_fees = table['transfer_fee'].dropna()
    max_fee = transfer_fees.max()
    min_fee = transfer_fees.min()
    total = len(transfer_fees)
    below_25m = (transfer_fees < 25_000_000).sum() / total * 100 if total else 0
    above_50m = (transfer_fees > 50_000_000).sum() / total * 100 if total else 0
    mode_fee = transfer_fees.mode().iloc[0] if not transfer_fees.mode().empty else None

    # Calculate average transfer fee for highlight leagues vs others
    highlight_leagues = ['ES1', 'FR1', 'IT1', 'GB1', 'L1']
    league_col = 'to_club_domestic_competition_id'
    filtered = table.dropna(subset=['transfer_fee', league_col])
    highlight_mask = filtered[league_col].isin(highlight_leagues)
    avg_highlight = filtered.loc[highlight_mask, 'transfer_fee'].mean()
    avg_others = filtered.loc[~highlight_mask, 'transfer_fee'].mean()
    count_highlight = highlight_mask.sum()
    count_others = (~highlight_mask).sum()
    percent_higher = ((avg_highlight - avg_others) / avg_others * 100) if avg_others else 0

    # Média das transferências apenas para GB1
    gb1_mask = filtered[league_col] == 'GB1'
    avg_gb1 = filtered.loc[gb1_mask, 'transfer_fee'].mean()
    count_gb1 = gb1_mask.sum()

    with open(os.path.join(output_dir, "basic_transfer_metrics.txt"), "w") as f:
        f.write(f"Maior transferência: {max_fee:.2f}\n")
        f.write(f"Menor transferência: {min_fee:.2f}\n")
        f.write(f"Porcentagem de transferências abaixo de 25 milhões: {below_25m:.2f}%\n")
        f.write(f"Porcentagem de transferências acima de 50 milhões: {above_50m:.2f}%\n")
        f.write(f"Moda das transferências: {mode_fee:.2f}\n")
        f.write(f"\n")
        f.write(f"Média das transferências (ligas destaque): {avg_highlight:.2f} ({count_highlight} transferências)\n")
        f.write(f"Média das transferências (outras ligas): {avg_others:.2f} ({count_others} transferências)\n")
        f.write(f"Percentual maior da média das ligas destaque em relação às outras: {percent_higher:.2f}%\n")
        f.write(f"Média das transferências (GB1): {avg_gb1:.2f} ({count_gb1} transferências)\n")

def check_and_save_null_values(table, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    with open(os.path.join(output_dir, "null_values_report.txt"), "w") as file:
        total_entries = len(table)
        file.write(f"Null values (Total entries: {total_entries}):\n")
        null_counts = table.isnull().sum()
        for column, count in null_counts.items():
            file.write(f"  {column}: {count} null values\n")
            file.write("\n")

def get_kendall_correlation(table, output_dir, target_column):
    # Columns to ignore in correlation
    ignore_columns = [
        "player_last_valuation",
        "player_highest_valuation",
        "player_highest_valuation_last_year",
        "player_highest_valuation_last_3_years",
        "player_avg_valuation",
        "player_avg_valuation_last_year",
        "player_avg_valuation_last_3_years"
    ]

    # Compute Kendall correlation of all features with the target column
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    features = [
        col for col in numeric_cols
        if col != target_column and col not in ignore_columns
    ]
    correlations = []
    for feature in features:
        corr = table[feature].corr(table[target_column], method='kendall')
        correlations.append((feature, corr))

    # Sort by absolute correlation value and select top 10
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top_correlations = correlations[:10]
    feature_names, corr_values = zip(*top_correlations) if top_correlations else ([], [])

    # Save all correlations to txt
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"kendall_correlations_with_{target_column}.txt"), "w") as f:
        f.write(f"Correlação de Kendall com {target_column}:\n")
        for feature, corr in correlations:
            f.write(f"{feature}: {corr:.4f}\n")

    plt.figure(figsize=(10, 0.6 * len(feature_names) + 2))  # Aumenta a altura para mais espaço
    ax = sns.barplot(x=corr_values, y=feature_names, orient='h')
    ax.set_yticklabels(feature_names, fontsize=10)
    plt.title(f'Top 10 Correlações de Kendall com {target_column}')
    plt.xlabel('Correlação de Kendall')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top10_kendall_correlation_with_{target_column}.png"))
    plt.close()

def get_top10_feature_importances(table, output_dir, target_column):
    # Top 10 features with highest importance (use decision tree)
    X = table.drop(columns=[target_column])
    y = table[target_column]

    # Remove columns with non-numeric data
    X_numeric = X.select_dtypes(include=[np.number])

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_numeric, y)

    importances = clf.feature_importances_
    feature_names = X_numeric.columns
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices], orient='h')
    plt.title('Top 10 Feature Importances (Decision Tree)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top10_feature_importances_decision_tree.png"))
    plt.close()

def plot_boxplots_for_features(table, features, output_dir):
    plt.figure(figsize=(12, 3 * len(features)))
    for i, feature in enumerate(features, 1):
        if feature in table.columns:
            plt.subplot(len(features), 1, i)
            sns.boxplot(x=table[feature])
            plt.title(f'Box Plot of {feature}')
            plt.xlabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots_features.png"))
    plt.close()

def get_baseline(table, target_column, output_dir):
    y_true = table[target_column]
    y_pred = table["player_last_valuation"]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "baseline_metrics.txt"), "w") as f:
        f.write(f"Baseline metrics (player_last_valuation vs {target_column}):\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"  R2 Score: {r2:.4f}\n")

def plot_transfer_fee_pyramid(table, output_dir):
    """
    Plots a pyramid (population pyramid style) of transfer counts by transfer fee value group.
    """
    plt.figure(figsize=(8, 6))
    # Define transfer fee bins (in millions)
    bins = [0, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 50_000_000, 75_000_000, 100_000_000, np.inf]
    labels = [
        "0-5M", "5-10M", "10-20M", "20-30M", "30-50M",
        "50-75M", "75-100M", "100M+"
    ]
    table = table.dropna(subset=['transfer_fee'])
    table['fee_group'] = pd.cut(table['transfer_fee'], bins=bins, labels=labels, right=False)

    counts = table['fee_group'].value_counts().sort_index()
    counts = counts.reindex(labels, fill_value=0)

    plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Número de Transferências')
    plt.title('Contagem de Transferências por Faixa de Valor')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_fee_pyramid.png"))
    plt.close()

def plot_transfer_fee_by_league(table, output_dir):
    plt.figure(figsize=(14, 7))
    # Ensure the league column is string type for consistent filtering
    table['to_club_domestic_competition_id'] = table['to_club_domestic_competition_id'].astype(str)

    # Remove rows with null transfer_fee or league
    filtered = table.dropna(subset=['transfer_fee', 'to_club_domestic_competition_id'])

    # Drop unwanted leagues
    unwanted_leagues = ['UKR1', 'SC1', 'GR1', 'DK1']
    filtered = filtered[~filtered['to_club_domestic_competition_id'].isin(unwanted_leagues)]

    # Highlight leagues of interest
    highlight_leagues = ['ES1', 'FR1', 'IT1', 'GB1', 'L1']
    # Build palette dict from all unique values in the original column
    all_leagues = sorted(filtered['to_club_domestic_competition_id'].dropna().unique())
    palette_dict = {league: ('orange' if league in highlight_leagues else 'lightgray') for league in all_leagues}

    sns.boxplot(
        x='to_club_domestic_competition_id',
        y='transfer_fee',
        data=filtered,
        showfliers=False,
        palette=palette_dict
    )
    plt.title('Variação da Taxa de Transferência por Liga')
    plt.xlabel('Liga (to_club_domestic_competition_id)')
    plt.ylabel('Taxa de Transferência')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_fee_by_league.png"))
    plt.close()

def run_feature_engineering(output_dir, target_column):
    # Load dataset
    table = load_training_table("../results/features_engineering_manual/feature_engineered_transfers.csv", category_encode=False)

    # Get basic metrics
    get_basic_metrics(table, output_dir)

    # Get baseline metrics
    get_baseline(table, target_column, output_dir)

    # Check and save null values
    check_and_save_null_values(table, output_dir)

    # Get Kendall correlation
    get_kendall_correlation(table, output_dir, target_column)

    # Call the function in run_feature_engineering
    get_top10_feature_importances(table, output_dir, target_column)

    # Plot boxplots for specific features
    features_to_plot = [
        'transfer_fee',
    ]
    plot_boxplots_for_features(table, features_to_plot, output_dir)

    # Plot transfer fee distribution
    plot_transfer_fee_pyramid(table, output_dir)

    # Plot transfer fee by league
    plot_transfer_fee_by_league(table, output_dir)

@click.command()
@click.option('--output-dir', default='.', help='Directory to save output files')
@click.option('--target-column', default='transfer_fee', help='Target column for feature importance')
def main(output_dir, target_column):
    run_feature_engineering(output_dir, target_column)

if __name__ == '__main__':
    main()