import os
import click

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from load_training_table import load_training_table
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    # Compute Kendall correlation of all features with the target column
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col != target_column]
    correlations = []
    for feature in features:
        corr = table[feature].corr(table[target_column], method='kendall')
        correlations.append((feature, corr))

    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    feature_names, corr_values = zip(*correlations) if correlations else ([], [])

    plt.figure(figsize=(10, 0.6 * len(feature_names) + 2))  # Increase height for more space
    ax = sns.barplot(x=corr_values, y=feature_names, orient='h')
    ax.set_yticklabels(feature_names, fontsize=10)
    plt.title(f'Kendall Correlation with {target_column}')
    plt.xlabel('Kendall Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"kendall_correlation_with_{target_column}.png"))
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

def run_feature_engineering(output_dir, target_column):
    # Load dataset
    table = load_training_table("../results/features_engineering_manual/feature_engineered_transfers.csv", category_encode=True)

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

@click.command()
@click.option('--output-dir', default='.', help='Directory to save output files')
@click.option('--target-column', default='transfer_fee', help='Target column for feature importance')
def main(output_dir, target_column):
    run_feature_engineering(output_dir, target_column)

if __name__ == '__main__':
    main()