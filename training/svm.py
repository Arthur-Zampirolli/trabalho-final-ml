import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from codecarbon import EmissionsTracker

from load_training_table import load_training_table

def run_svr(target_column, output_dir):
  # Start CodeCarbon tracker
  # tracker = EmissionsTracker(output_dir=output_dir, measure_power_secs=1, log_level="error")
  # tracker.start()

  # Load dataset
  data = load_training_table("../results/features_engineering_manual/feature_engineered_transfers.csv", category_encode=True)
  X = data.drop(columns=[target_column])
  y = data[target_column]

  # Print the number of NaN values in each column
  nan_counts = X.isna().sum()
  print("NaN values in features:\n", nan_counts[nan_counts > 0])
  nan_target = y.isna().sum()
  print(f"NaN values in target column '{target_column}': {nan_target}")

  # Drop valuation columns if they exist
  # columns_to_drop = [
  #   "player_last_valuation",
  #   "player_highest_valuation",
  #   "player_highest_valuation_last_year",
  #   "player_highest_valuation_last_3_years",
  #   "player_avg_valuation",
  #   "player_avg_valuation_last_year",
  #   "player_avg_valuation_last_3_years"
  # ]
  # X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

  # # Normalize features between 0 and 1
  # scaler = MinMaxScaler()
  # X_scaled = scaler.fit_transform(X)

  # # Split data into training and testing sets
  # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # # Define parameter grid for SVR
  # param_grid = {
  #     'C': [0.1, 1, 10, 100],
  #     'epsilon': [0.01, 0.1, 1, 10],
  #     'kernel': ['rbf', 'linear'],
  #     'gamma': ['scale', 'auto']
  # }

  # # Initialize SVR regressor
  # regressor = SVR()

  # # Set up GridSearchCV
  # grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
  # grid_search.fit(X_train, y_train)

  # # Use the best estimator found by grid search
  # regressor = grid_search.best_estimator_
  # print(f"Best hyperparameters: {grid_search.best_params_}")

  # # Train the regressor
  # regressor.fit(X_train, y_train)

  # # Predict on the test set
  # y_pred = regressor.predict(X_test)

  # # Stop CodeCarbon tracker and get emissions/energy data
  # tracker.stop()

  # # Calculate Mean Squared Error
  # mse = mean_squared_error(y_test, y_pred)
  # print(f"Mean Squared Error on test set: {mse}")
  # rmse = np.sqrt(mse)
  # print(f"RMSE: {rmse}")
  # # Calculate Mean Absolute Error
  # mae = mean_absolute_error(y_test, y_pred)
  # print(f"MAE: {mae}")
  # # Calculate R-squared
  # r2 = r2_score(y_test, y_pred)
  # print(f"R-squared: {r2}")

  # # 5-fold cross-validation on the training set
  # cv_scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
  # cv_rmse_scores = np.sqrt(-cv_scores)
  # print(f"5-fold CV RMSE scores: {cv_rmse_scores}")
  # print(f"5-fold CV RMSE mean: {cv_rmse_scores.mean()}, std: {cv_rmse_scores.std()}")

  # # Save metrics and emissions to a txt file
  # os.makedirs(output_dir, exist_ok=True)
  # metrics_path = os.path.join(output_dir, "svr_metrics.txt")
  # with open(metrics_path, "w") as f:
  #     f.write(f"Best hyperparameters: {grid_search.best_params_}\n")
  #     f.write(f"Mean Squared Error on test set: {mse}\n")
  #     f.write(f"RMSE: {rmse}\n")
  #     f.write(f"MAE: {mae}\n")
  #     f.write(f"R-squared: {r2}\n")
  #     f.write(f"5-fold CV RMSE mean: {cv_rmse_scores.mean()}, std: {cv_rmse_scores.std()}\n")
  # print(f"Metrics saved to {metrics_path}")

  # # Plot and save bar chart of predicted vs true values for a sample
  # sample_size = min(100, len(y_test))
  # y_test_sample = y_test[:sample_size].reset_index(drop=True)
  # y_pred_sample = pd.Series(y_pred[:sample_size])

  # plt.figure(figsize=(12, 6))
  # width = 0.35
  # indices = range(sample_size)
  # plt.bar(indices, y_test_sample, width=width, label='True Values')
  # plt.bar([i + width for i in indices], y_pred_sample, width=width, label='Predicted Values')
  # plt.xlabel("Sample Index")
  # plt.ylabel("Value")
  # plt.title("Bar Chart: Predicted vs True Values (Sample)")
  # plt.legend()
  # plt.tight_layout()
  # bar_chart_path = os.path.join(output_dir, "bar_predicted_vs_true.png")
  # plt.savefig(bar_chart_path, dpi=200, bbox_inches='tight')
  # plt.close()
  # print(f"Bar chart of predicted vs true values saved to {bar_chart_path}")

@click.command()
@click.option('--target-column', default='transfer_fee', help='Target column for regression')
@click.option('--output-dir', default='.', help='Directory to save output CSV files')
def main(target_column, output_dir):
    run_svr(target_column, output_dir)

if __name__ == '__main__':
    main()
