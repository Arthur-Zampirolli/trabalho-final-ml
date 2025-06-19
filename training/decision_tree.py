import click
import numpy as np
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from load_training_table import load_training_table
import os
from sklearn import tree
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

def run_decision_tree(target_column, output_dir):
  # Load dataset
  data = load_training_table("../results/features_engineering_manual/feature_engineered_transfers.csv", category_encode=True)
  X = data.drop(columns=[target_column])
  y = data[target_column]

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Define parameter grid for DecisionTreeRegressor
  param_grid = {
      'max_depth': [5, 10, 15, 20, None],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'max_features': [None, 'sqrt', 'log2']
  }

  # Initialize Decision Tree regressor
  regressor = DecisionTreeRegressor(random_state=42)

  # Set up GridSearchCV
  grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
  grid_search.fit(X_train, y_train)

  # Use the best estimator found by grid search
  regressor = grid_search.best_estimator_
  print(f"Best hyperparameters: {grid_search.best_params_}")

  # Train the regressor
  regressor.fit(X_train, y_train)

  # Predict on the test set
  y_pred = regressor.predict(X_test)

  # Print the best hyperparameters
  print(f"Best hyperparameters: {grid_search.best_params_}")
  # Calculate Mean Squared Error
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error on test set: {mse}")
  rmse = np.sqrt(mse)
  print(f"RMSE: {rmse}")

  # Save decision tree image
  os.makedirs(output_dir, exist_ok=True)
  # Adjust font size and layout for better readability
  fig, ax = plt.subplots(figsize=(24, 16))
  tree.plot_tree(
      regressor,
      feature_names=X.columns,
      filled=True,
      ax=ax,
      fontsize=12,  # Increase font size for readability
      rounded=True,
      precision=2
  )
  plt.tight_layout()
  image_path = os.path.join(output_dir, "decision_tree.png")
  plt.savefig(image_path, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f"Decision tree image saved to {image_path}")

  # Plot and save bar chart of predicted vs true values for a sample
  sample_size = min(100, len(y_test))
  y_test_sample = y_test[:sample_size].reset_index(drop=True)
  y_pred_sample = pd.Series(y_pred[:sample_size])

  plt.figure(figsize=(12, 6))
  width = 0.35
  indices = range(sample_size)
  plt.bar(indices, y_test_sample, width=width, label='True Values')
  plt.bar([i + width for i in indices], y_pred_sample, width=width, label='Predicted Values')
  plt.xlabel("Sample Index")
  plt.ylabel("Value")
  plt.title("Bar Chart: Predicted vs True Values (Sample)")
  plt.legend()
  plt.tight_layout()
  bar_chart_path = os.path.join(output_dir, "bar_predicted_vs_true.png")
  plt.savefig(bar_chart_path, dpi=200, bbox_inches='tight')
  plt.close()
  print(f"Bar chart of predicted vs true values saved to {bar_chart_path}")

@click.command()
@click.option('--target-column', default='transfer_fee', help='Target column for feature selection')
@click.option('--output-dir', default='.', help='Directory to save output CSV files')
def main(target_column, output_dir):
    run_decision_tree(target_column, output_dir)

if __name__ == '__main__':
    main()
