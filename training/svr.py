import os
import click
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from codecarbon import EmissionsTracker

from load_training_table import load_training_table

def run_svr(target_column, output_dir):
    # Start CodeCarbon tracker
    tracker = EmissionsTracker(output_dir=output_dir, measure_power_secs=1, log_level="error")
    tracker.start()

    # Load dataset
    data = load_training_table(
        "../results/features_engineering_manual/feature_engineered_transfers.csv",
        category_encode=True,
    )
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Apply log-transform to the target to normalize its distribution
    y_log = np.log1p(y)  # Use log(1+x) to avoid issues with zeros

    X_train, X_test, y_train_log, _ = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    _, _, _, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Drop valuation columns if they exist
    # columns_to_drop = [
    #     "player_last_valuation",
    #     "player_highest_valuation",
    #     "player_highest_valuation_last_year",
    #     "player_highest_valuation_last_3_years",
    #     "player_avg_valuation",
    #     "player_avg_valuation_last_year",
    #     "player_avg_valuation_last_3_years"
    # ]
    # cols_to_drop = [col for col in columns_to_drop if col in X_train.columns]
    # # Drop from original DataFrames
    # X_train = X_train.drop(columns=cols_to_drop)
    # X_test = X_test.drop(columns=cols_to_drop)

    # Normalize features between 0 and 1
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid for SVR
    param_grid = [
        # Best linear hyperparameters (from previous runs)
        # {'C': [1], 'epsilon': [0.1], 'kernel': ['linear']},
        # Best RBF hyperparameters (from previous runs)
        {'kernel': ['rbf'], 'C': [10], 'epsilon': [0.2], 'gamma': ['scale']},
        # Best polynomial hyperparameters (from previous runs)
        # {'C': [1], 'coef0': [1], 'degree': [2], 'epsilon': [0.1], 'gamma': ['scale'], 'kernel': ['poly']},
        # Best sigmoid hyperparameters (from previous runs)
        # {'C': [0.1], 'coef0': [1], 'epsilon': [0.5], 'gamma': ['scale'], 'kernel': ['sigmoid']},

        # RBF Kernel (your best performer)
        # {
        #     'kernel': ['rbf'],
        #     'C': [1, 10, 50, 100, 200],
        #     'epsilon': [0.005, 0.01, 0.05, 0.1, 0.2],
        #     'gamma': ['scale', 'auto', 0.01, 0.1, 1.0]
        # },
        # # Linear Kernel (simpler model)
        # Polynomial Kernel (captures interactions)
        # {
        #     'kernel': ['poly'],
        #     'degree': [2, 3],          # Higher degrees risk overfitting
        #     'C': [1, 10, 100],
        #     'gamma': ['scale', 'auto'],
        #     'epsilon': [0.1, 1.0],
        #     'coef0': [0, 1]            # Constant term in kernel function
        # },
        # Sigmoid Kernel (neural network-like)
        # {
        #     'kernel': ['sigmoid'],
        #     'C': [0.1, 1, 10],
        #     'gamma': ['scale', 0.1, 1],
        #     'epsilon': [0.1, 0.5],
        #     'coef0': [0, 0.5, 1]
        # }
    ]
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        SVR(),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=3,
    )
    grid_search.fit(X_train_scaled, y_train_log)

    # Use the best estimator found by grid search
    best_regressor = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Train the regressor
    y_pred_log = best_regressor.predict(X_test_scaled)

    # Predict on the test set
    y_pred_orig = np.expm1(y_pred_log)

    # Stop CodeCarbon tracker and get emissions/energy data
    tracker.stop()

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    print(f"Mean Squared Error on test set: {mse}")
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    print(f"MAE: {mae}")
    # Calculate R-squared
    r2 = grid_search.best_score_
    print(f"R-squared: {r2}")

    # Save metrics and emissions to a txt file
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best hyperparameters: {grid_search.best_params_}\n")
        f.write(f"Mean Squared Error on test set: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R-squared (5 fold CV): {r2}\n")
    print(f"Metrics saved to {metrics_path}")

    # Plot and save bar chart of predicted vs true values for a sample
    sample_size = min(100, len(y_test_orig))
    # Ensure alignment by using iloc and numpy arrays
    y_test_sample = y_test_orig.iloc[:sample_size].to_numpy()
    y_pred_sample = y_pred_orig[:sample_size]

    plt.figure(figsize=(12, 6))
    width = 0.35
    indices = np.arange(sample_size)
    plt.bar(indices, y_test_sample, width=width, label='True Values')
    plt.bar(indices + width, y_pred_sample, width=width, label='Predicted Values')
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
@click.option('--target-column', default='transfer_fee', help='Target column for regression')
@click.option('--output-dir', default='.', help='Directory to save output CSV files')
def main(target_column, output_dir):
    run_svr(target_column, output_dir)

if __name__ == '__main__':
    main()
