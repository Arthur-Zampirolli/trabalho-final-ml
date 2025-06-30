import os
import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from codecarbon import EmissionsTracker
from load_training_table import load_training_table

def run_xgboost(target_column, output_dir):
    # Start CodeCarbon tracker
    tracker = EmissionsTracker(output_dir=output_dir, measure_power_secs=1, log_level="error")
    tracker.start()

    # Load dataset
    data = load_training_table(
        "../results/features_engineering_manual/feature_engineered_transfers.csv",
        category_encode=True,
    )
    X = data.drop(columns=[target_column])
    y = data[target_column]  # Keep target in original scale for final evaluation

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply log transformation to training target
    y_train_log = np.log1p(y_train)

    # Define Gradient Boosting model (not XGBoost)
    model = GradientBoostingRegressor(
        random_state=42
    )

    # Define hyperparameter grid
    params = {
        'n_estimators': [300],
        'learning_rate': [0.05],
        'max_depth': [3],
        'subsample': [0.9],
    }

    # Create custom scorer for MAE in dollars
    mae_dollar_scorer = make_scorer(
        lambda y_true_log, y_pred_log: -mean_absolute_error(
            np.expm1(y_true_log),
            np.expm1(y_pred_log)
        ),
        greater_is_better=True
    )

    # Set up GridSearchCV with custom scorer
    grid_search = GridSearchCV(
        model, 
        params, 
        scoring=mae_dollar_scorer,
        cv=5,
        verbose=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_log)

    # Use the best estimator
    best_regressor = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Stop CodeCarbon tracker
    tracker.stop()

    # Predict on test set
    y_pred_log = best_regressor.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # Convert back to dollars

    # Calculate metrics in dollars
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    print(f"MAE (test set): ${mae_test:,.2f}")
    print(f"R-squared (test set): {r2_test:.4f}")
    
    # Custom scoring functions for CV in dollars
    def mae_dollar_scorer_cv(estimator, X, y_log):
        y_pred_log = estimator.predict(X)
        return mean_absolute_error(np.expm1(y_log), np.expm1(y_pred_log))
    
    def r2_dollar_scorer_cv(estimator, X, y_log):
        y_pred_log = estimator.predict(X)
        return r2_score(np.expm1(y_log), np.expm1(y_pred_log))

    # Calculate cross-validated metrics in dollars
    cv_mae = cross_val_score(
        best_regressor, X_train, y_train_log, 
        cv=5, scoring=mae_dollar_scorer_cv
    )
    print(f"MAE (CV): ${cv_mae.mean():,.2f} ± ${cv_mae.std():,.2f}")
    cv_r2 = cross_val_score(
        best_regressor, X_train, y_train_log, 
        cv=5, scoring=r2_dollar_scorer_cv
    )
    print(f"R-squared (CV): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Save metrics and emissions
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best hyperparameters: {grid_search.best_params_}\n")
        f.write(f"MAE (test set): ${mae_test:,.2f}\n")
        f.write(f"R-squared (test set): {r2_test:.4f}\n")
        f.write(f"MAE (CV): ${cv_mae.mean():,.2f} ± ${cv_mae.std():,.2f}\n")
        f.write(f"R-squared (CV): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}\n")
        # Get emissions data safely
        f.write(f"Emissions: {getattr(tracker.final_emissions_data, 'emissions', 'N/A')} kg CO2\n")
        f.write(f"Energy: {getattr(tracker.final_emissions_data, 'energy_consumed', 'N/A')} kWh\n")
    print(f"Metrics saved to {metrics_path}")

    # Feature importance plot for GradientBoostingRegressor
    plt.figure(figsize=(12, 8))
    importances = best_regressor.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Top 20 Feature Importances (GradientBoostingRegressor)")
    plt.tight_layout()
    importance_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(importance_path, dpi=200)
    plt.close()
    print(f"Feature importance plot saved to {importance_path}")

    # Plot in dollars
    sample_size = min(100, len(y_test))
    plt.figure(figsize=(12, 6))
    indices = np.arange(sample_size)
    width = 0.35
    plt.bar(indices, y_test.iloc[:sample_size], width, label='True Values')
    plt.bar(indices + width, y_pred[:sample_size], width, label='Predicted Values')
    plt.xlabel("Sample Index")
    plt.ylabel("Transfer Fee ($)")
    plt.title("Predicted vs True Transfer Fees (XGBoost)")
    plt.legend()
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, "predicted_vs_true.png")
    plt.savefig(bar_chart_path, dpi=200)
    plt.close()
    print(f"Bar chart saved to {bar_chart_path}")

@click.command()
@click.option('--target-column', default='transfer_fee', help='Target column for regression')
@click.option('--output-dir', default='.', help='Output directory')
def main(target_column, output_dir):
    run_xgboost(target_column, output_dir)

if __name__ == '__main__':
    main()