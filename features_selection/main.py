import click
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from load_training_table import load_training_table

def run_rfecv(target_column, output_dir):
    # Load dataset
    data = load_training_table("../results/features_engineering_manual/feature_engineered_transfers_sv.csv", category_encode=True)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree regressor
    regressor = DecisionTreeRegressor(random_state=42)

    # Perform RFECV
    print("Starting RFECV feature selection...")
    selector = RFECV(estimator=regressor, step=1, cv=5, scoring='neg_mean_squared_error', verbose=1)
    selector.fit(X_train, y_train)
    print(f"Optimal number of features: {selector.n_features_}")
    print("Feature ranking:", selector.ranking_)

    # Select features
    selected_features = X_train.columns[selector.support_]
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Train and evaluate the model with selected features
    regressor.fit(X_train_selected, y_train)
    y_pred = regressor.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)

    # Save selected features to CSV
    selected_features_df = pd.DataFrame({'Selected Features': selected_features})
    selected_features_df.to_csv(f"{output_dir}/selected_features.csv", index=False)

    print(f"RFECV completed! MSE with selected features: {mse:.4f}")
    print(f"Selected features saved to {output_dir}/selected_features.csv")

@click.command()
@click.option('--target-column', default='transfer_fee', help='Target column for feature selection')
@click.option('--output-dir', default='.', help='Directory to save output CSV files')
def main(target_column, output_dir):
    run_rfecv(target_column, output_dir)

if __name__ == '__main__':
    main()
