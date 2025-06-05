import click
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_rfecv(input_csv, target_column, output_csv):
    # Load dataset
    data = pd.read_csv(input_csv)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree classifier
    classifier = DecisionTreeClassifier(random_state=42)

    # Perform RFECV
    selector = RFECV(estimator=classifier, step=1, cv=5, scoring='accuracy')
    selector.fit(X_train, y_train)

    # Select features
    selected_features = X_train.columns[selector.support_]
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Train and evaluate the model with selected features
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)

    # Save selected features to CSV
    selected_features_df = pd.DataFrame({'Selected Features': selected_features})
    selected_features_df.to_csv(output_csv, index=False)

    print(f"RFECV completed! Accuracy with selected features: {accuracy:.4f}")
    print(f"Selected features saved to {output_csv}")

@click.command()
@click.option('--output-dir', default='.', help='Directory to save output CSV files')
def main(input_csv, target_column, output_csv):
    run_rfecv(input_csv, target_column, output_csv)

if __name__ == '__main__':
    main()
