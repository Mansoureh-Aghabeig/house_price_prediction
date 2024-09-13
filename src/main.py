from preprocessing import preprocess_data
from model import train_models
from evaluation import evaluate_models
import pandas as pd
import os
import seaborn as sns

def main():
    # Load and preprocess data
    # Adjust the path to go one level up to access the 'data' folder
    csv_path = os.path.join('..', 'data', 'USA_Housing.csv')
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Train models
    lr_model, rf_model = train_models(X_train, y_train)

    # Evaluate models
    evaluate_models(lr_model, rf_model, X_test, y_test)


if __name__ == "__main__":
    import sklearn

    print(sklearn.__version__)
    #main()
