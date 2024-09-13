from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_models(X_train, y_train):
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=101)
    rf_model.fit(X_train, y_train)

    # Save models
    model_path_lr = os.path.join('..', 'models', 'lr_model.pkl')
    model_path_rf = os.path.join('..', 'models', 'rf_model.pkl')

    with open(model_path_lr, 'wb') as file:
        pickle.dump(lr_model, file)
    with open(model_path_rf, 'wb') as file:
        pickle.dump(rf_model, file)

    return lr_model, rf_model
