import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(lr_model, rf_model, X_test, y_test):
    # Load models
    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    # Evaluation metrics
    print("Linear Regression:")
    print('MAE:', mean_absolute_error(y_test, y_pred_lr))
    print('MSE:', mean_squared_error(y_test, y_pred_lr))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    print('R^2:', r2_score(y_test, y_pred_lr))

    print("\nRandom Forest Regressor:")
    print('MAE:', mean_absolute_error(y_test, y_pred_rf))
    print('MSE:', mean_squared_error(y_test, y_pred_rf))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    print('R^2:', r2_score(y_test, y_pred_rf))

    # Visualizations
    plt.scatter(y_test, y_pred_lr - y_test)
    plt.title('Residuals for Linear Regression')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()

    # Feature importance
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(X_test.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(X_test.shape[1]), X_test.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
