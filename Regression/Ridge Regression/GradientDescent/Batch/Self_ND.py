import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class MeraGDRidge:
    def __init__(self, learning_rate=0.1, epochs=100, alpha=0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        beta = np.ones(X_train.shape[1])
        for i in range(self.epochs):
            dbeta = (
                np.dot(X_train.T, (np.dot(X_train, beta) - y_train)) + self.alpha * beta
            )
            beta = beta - self.learning_rate * dbeta

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        print(f"m: {self.coef_}")
        print(f"c: {self.intercept_}")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (
            len(y_test) - X_test.shape[1] - 1
        )

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Adjusted R2 Score: {adjusted_r2:.4f}")


R = MeraGDRidge(learning_rate=0.005, epochs=500, alpha=0.001)
R.fit(X_train, y_train)
R.score(X_test, y_test)
