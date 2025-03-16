import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X, y = make_regression(
    n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
m = lr.coef_
c = lr.intercept_
print(f"m : {m}")
print(f"c : {c}")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2 Score: {adjusted_r2:.4f}")


# MAIN
class GDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        self.coef_ = 0
        self.intercept_ = 0
        for i in range(self.epochs):
            y_pred = self.coef_ * X_train.ravel() + self.intercept_
            dm = (-2 / len(X_train)) * sum(X_train.ravel() * (y_train - y_pred))
            dc = (-2 / len(X_train)) * sum(y_train - y_pred)
            self.coef_ = self.coef_ - self.learning_rate * dm
            self.intercept_ = self.intercept_ - self.learning_rate * dc

    def predict(self, X_test):
        return self.coef_ * X_test.ravel() + self.intercept_

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
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


lr = GDRegressor(0.01, 1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"m: {lr.coef_}")
print(f"c: {lr.intercept_}")
lr.score(X_test, y_test)
