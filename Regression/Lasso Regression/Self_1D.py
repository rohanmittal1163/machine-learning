import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso


df = pd.read_csv("placement.csv")
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


l = Lasso(0.01)
l.fit(X_train, y_train)

y_pred = l.predict(X_test)

print(f"m: {l.coef_}")
print(f"c: {l.intercept_}")
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


class Mera1DLasso:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = 0
        X_train_mean = X_train.mean()
        y_train_mean = y_train.mean()
        num = np.sum((X_train.ravel() - X_train_mean) * (y_train - y_train_mean))
        den = np.sum((X_train - X_train_mean) ** 2)
        if num - self.alpha > 0:
            m = (num - self.alpha) / den
        elif num + self.alpha < 0:
            m = (num + self.alpha) / den
        else:
            m = 0
        self.coef_ = m
        self.intercept_ = y_train_mean - m * X_train_mean

    def predict(self, X_test):
        return self.coef_ * X_test + self.intercept_

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
        print(f"Adjusted R2 Score: {adjusted_r2:.4f}\n")


l = Mera1DLasso(0.01)
l.fit(X_train, y_train)
l.score(X_test, y_test)
