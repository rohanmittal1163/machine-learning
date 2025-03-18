import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


class MeraRidge:
    def __init__(self, alpha=0.1):
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = 0
        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        num = np.sum((y_train - y_mean) * (X_train - X_mean))
        den = np.sum(np.power(X_train - X_mean, 2)) + self.alpha
        self.coef_ = num / den
        self.intercept_ = y_mean - self.coef_ * X_mean

    def predict(self, X_test):
        return self.intercept_ + self.coef_ * X_test

    def score(self, X_test):
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


R = MeraRidge(0.01)
R.fit(X_train, y_train)
R.score(X_test)
