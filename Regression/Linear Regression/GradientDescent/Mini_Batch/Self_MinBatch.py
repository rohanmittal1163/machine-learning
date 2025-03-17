import random
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"m: {lr.coef_}")
print(f"c: {lr.intercept_}")

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
class MBGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        # X_train[idx] = B*C
        # y_pred =  B,
        # y_train[idx] = B,
        self.coef_ = np.ones(X_train.shape[1])  # C,
        self.intercept_ = 0
        for i in range(self.epochs):
            for j in range(int(X_train.shape[0] / self.batch_size)):
                idx = random.sample(range(0, X_train.shape[0]), self.batch_size)
                y_pred = np.dot(X_train[idx], self.coef_) + self.intercept_

                dm = -2 * np.dot(y_train[idx] - y_pred, X_train[idx])
                dc = -2 * np.mean(y_train[idx] - y_pred)

                self.coef_ = self.coef_ - self.learning_rate * dm
                self.intercept_ = self.intercept_ - self.learning_rate * dc

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_

    def score(self, X_test):
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


lr = MBGDRegressor(learning_rate=0.01, epochs=100, batch_size=10)
lr.fit(X_train, y_train)
print(f"m: {lr.coef_}")
print(f"c: {lr.intercept_}")
lr.score(X_test)
