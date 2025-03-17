import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True, as_frame=True)
df = pd.concat([X, y], axis=1)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class MultiLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        beta = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_


lr = MultiLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("r2 score: ", r2_score(y_test, y_pred))
print("Coef: ", lr.coef_)
print("Intercept: ", lr.intercept_)
