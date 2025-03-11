import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SimpleLR:
    def __init__(self):
        self.m = None
        self.c = None

    def fit(self, X_train, y_train) -> None:
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))
        self.m = num / den
        self.c = y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):
        return self.m * X_test + self.c

    def intercept(self):
        return self.c

    def coef(self):
        return self.m


df = pd.read_csv("placement.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
lr = SimpleLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
