from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


lr = SGDRegressor(max_iter=10000, eta0=0.1, learning_rate="constant")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"m: {lr.coef_}")
print(f"c: {lr.intercept_}")
