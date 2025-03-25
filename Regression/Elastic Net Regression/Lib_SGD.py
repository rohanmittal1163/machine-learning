from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import (
    SGDRegressor,
)


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
sgd_elastic_net = SGDRegressor(
    penalty="l1", alpha=0.01, l1_ratio=0.5, max_iter=1000, tol=1e-3
)
sgd_elastic_net.fit(X_train, y_train)

y_pred = sgd_elastic_net.predict(X_test)
print(f"Lasso Regression: {r2_score(y_test,sgd_elastic_net.predict(X_test))}")
# penalty=l1,l2,elasticnet
