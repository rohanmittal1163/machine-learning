from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import (
    ElasticNet,
)


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
reg = ElasticNet(alpha=0.005, l1_ratio=0.9)
reg.fit(X_train, y_train)
print(f"ElasticNet Regression: {r2_score(y_test,reg.predict(X_test))}")
