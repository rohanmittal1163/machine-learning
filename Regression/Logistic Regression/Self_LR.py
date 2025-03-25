from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import expit

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    random_state=3,
    n_classes=2,
    n_clusters_per_class=1,
    hypercube=False,
    class_sep=10,
)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# Perceptron Activation Function
def step(z):
    return 0 if z <= 0 else 1


# Logistic Regression Activation Function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def perceptron_step(X, y):
    X = np.insert(X, 0, 1, axis=1)
    w = np.ones(X.shape[1])
    lr = 0.03
    epochs = 10000
    for i in range(epochs):
        j = np.random.randint(0, X.shape[0])
        y_pred = step(np.dot(X[j], w))
        w = w + lr * (y[j] - y_pred) * X[j]
    return w[0], w[1:]


def LogisticRegression_sigmoid(X, y):
    X = np.insert(X, 0, 1, axis=1)
    w = np.ones(X.shape[1])
    lr = 0.03
    epochs = 10000
    for i in range(epochs):
        j = np.random.randint(0, X.shape[0])
        y_pred = expit(np.dot(X[j], w))
        w = w + lr * (y[j] - y_pred) * X[j]
    return w[0], w[1:]


def logisiticRegression_batchGD(X, y):
    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    epochs = 10000
    lr = 0.01
    for i in range(epochs):
        y_hat = expit(np.dot(X, weights))
        weights = weights + lr * np.dot((y - y_hat), X) / X.shape[0]
    return weights[0], weights[1:]


intercept_, coef_ = perceptron_step(X, y)
m = -coef_[0] / coef_[1]
c = -intercept_ / coef_[1]

intercept_, coef_ = LogisticRegression_sigmoid(X, y)
m1 = -coef_[0] / coef_[1]
c1 = -intercept_ / coef_[1]

intercept_, coef_ = logisiticRegression_batchGD(X, y)
m2 = -coef_[0] / coef_[1]
c2 = -intercept_ / coef_[1]


x_input = np.linspace(-3, 3, 100)
plt.plot(x_input, m * x_input + c, color="red")
plt.plot(x_input, m1 * x_input + c1, color="blue")
plt.plot(x_input, m2 * x_input + c2, color="green")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
