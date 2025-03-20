import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge

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
print(f"Adjusted R2 Score: {adjusted_r2:.4f}\n")

# MAIN
R = Ridge(0.01)
R.fit(X_train, y_train)
y_pred = R.predict(X_test)

print(f"m: {R.coef_}")
print(f"c: {R.intercept_}")
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


# for calculating optimal alpha where bias and variance intersects
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# poly=PolynomialFeatures(degree=2)
# X_train=poly.fit_transform(X_train)
# X_test=poly.transform(X_test)

alphas = np.linspace(0, 30, 1000)
loss = []
bias = []
var = []

for i in alphas:
    reg = Ridge(i)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        reg, X_train, y_train, X_test, y_test, loss="mse", random_seed=123
    )
    loss.append(avg_expected_loss)
    bias.append(avg_bias)
    var.append(avg_var)
plt.plot(alphas, loss, label="Loss")
plt.plot(alphas, bias, label="Bias")
plt.plot(alphas, var, label="Variance")
plt.legend()
plt.show()
