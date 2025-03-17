import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


df = pd.read_csv("/content/placement.csv")

# graph
plt.scatter(df["cgpa"], df["package"])
plt.xlabel("cgpa")
plt.ylabel("package (in LPA)")
plt.show()

# LR
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X.shape)  # 2d
print(y.shape)  # 1d

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

# Print results
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2 Score: {adjusted_r2:.4f}")


# graph with plot
plt.scatter(df["cgpa"], df["package"])
plt.plot(X_test, y_pred, color="red")
plt.xlabel("cgpa")
plt.ylabel("package (in LPA)")
plt.show()

# methods
m = lr.coef_
b = lr.intercept_
print(m, b)
