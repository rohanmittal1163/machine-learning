import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

df = pd.read_csv("/content/heart.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, [-1]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lor = LogisticRegression()
lor.fit(X_train, y_train)
y_pred = lor.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision
print(
    "Precision (None):", precision_score(y_test, y_pred, average=None)
)  # Per-class precision
print("Precision (Macro):", precision_score(y_test, y_pred, average="macro"))
print("Precision (Weighted):", precision_score(y_test, y_pred, average="weighted"))
print("Precision (Micro):", precision_score(y_test, y_pred, average="micro"))

# Recall
print("Recall (None):", recall_score(y_test, y_pred, average=None))  # Per-class recall
print("Recall (Macro):", recall_score(y_test, y_pred, average="macro"))
print("Recall (Weighted):", recall_score(y_test, y_pred, average="weighted"))
print("Recall (Micro):", recall_score(y_test, y_pred, average="micro"))

# F1-score
print("F1-score (None):", f1_score(y_test, y_pred, average=None))  # Per-class F1-score
print("F1-score (Macro):", f1_score(y_test, y_pred, average="macro"))
print("F1-score (Weighted):", f1_score(y_test, y_pred, average="weighted"))
print("F1-score (Micro):", f1_score(y_test, y_pred, average="micro"))


# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision
print(
    "Precision (None):", precision_score(y_test, y_pred, average=None)
)  # Per-class precision
print("Precision (Macro):", precision_score(y_test, y_pred, average="macro"))
print("Precision (Weighted):", precision_score(y_test, y_pred, average="weighted"))
print("Precision (Micro):", precision_score(y_test, y_pred, average="micro"))

# Recall
print("Recall (None):", recall_score(y_test, y_pred, average=None))  # Per-class recall
print("Recall (Macro):", recall_score(y_test, y_pred, average="macro"))
print("Recall (Weighted):", recall_score(y_test, y_pred, average="weighted"))
print("Recall (Micro):", recall_score(y_test, y_pred, average="micro"))

# F1-score
print("F1-score (None):", f1_score(y_test, y_pred, average=None))  # Per-class F1-score
print("F1-score (Macro):", f1_score(y_test, y_pred, average="macro"))
print("F1-score (Weighted):", f1_score(y_test, y_pred, average="weighted"))
print("F1-score (Micro):", f1_score(y_test, y_pred, average="micro"))

cm = confusion_matrix(y_test, y_pred)
acc = np.trace(cm) / np.sum(cm)

print(classification_report(y_test, y_pred))
