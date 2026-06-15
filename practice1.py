# class 7: MOdel evaluation metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score , roc_curve , roc_auc_score)

data = {
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "sleep": [9,8,8,7,7,6,6,5,5,4],
    "passed": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)
X = df[["hours", "sleep"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = 1
)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Normal prediction: ", pred)

acc = accuracy_score(y_test, pred)
print("Accuracy: ", acc)

proba = model.predict_proba(X_test)
print("Probabilities: ", proba)

class1_proba = proba[:, 1]
print("Class 1 prob: ", class1_proba)

fpr, tpr, thresholds = roc_curve(
    y_test, class1_proba
)

auc_score = roc_auc_score(
    y_test, class1_proba
)
print("Auc score: ", auc_score)

# plot roc curve
plt.figure(figsize=(6,4))
plt.plot(
    fpr, tpr, label=f"AUC = {auc_score:.2f}"
)
# random guessing
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc curve")
plt.legend()
plt.show()
