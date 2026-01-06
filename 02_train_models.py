#02_train_models.py

import pandas as pd

# Load extracted features
df = pd.read_csv("data_features/asl_features.csv")

print("Loaded dataset with shape:", df.shape)
print(df.head())

from sklearn.model_selection import train_test_split

# Separate features (X) and labels (y)
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("KNN Accuracy:", acc_knn)

from sklearn.svm import SVC

# Train SVM model
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train, y_train)

# Evaluate
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("SVM Accuracy:", acc_svm)

from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluate
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", acc_dt)

from sklearn.metrics import classification_report, confusion_matrix

print("\n=== KNN Classification Report ===")
print(classification_report(y_test, y_pred_knn))

print("\n=== SVM Classification Report ===")
print(classification_report(y_test, y_pred_svm))

print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test, y_pred_dt))

print("\n=== KNN Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_knn))

print("\n=== SVM Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_svm))

print("\n=== Decision Tree Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_dt))


import matplotlib.pyplot as plt

# Accuracy values
accuracies = [acc_knn, acc_svm, acc_dt]
models = ["KNN", "SVM", "Decision Tree"]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=["skyblue", "lightgreen", "salmon"])
plt.ylim(0.8, 1.0)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

import seaborn as sns

def plot_confusion(cm, title):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion(confusion_matrix(y_test, y_pred_knn), "KNN Confusion Matrix")
plot_confusion(confusion_matrix(y_test, y_pred_svm), "SVM Confusion Matrix")
plot_confusion(confusion_matrix(y_test, y_pred_dt), "Decision Tree Confusion Matrix")