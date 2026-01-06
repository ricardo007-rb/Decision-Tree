"""
Full Part 2 (a–d) pipeline for ASL hand pose classification and clustering.

- Part 2a: MediaPipe feature extraction (template, commented out)
- Part 2b: Pre-processing (cleaning + scaling)
- Part 2c: Supervised learning (kNN from scratch, Decision Tree, SVM)
- Part 2d: Unsupervised learning (K-means, hierarchical clustering)

Adjust paths as needed.
"""

import os
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA


# ============================================================
# PART 2a – Feature extraction with MediaPipe (TEMPLATE ONLY)
# ============================================================
"""
This section shows how you COULD extract features with MediaPipe.
You already have asl_features.csv, so this is commented out.

Uncomment and adapt if you ever need to re-extract features.
"""

# import cv2
# import mediapipe as mp
#
# def extract_features_from_folder(image_folder, output_csv_path):
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=True,
#         max_num_hands=1,
#         min_detection_confidence=0.5
#     )
#
#     records = []
#
#     for filename in os.listdir(image_folder):
#         if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
#             continue
#
#         label = ...  # derive label from filename or folder structure
#         image_path = os.path.join(image_folder, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             continue
#
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image_rgb)
#
#         if not results.multi_hand_landmarks:
#             # Treat as noise, skip
#             continue
#
#         hand_landmarks = results.multi_hand_landmarks[0]
#         row = []
#         for lm in hand_landmarks.landmark:
#             row.extend([lm.x, lm.y, lm.z])
#
#         if len(row) == 63:
#             row.append(label)
#             records.append(row)
#
#     # Build DataFrame
#     cols = [f"f{i}" for i in range(63)] + ["label"]
#     df = pd.DataFrame(records, columns=cols)
#     df.to_csv(output_csv_path, index=False)
#
# # Example usage
# # extract_features_from_folder(
# #     image_folder=r"C:\Users\benja\OneDrive\Desktop\AI CW2\CW2_dataset_final",
# #     output_csv_path=r"C:\Users\benja\OneDrive\Desktop\AI CW2\asl_features.csv"
# # )


# ============================================================
# PART 2b – Pre-processing
# ============================================================

# 1. Load dataset
data_path = r"C:\Users\benja\OneDrive\Desktop\AI CW2\data_features\asl_features.csv"
df = pd.read_csv(data_path)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Remove rows with missing values (MediaPipe failures)
df_clean = df.dropna()
print("After dropping NaNs:", df_clean.shape)

# 3. Remove duplicate rows if any
df_clean = df_clean.drop_duplicates()
print("After dropping duplicates:", df_clean.shape)

# 4. Basic class distribution
print("\nClass distribution (after cleaning):")
print(df_clean["label"].value_counts())

plt.figure(figsize=(8, 5))
df_clean["label"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Class Distribution After Cleaning")
plt.xlabel("ASL Sign")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. Separate features and labels
X = df_clean.drop("label", axis=1).values  # shape: (n_samples, 63)
y_raw = df_clean["label"].values

# Encode labels to integers for models, but keep mapping
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_
print("\nEncoded classes:", list(class_names))

# 6. Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# PART 2c – Supervised learning
#   - Train/test split
#   - kNN from scratch
#   - Decision Tree
#   - SVM
#   - Hyperparameter tuning (5-fold CV)
#   - Best model selection + evaluation
# ============================================================

# 3.1 Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))


# ------------------------------
# 3.2 kNN from scratch (built-ins only)
# ------------------------------

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two 1D arrays/lists using pure Python.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be same length")
    acc = 0.0
    for ai, bi in zip(a, b):
        diff = ai - bi
        acc += diff * diff
    return math.sqrt(acc)


class KNNFromScratch:
    """
    k-Nearest Neighbours classifier implemented from scratch.

    - Uses only Python built-ins and math.
    - Distance: Euclidean
    """

    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Store training data as Python lists for iteration
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)

    def _predict_one(self, x):
        # Compute distance to all training points
        distances = []
        for idx, x_train in enumerate(self.X_train):
            d = euclidean_distance(x, x_train)
            distances.append((d, self.y_train[idx]))

        # Sort by distance
        distances.sort(key=lambda t: t[0])

        # Take k nearest labels
        k_nearest = [label for _, label in distances[: self.k]]

        # Majority vote
        counts = Counter(k_nearest)
        # Most common label
        return counts.most_common(1)[0][0]

    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self._predict_one(list(row)))
        return np.array(predictions)


# Baseline kNN with k=5
knn_scratch = KNNFromScratch(n_neighbors=5)
knn_scratch.fit(X_train, y_train)
y_pred_knn = knn_scratch.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
print("\n[Baseline kNN (from scratch, k=5)] Test accuracy:", acc_knn)

cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix – kNN (from scratch)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("\nClassification report – kNN (from scratch):")
print(classification_report(y_test, y_pred_knn, target_names=class_names))


# ------------------------------
# 3.3 Decision Tree – hyperparameter tuning with 5-fold CV
# ------------------------------

dt = DecisionTreeClassifier(random_state=42)

dt_param_grid = {
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_grid = GridSearchCV(
    dt,
    dt_param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)
dt_grid.fit(X_train, y_train)

print("\nBest Decision Tree params:", dt_grid.best_params_)
print("Best Decision Tree CV accuracy:", dt_grid.best_score_)

# Extract CV results for plotting
dt_results = pd.DataFrame(dt_grid.cv_results_)

# Plot effect of max_depth
plt.figure(figsize=(8, 5))
for mss in dt_param_grid["min_samples_split"]:
    subset = dt_results[dt_results["param_min_samples_split"] == mss]
    plt.plot(
        subset["param_max_depth"].astype(str),
        subset["mean_test_score"],
        marker="o",
        label=f"min_samples_split={mss}",
    )

plt.title("Decision Tree – CV Accuracy vs max_depth")
plt.xlabel("max_depth")
plt.ylabel("Mean CV Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

best_dt = dt_grid.best_estimator_
y_pred_dt = best_dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print("\n[Best Decision Tree] Test accuracy:", acc_dt)

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix – Decision Tree (best)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("\nClassification report – Decision Tree (best):")
print(classification_report(y_test, y_pred_dt, target_names=class_names))


# ------------------------------
# 3.4 SVM – hyperparameter tuning with 5-fold CV
# ------------------------------

svm = SVC(random_state=42)

svm_param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf"],  # we can also try 'linear', but keep it focused
}

svm_grid = GridSearchCV(
    svm,
    svm_param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(X_train, y_train)

print("\nBest SVM params:", svm_grid.best_params_)
print("Best SVM CV accuracy:", svm_grid.best_score_)

svm_results = pd.DataFrame(svm_grid.cv_results_)

# Plot effect of C for different gamma values
plt.figure(figsize=(8, 5))
for g in set(svm_results["param_gamma"]):
    subset = svm_results[svm_results["param_gamma"] == g]
    plt.plot(
        subset["param_C"].astype(float),
        subset["mean_test_score"],
        marker="o",
        label=f"gamma={g}",
    )

plt.xscale("log")
plt.title("SVM – CV Accuracy vs C (RBF kernel)")
plt.xlabel("C (log scale)")
plt.ylabel("Mean CV Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("\n[Best SVM] Test accuracy:", acc_svm)

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix – SVM (best)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("\nClassification report – SVM (best):")
print(classification_report(y_test, y_pred_svm, target_names=class_names))


# ------------------------------
# 3.5 Compare classifiers
# ------------------------------

results_summary = pd.DataFrame({
    "Classifier": ["kNN (scratch, k=5)", "Decision Tree (best)", "SVM (best)"],
    "Test Accuracy": [acc_knn, acc_dt, acc_svm],
})

print("\n=== Classifier Comparison (Test Accuracy) ===")
print(results_summary)

plt.figure(figsize=(7, 5))
sns.barplot(x="Classifier", y="Test Accuracy", data=results_summary, palette="pastel")
plt.ylim(0, 1)
plt.title("Classifier Test Accuracy Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Choose overall best model based on test accuracy
best_idx = results_summary["Test Accuracy"].idxmax()
overall_best_name = results_summary.loc[best_idx, "Classifier"]
overall_best_acc = results_summary.loc[best_idx, "Test Accuracy"]

print(f"\nOverall best model: {overall_best_name} (Test accuracy = {overall_best_acc:.4f})")


# ============================================================
# PART 2d – Unsupervised learning (clustering)
# ============================================================

# Use full cleaned + scaled dataset (X_scaled) without labels
# 1. K-means clustering
k = len(class_names)  # 10 clusters (same as number of classes)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels_kmeans = kmeans.fit_predict(X_scaled)

print("\nK-means cluster counts:", Counter(cluster_labels_kmeans))

# 2. Hierarchical clustering (Agglomerative)
agg = AgglomerativeClustering(n_clusters=k)
cluster_labels_agg = agg.fit_predict(X_scaled)
print("Agglomerative cluster counts:", Counter(cluster_labels_agg))

# 3. PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Plot K-means clusters in PCA space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_kmeans, cmap="tab10", s=10)
plt.title("K-means Clusters (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# Plot true labels in PCA space for comparison
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=10)
plt.title("True Labels (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
cb = plt.colorbar(scatter)
cb.set_ticks(range(len(class_names)))
cb.set_ticklabels(class_names)
plt.tight_layout()
plt.show()

# Plot Agglomerative clusters in PCA space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_agg, cmap="tab10", s=10)
plt.title("Agglomerative Clusters (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# 4. Compare clustering with true labels: contingency tables
df_clusters = pd.DataFrame({
    "true_label": y,
    "true_label_name": label_encoder.inverse_transform(y),
    "kmeans_cluster": cluster_labels_kmeans,
    "agg_cluster": cluster_labels_agg,
})

print("\n=== Contingency table: True label vs K-means cluster ===")
ct_kmeans = pd.crosstab(df_clusters["true_label_name"], df_clusters["kmeans_cluster"])
print(ct_kmeans)

print("\n=== Contingency table: True label vs Agglomerative cluster ===")
ct_agg = pd.crosstab(df_clusters["true_label_name"], df_clusters["agg_cluster"])
print(ct_agg)

# Optional: normalise rows to see purity per class
print("\nRow-normalised (proportions) – True vs K-means:")
print(ct_kmeans.div(ct_kmeans.sum(axis=1), axis=0).round(2))

print("\nRow-normalised (proportions) – True vs Agglomerative:")
print(ct_agg.div(ct_agg.sum(axis=1), axis=0).round(2))


print("\nPipeline complete: Part 2a–2d executed.")