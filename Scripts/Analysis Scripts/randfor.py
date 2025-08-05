import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Import Dataset
dataset = pd.read_csv("rise-battery-research\Data\Analysis Data\info2.csv")
dataset = dataset.drop(["Battery Name", "RPT Number", "Discharge Capacity", "Past Discharge Capacity", "Percent Capacity Decrease"], axis = 1)

# Normalize data
scaler = StandardScaler()

# Split the data
X = dataset.drop(["Category"], axis = 1)
X = pd.DataFrame(scaler.fit_transform(X), columns= X.columns)
y = dataset["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestClassifier(n_estimators= 200, max_depth= 5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Random Forest results:

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

labels = np.array([
    [f"{int(count)}\n{percent:.1f}%" for count, percent in zip(row_counts, row_percents)]
    for row_counts, row_percents in zip(cm, cm_percentage)
])

sns.heatmap(cm_percentage, annot=labels, fmt='', cmap='Blues', cbar=True, linewidths=0.5)
plt.title("Confusion Matrix with Percentages")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("rise-battery-research/Output/Results/Confusion Matrix for Random Forest", bbox_inches="tight")

# Classification Report
print("Classification Report:\n" , classification_report(y_test, predictions))

# FEATURE ranking
importances = model.feature_importances_
features = X_train.columns

# Print importances
sorted_features = sorted(zip(importances, features), reverse=True)
for importance, feature in sorted_features:
    print(f"{importance:.2f} : {feature}")
indices = np.argsort(importances)[::-1]  # Sort features by importance descending

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.savefig("rise-battery-research/Output/Results/Random Forrest Feature Importances", bbox_inches="tight")

# PLOT WITH STD BARS
# Extract feature importances for each tree
all_importances = np.array([tree.feature_importances_ for tree in model.estimators_])

# Calculate mean and std deviation of feature importances across trees
mean_importances = np.mean(all_importances, axis=0)
std_importances = np.std(all_importances, axis=0)

# Sort features by mean importance descending
indices = np.argsort(mean_importances)[::-1]
sorted_features = X_train.columns[indices]
sorted_means = mean_importances[indices]
sorted_stds = std_importances[indices]

# Plot with error bars
plt.figure(figsize=(12,6))
plt.bar(range(len(sorted_means)), sorted_means, yerr=sorted_stds, capsize=5, color='skyblue')
plt.xticks(range(len(sorted_means)), sorted_features, rotation=90, fontweight='bold')
plt.ylabel("Mean Feature Importance", fontweight='bold')
plt.title("Random Forest Feature Importances with Standard Deviation", fontweight='bold')
plt.tight_layout()
plt.savefig("rise-battery-research/Output/Results/Random Forest Feature Importances with STD.png")
