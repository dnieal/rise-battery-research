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

plt.figure(1)
sns.heatmap(cm_percentage, annot=labels, fmt='', cmap='Blues', cbar=True, linewidths=0.5)
plt.title("Confusion Matrix with Percentages")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Classification Report
print("Classification Report:\n" , classification_report(y_test, predictions))

# FEATURE ranking
importances = model.feature_importances_
features = X_train.columns

indices = np.argsort(importances)[::-1]  # Sort features by importance descending

plt.figure(2)
plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()


# Plots
plt.show()