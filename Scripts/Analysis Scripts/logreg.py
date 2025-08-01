# Importing modules
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

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_train, model.predict(X_train)))

# Confusion matrix code 
cm = confusion_matrix(y_test, y_pred)
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

# Get coefficients and sort by absolute value
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
})
coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
sorted_coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)

# Print sorted coefficients
print(sorted_coef_df[['feature', 'coefficient']])

# SHAP
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer(X_test)

# beeswarm plot with shap values

plt.figure(2)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test, plot_type="dot", max_display=21, show=False)
plt.savefig("beeswarm_linearexplainer.png", bbox_inches="tight")

plt.figure(3)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test, plot_type="dot", max_display=5, show=False)
plt.savefig("beeswarm_linearexplainer.png", bbox_inches="tight")

# Print both Beeswarm and Confusion Matrix plots
plt.show()
