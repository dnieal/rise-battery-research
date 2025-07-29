import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset = pd.read_csv("rise-battery-research\Data\Analysis Data\info2.csv")
dataset = dataset.drop(["Battery Name", "RPT Number", "Discharge Capacity", "Past Discharge Capacity", "Percent Capacity Decrease"], axis = 1)

scaler = StandardScaler()

X = dataset.drop(["Category"], axis = 1)
X = pd.DataFrame(scaler.fit_transform(X), columns= X.columns)
y = dataset["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_train, model.predict(X_train)))

# Confusion matrix code 
title='Confusion Matrix'
y_true = y_test
cm = confusion_matrix(y_true, y_pred)
cm_sum = np.sum(cm)
cm_perc = cm / cm_sum * 100  # <- percentage

annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape

for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        s = f"{c}\n{p:.1f}%" if c != 0 else "0\n0.0%"
        annot[i, j] = s

plt.figure(1)
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(title)
plt.tight_layout()

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
shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0), max_display = 11)

# Print both Beeswarm and Confusion Matrix plots
plt.show()