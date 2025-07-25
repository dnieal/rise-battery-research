import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Data/Analysis Data/info2.csv")
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
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_train, model.predict(X_train)))
print("\nClassification Report:\n", classification_report(y_train, model.predict(X_train)))

# Get coefficients and sort by absolute value
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
})
coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
sorted_coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)

# Print sorted coefficients
print(sorted_coef_df[['feature', 'coefficient']])
