import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap

dataset = pd.read_csv("rise-battery-research\Data\Analysis Data\infoBatName.csv")
dataset = dataset.drop(["Percent Capacity Decrease"], axis = 1)


drop_list = [['Commercial', 'Residential'], ['Calendar', 'No Calendar'], ['1C','3C','C/2','C/4'],
             ['g1','v4','v5','w10','w8','w9'], ['Discharge Capacity'], ['Current Cycles'], ['Temperature'], 
             ['Past Cycles'], ['Past Discharge Capacity']]

for x in drop_list:
    print(x)
    y = dataset.drop(x, axis = 1)

    scaler = StandardScaler()

    X = y.drop(["Category"], axis = 1)
    X = pd.DataFrame(scaler.fit_transform(X), columns= X.columns)
    y = y["Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n", confusion_matrix(y_train, model.predict(X_train)))
    print("\nClassification Report:\n", classification_report(y_train, model.predict(X_train)))