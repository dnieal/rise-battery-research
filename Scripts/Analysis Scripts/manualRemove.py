import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset = pd.read_csv("rise-battery-research\Data\Analysis Data\infoBatName.csv")
dataset = dataset.drop(["Percent Capacity Decrease"], axis = 1)


drop_list = [[]
             , ['Commercial', 'Residential'], ['Calendar', 'No Calendar'], ['1C','3C','C/2','C/4'],
             ['g1','v4','v5','w10','w8','w9'], ['Discharge Capacity'], ['Current Cycles'], ['Temperature'], 
             ['Past Cycles'], ['Past Discharge Capacity']
             ]
v =["Baseline", "Removed Usage Type", "Removed Calendar Aging", "Removed First Life Charge Rate", "Removed Battery Name",
    "Removed Discharge Capacity", "Removed Current Cycles", "Removed Temperatures", "Removed Past Cycles", "Removed Past Discharge Capacity"]

for i in range(10):
    x = drop_list[i]
    varname = v[i]
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

    # cm = confusion_matrix(y_test, y_pred)
    # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # labels = np.array([
    #     [f"{int(count)}\n{percent:.1f}%" for count, percent in zip(row_counts, row_percents)]
    #     for row_counts, row_percents in zip(cm, cm_percentage)])
        
    # print(labels)
        
    # plt.figure(1)
    # sns.heatmap(cm_percentage, annot=labels
    #     , fmt=''
    #     , cmap='Blues', cbar=True, linewidths=0.5
    #     )

    # plt.title(f"Confusion Matrix for {varname}")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.tight_layout()
    # plt.show()


    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) for {varname}')
    plt.legend(loc="lower right")
    plt.grid(True)
    name = f"rise-battery-research/Output/Results/roc {varname}.png"
    plt.savefig(name)  


# Commenting this out so we can run the code for all confusion matrices

# # CONFUSION MATRIX
# cm = confusion_matrix(y_test, y_pred)
# cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# labels = np.array([
#     [f"{int(count)}\n{percent:.1f}%" for count, percent in zip(row_counts, row_percents)]
#     for row_counts, row_percents in zip(cm, cm_percentage)
# ])

# plt.figure(1)
# sns.heatmap(cm_percentage, annot=labels, fmt='', cmap='Blues', cbar=True, linewidths=0.5)
# plt.title("Confusion Matrix with Percentages")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.tight_layout()

# # SHAP
# explainer = shap.LinearExplainer(model, X_train)
# shap_values = explainer(X_test)

# # beeswarm plot with shap values
# plt.figure(2)
# plt.figure(figsize=(10, 8))
# shap.summary_plot(shap_values.values, X_test, plot_type="dot", max_display=21, show=False)
# plt.savefig("beeswarm_linearexplainer.png", bbox_inches="tight")

# plt.figure(3)
# plt.figure(figsize=(10, 8))
# shap.summary_plot(shap_values.values, X_test, plot_type="dot", max_display=5, show=False)
# plt.savefig("beeswarm_linearexplainer.png", bbox_inches="tight")

# # Print both Beeswarm and Confusion Matrix plots
# plt.show()
