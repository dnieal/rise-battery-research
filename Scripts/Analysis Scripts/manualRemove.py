import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
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

    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = np.array([
        [f"{int(count)}\n{percent:.1f}%" for count, percent in zip(row_counts, row_percents)]
        for row_counts, row_percents in zip(cm, cm_percentage)])
    
    plt.figure()
    sns.heatmap(cm_percentage, annot=labels
        , fmt=''
        , cmap='Blues', cbar=True, linewidths=0.5
        )

    plt.title(f"Confusion Matrix for {varname}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    name = f"rise-battery-research/Output/Results/Confusion Matrix for {varname}.png"
    plt.savefig(name)

    # BOOTSTRAPPING FOR STD ERROR BAR PLOTS
    n_iterations = 1000
    coefs = []

    for j in range(n_iterations):
        X_sample, y_sample = resample(X_train, y_train)
        model = LogisticRegression()
        model.fit(X_sample, y_sample)
        coefs.append(model.coef_[0])

    coefs = np.array(coefs)
    coefs = np.array(coefs)
    coef_means = np.mean(coefs, axis=0)
    coef_stds = np.std(coefs, axis=0)
    feature_names = np.array(X.columns)  # make sure it's a NumPy array

    # Get indices of the 5 largest coefficients by absolute value
    top5_idx = np.argsort(np.abs(coef_means))[-5:][::-1]

    # Slice data for the top 5
    top5_means = coef_means[top5_idx]
    top5_stds = coef_stds[top5_idx]
    top5_names = feature_names[top5_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(top5_names, top5_means, yerr=top5_stds, capsize=5, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks(rotation=90, fontweight='bold')
    plt.title('Top 5 Logistic Regression Coefficients with Bootstrapped Standard Errors', fontweight='bold')
    plt.ylabel('Coefficient Value', fontweight='bold')
    plt.tight_layout()
    name_bar = f"rise-battery-research/Output/Results/Bar Plot with std error bars for {varname}.png"
    plt.savefig(name_bar)


# PLOT ACCURACY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load your dataset
dataset = pd.read_csv("rise-battery-research/Data/Analysis Data/infoBatName.csv")
dataset = dataset.drop(["Percent Capacity Decrease"], axis=1)

drop_list = [
    [],
    ['Commercial', 'Residential'],
    ['Calendar', 'No Calendar'],
    ['1C','3C','C/2','C/4'],
    ['g1','v4','v5','w10','w8','w9'],
    ['Discharge Capacity'],
    ['Current Cycles'],
    ['Temperature'],
    ['Past Cycles'],
    ['Past Discharge Capacity']
]

v = [
    "Baseline",
    "Removed Usage Type",
    "Removed Calendar Aging",
    "Removed First Life Charge Rate",
    "Removed Battery Name",
    "Removed Discharge Capacity",
    "Removed Current Cycles",
    "Removed Temperatures",
    "Removed Past Cycles",
    "Removed Past Discharge Capacity"
]

mean_accuracies = []
std_accuracies = []

# Main loop through feature removal scenarios
for k in range(10):
    drop_features = drop_list[k]
    varname = v[k]

    y_df = dataset.drop(drop_features, axis=1)

    scaler = StandardScaler()
    X = y_df.drop(["Category"], axis=1)
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    y = y_df["Category"]

    # Bootstrapping accuracy estimates
    boot_accuracies = []
    for l in range(100):  # 100 resamples per config
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=l
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        boot_accuracies.append(acc)

    # Collect mean and std dev (or std error)
    mean_accuracies.append(np.mean(boot_accuracies))
    std_accuracies.append(np.std(boot_accuracies))

# PLOT ACCURACIES - mean accuracy from 100 train/test splits, bars are standard deviation bars
plt.figure(figsize=(12, 6))
plt.bar(v, mean_accuracies, yerr=std_accuracies, capsize=5, color='skyblue')
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.ylabel("Accuracy", fontweight='bold')
plt.title("Model Accuracy by Feature Removal (with Std Dev Error Bars)", fontweight='bold')
plt.tight_layout()
plt.savefig("rise-battery-research/Output/Results/All_Model_Accuracies_with_Error_Bars.png")

# SHAP
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer(X_test)

# beeswarm plot with shap values
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test, plot_type="dot", max_display=21, show=False)
plt.savefig("rise-battery-research/Output/Results/Full SHAP for Manual Removal Regression.png", bbox_inches="tight")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test, plot_type="dot", max_display=5, show=False)
plt.savefig("rise-battery-research/Output/Results/Partial SHAP for Manual Removal Regression.png", bbox_inches="tight")
