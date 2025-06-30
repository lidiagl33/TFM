from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
df = pd.read_csv("dataset.csv")
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Models
models = {
    "GaussianNB": GaussianNB(var_smoothing=1e-09),
    "DecisionTree": DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=2, min_samples_leaf=1, min_samples_split=10),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3, min_samples_leaf=2, min_samples_split=6),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, colsample_bytree=0.8253102287905535, learning_rate=0.051799436321762704,
                             max_depth=9, n_estimators=286, subsample=0.9771414282231924, reg_alpha=0.5398410913016731,
                             reg_lambda=0.2030612247347694)
}

results = {}
class_names = np.unique(y_encoded)

# Training and evaluation with metrics
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)

    # A) weighted = global metric, balanced according to the size of each class
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # B) binary + pos_label = metric focused only in one class ("in"/"out")
    # prec = precision_score(y_test, y_pred, pos_label=class_names[1])
    # rec = recall_score(y_test, y_pred, pos_label=class_names[1])
    # f1 = f1_score(y_test, y_pred, pos_label=class_names[1])
    print(classification_report(y_test, y_pred, target_names=np.unique(y)))
    
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    # Calculate the confusion matrix with a defined class order
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0]).T

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['out', 'in'], yticklabels=['out', 'in'])   
    plt.suptitle(f"{name}", fontsize=13, fontweight="bold")
    plt.title('Confusion Matrix Heatmap', fontsize=11)
    plt.xlabel('True Labels', fontsize=9)
    plt.ylabel('Predicted Labels', fontsize=9)

    # Show the feature importance just for XGBoost and Random Forest
    if name in ["RandomForest", "XGBoost"]:
        importances = model.feature_importances_
        feature_names = df.columns[3:-1] # take the names instead of the values
        feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure()
        plt.suptitle(f"Feature Importance: {name}", fontsize=13, fontweight="bold")
        sns.barplot(x=feature_series.values, y=feature_series.index, palette="viridis")
        plt.xlabel("Importance", fontsize=9)
        plt.ylabel("Features", fontsize=9)
        plt.tight_layout()


# Show the results
print("Results in test set:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# DECISION TREE DIAGRAM
dt_model = models["DecisionTree"] # already trained

plt.figure(figsize=(12, 6))
plot_tree(dt_model, feature_names=df.columns[3:-1], class_names=le.classes_, filled=True)
plt.suptitle("Trained Decision Tree", fontsize=13, fontweight="bold")
plt.title("Optimized parameters", fontsize=11)

# ESTIMATED FEATURE IMPORTANCE (GaussianNB)
gnb_model = models["GaussianNB"] # already trained
means = pd.DataFrame(gnb_model.theta_, columns=df.columns[3:-1], index=le.classes_)
variances = pd.DataFrame(gnb_model.var_, columns=df.columns[3:-1], index=le.classes_)

mean_diff = abs(means.loc['in'] - means.loc['out']) # absolute difference between means
importance_estimate = mean_diff.sort_values(ascending=False) # order

print("\nEstimated variable importance in GaussianNB (|media_in - media_out|):")
print(importance_estimate)

plt.figure()
sns.barplot(x=importance_estimate.values, y=importance_estimate.index, palette="viridis")
plt.suptitle("Estimated Feature Importance: GaussianNB", fontsize=13, fontweight="bold")
plt.xlabel("Importance = |mean_in - mean_out|", fontsize=9)
plt.ylabel("Features", fontsize=9)
plt.tight_layout()


# CROSS-VALIDATION
print("\nResults cross-validation (accuracy):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

features_df = pd.DataFrame(X, columns=df.columns[3:-1])
features_df['class'] = le.inverse_transform(y_encoded)  # original labels: 'in', 'out'

means = features_df.groupby('class').mean().T # mean per class
print(means)

means.plot(kind='bar', colormap='Set3')
plt.suptitle("Mean of the features per class", fontsize=13, fontweight='bold')
plt.title('Both classes', fontsize=11)
plt.ylabel("Mean", fontsize=9)
plt.xticks(rotation=45, fontsize=9)
plt.grid(axis='y')
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()


plt.show(block=False)
input("Press [enter] key to close plots...")