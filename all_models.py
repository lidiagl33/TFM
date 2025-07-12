from sklearn.model_selection import train_test_split, StratifiedKFold
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
from scipy.spatial.distance import cdist


def keep_samples_far_from_opposite_centroid(df, label_col='class', threshold_margin=0.1):
    # Eliminate samples that are close to the centroid that bleongs to the opposite 
    # class, keeping the samples that are further from the opposite centroid than from 
    # the own centroid (for at least a certain margin)
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Calculate centroids of each class
    centroids = {
        label: X[y == label].mean().values.reshape(1, -1)
        for label in y.unique()
    }

    keep_indices = []

    for idx, row in X.iterrows():
        sample = row.values.reshape(1, -1)
        label = y.iloc[idx]
        other_label = [l for l in y.unique() if l != label][0]

        dist_own = cdist(sample, centroids[label])[0][0]
        dist_other = cdist(sample, centroids[other_label])[0][0]

        # It keeps if there is further from the opposite centroid (margin)
        if dist_other - dist_own >= threshold_margin:
            keep_indices.append(idx)

    return df.loc[keep_indices].sample(frac=1, random_state=42).reset_index(drop=True)


def get_confusion_matrix(y_test, y_pred, labels, name):
    # Calculate the confusion matrix with a defined class order
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels).T

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['out', 'in'], yticklabels=['out', 'in'])   
    plt.suptitle(f"{name}", fontsize=13, fontweight="bold")
    plt.title('Confusion Matrix Heatmap', fontsize=11)
    plt.xlabel('True Labels', fontsize=9)
    plt.ylabel('Predicted Labels', fontsize=9)


def get_feature_importance_rf_xgb(df, name, model):
    # Show the feature importance just for XGBoost and Random Forest
    if name in ["RandomForest", "XGBoost"]:
        importances = model.feature_importances_
        feature_names = df.columns[0:-1] # take the names instead of the values
        feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure()
        plt.suptitle(f"Feature Importance: {name}", fontsize=13, fontweight="bold")
        sns.barplot(x=feature_series.values, y=feature_series.index, palette="viridis")
        plt.xlabel("Importance", fontsize=9)
        plt.ylabel("Features", fontsize=9)
        plt.tight_layout()


def get_feature_importance_gnb(df, models, le):
    # ESTIMATED FEATURE IMPORTANCE (GaussianNB)
    gnb_model = models["GaussianNB"] # already trained
    means = pd.DataFrame(gnb_model.theta_, columns=df.columns[0:-1], index=le.classes_)
    # variances = pd.DataFrame(gnb_model.var_, columns=df.columns[0:-1], index=le.classes_)

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


def get_tree_diagram(models):
    # DECISION TREE DIAGRAM
    dt_model = models["DecisionTree"] # already trained

    plt.figure(figsize=(12, 6))
    plot_tree(dt_model, feature_names=df.columns[0:-1], class_names=le.classes_, filled=True)
    plt.suptitle("Trained Decision Tree", fontsize=13, fontweight="bold")
    plt.title("Optimized parameters", fontsize=11)


def get_feature_means(df, X, le, y_encoded):
    features_df = pd.DataFrame(X, columns=df.columns[0:-1])
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


def results_test_set(X, y_encoded, models):
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    results = {}
    # class_names = np.unique(y_encoded)

    # Training and evaluation with metrics
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)

        # macro = metric for each class and then claculate the mean
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print(classification_report(y_test, y_pred, target_names=np.unique(y)))
        
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        get_confusion_matrix(y_test, y_pred, [1,0], name)
        get_feature_importance_rf_xgb(df, name, model)
    
    return results


def results_cross_validation(y, y_cv_encoded, models, X_cv, y_cv):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    y_pred = np.empty(len(y_cv_encoded), dtype=object)
    confidence = np.empty(len(y_cv_encoded))

    for name, model in models.items():
        for train_idx, test_idx in cv.split(X_cv, y_cv_encoded):
            X_train = X_cv.iloc[train_idx].reset_index(drop=True)
            y_train = y_cv_encoded[train_idx]
            X_test = X_cv.iloc[test_idx].reset_index(drop=True)

            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)
            classes = model.classes_
            
            for i, idx in enumerate(test_idx):
                prob_vector = proba[i]
                pred_class = classes[np.argmax(prob_vector)]

                y_pred[idx] = pred_class
                # Additional analysis: CONFIDENCE
                if name == "RandomForest": # only the best model
                    confidence[idx] = np.max(prob_vector)
        
        # Show the results
        print(f"\n{name}:\n")
        y_pred_labels = le.inverse_transform(y_pred.astype(int))

        accuracy = accuracy_score(y_cv, y_pred_labels)
        precision_macro = precision_score(y_cv, y_pred_labels, labels=["out","in"], average=None).mean()
        precision_per_class = precision_score(y_cv, y_pred_labels, labels=["out","in"], average=None)
        recall_macro = recall_score(y_cv, y_pred_labels, labels=["out","in"], average=None).mean()
        recall_per_class = recall_score(y_cv, y_pred_labels, labels=["out","in"], average=None)
        f1_macro = f1_score(y_cv, y_pred_labels, labels=["out","in"], average=None).mean()
        f1_per_class = f1_score(y_cv, y_pred_labels, labels=["out","in"], average=None)

        print(f"Accuracy: {accuracy * 100:.2f}%")

        print(f"\nPrecision (macro average): {precision_macro * 100:.2f}%")
        print("Precision per class:")
        for label, score in zip(["out","in"], precision_per_class):
            print(f"  {label}: {score * 100:.2f}%")

        print(f"\nRecall (macro average): {recall_macro * 100:.2f}%")
        print("Recall per class:")
        for label, score in zip(["out","in"], recall_per_class):
            print(f"  {label}: {score * 100:.2f}%")

        print(f"\nF1-score (macro average): {f1_macro * 100:.2f}%")
        print("F1-score per class:")
        for label, score in zip(["out","in"], f1_per_class):
            print(f"  {label}: {score * 100:.2f}%")

        # Additional analysis: CONFIDENCE
        if name == "RandomForest": # only the best model
            correct = (y_pred == y_cv_encoded) # True/False depending on wether the prediciton = real value
            high_confidence_correct = np.sum((confidence >= 0.75) & (correct))
            high_confidence_incorrect = np.sum((confidence >= 0.75) & (~correct))
            total_correct = np.sum(correct)
            total_incorrect = len(y_pred) - total_correct 

            print(f"\nTotal correct predictions: {total_correct}")
            print(f"\t→ Of them, {high_confidence_correct} ({(high_confidence_correct / total_correct) * 100:.2f}%) had a confidence ≥ 75%")

            print(f"Total incorrect predictions: {total_incorrect}")
            print(f"\t→ Of them, {high_confidence_incorrect} ({(high_confidence_incorrect / total_incorrect) * 100:.2f}%) had a confidence ≥ 75%")

        get_confusion_matrix(y_cv, y_pred_labels, ["out","in"], name)


# Load data
df = pd.read_csv("dataset.csv")
print(df.shape)
print(df['class'].value_counts())

df = df.drop(columns=['id','attempt','name'])

# Remove similar inputs
df = keep_samples_far_from_opposite_centroid(df, label_col='class', threshold_margin=0.0001)
print(df['class'].value_counts())

# Data for test set
X = df.iloc[:, 0:-1].values # features
y = df.iloc[:, -1].values # class

# Data for cross-validation
X_cv = df.drop(columns=['class'])
X_cv.columns = X_cv.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True)
y_cv = df['class']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cv_encoded = le.fit_transform(y_cv)

# Models
models = {
    "GaussianNB": GaussianNB(var_smoothing=1e-09),
    "DecisionTree": DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=2, min_samples_leaf=1, min_samples_split=11),
    "RandomForest": RandomForestClassifier(max_features='sqrt', class_weight='balanced', random_state=42, n_estimators=300, max_depth=20, 
                                           min_samples_leaf=3, min_samples_split=10),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, colsample_bytree=0.6122000999756197, learning_rate=0.021204456624764327,
                             max_depth=8, n_estimators=64, subsample=0.8088973040219217, reg_alpha=0.3601906414112629,
                             reg_lambda=0.12706051265188478)
}

# ---------- TEST SET ----------
results = results_test_set(X, y_encoded, models)
# Show the results
print("Results in test set:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

get_tree_diagram(models)
get_feature_importance_gnb(df, models, le)
get_feature_means(df, X, le, y_encoded)

# ---------- CROSS-VALIDATION ----------
print("\nResults cross-validation:")
results_cross_validation(y, y_cv_encoded, models, X_cv, y_cv)

plt.show(block=False)
input("Press [enter] key to close plots...")