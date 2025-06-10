from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import pandas as pd

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

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# 1) GaussianNB (GridSearchCV)
print("\nGridSearchCV for GaussianNB...")
param_grid_gnb = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
gs_gnb = GridSearchCV(GaussianNB(), param_grid=param_grid_gnb, cv=cv, scoring='accuracy', n_jobs=-1)
gs_gnb.fit(X_train, y_train)
print("Best parameters GNB:", gs_gnb.best_params_)


# 2) DecisionTree (Randomized + Grid)
print("\nRandomizedSearchCV for DecisionTree...")
param_dist_dt = {
    'max_depth': randint(1, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}
rs_dt = RandomizedSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_distributions=param_dist_dt,
                            n_iter=50, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
rs_dt.fit(X_train, y_train)
print("Best parameters DT (Randomized):", rs_dt.best_params_)

print("\nGridSearchCV for DecisionTree...")
param_grid_dt = {
    'max_depth': [1, 2, 3, 4],
    'min_samples_split': [11, 13, 15],
    'min_samples_leaf': [1, 2, 3]
}
gs_dt = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid=param_grid_dt, cv=cv,
                      scoring='accuracy', n_jobs=-1)
gs_dt.fit(X_train, y_train)
print("Best parameters DT (Grid):", gs_dt.best_params_)


# 3) RandomForest (Randomized + Grid)
print("\nRandomizedSearchCV for RandomForest...")
param_dist_rf = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(1, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}
rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist_rf,
                           n_iter=50, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
rs_rf.fit(X_train, y_train)
print("Best parameters RF (Randomized):", rs_rf.best_params_)

print("\nGridSearchCV for RandomForest...")
param_grid_rf = {
    'n_estimators': [50, 70, 90],
    'max_depth': [1, 3, 5],
    'min_samples_split': [6, 8, 10],
    'min_samples_leaf': [2, 5, 8]
}
gs_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=cv,
                     scoring='accuracy', n_jobs=-1)
gs_rf.fit(X_train, y_train)
print("Best parameters RF (Grid):", gs_rf.best_params_)


# 4) XGBoost (RandomizedSearchCV)
print("\nRandomizedSearchCV for XGBoost...")
param_dist_xgb = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}
rs_xgb = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist_xgb,
    n_iter=50,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
rs_xgb.fit(X_train, y_train)
print("Best parameters XGB:", rs_xgb.best_params_)

# The final models are in gs_gnb.best_estimator_, gs_dt.best_estimator_, gs_rf.best_estimator_, rs_xgb.best_estimator_
