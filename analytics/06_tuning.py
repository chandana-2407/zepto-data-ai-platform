import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("titanic_cleaned.csv")

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]


# -----------------------------
# ENCODE CATEGORICAL VARIABLES
# -----------------------------

X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# DECISION TREE - HYPERPARAMETER TUNING
# ==========================================

dt = DecisionTreeClassifier(random_state=42)

dt_params = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    dt,
    dt_params,
    cv=5,
    scoring="accuracy"
)

dt_grid.fit(X_train, y_train)

print("\n================================")
print("DECISION TREE TUNING")
print("================================")

print("Best Parameters:")
print(dt_grid.best_params_)

print("Best Cross-Validation Score:")
print(dt_grid.best_score_)


# ==========================================
# RANDOM FOREST - HYPERPARAMETER TUNING
# ==========================================

rf = RandomForestClassifier(
    random_state=42
)

rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring="accuracy"
)

rf_grid.fit(X_train, y_train)

print("\n================================")
print("RANDOM FOREST TUNING")
print("================================")

print("Best Parameters:")
print(rf_grid.best_params_)

print("Best Cross-Validation Score:")
print(rf_grid.best_score_)


# ==========================================
# CROSS VALIDATION
# ==========================================

best_rf = rf_grid.best_estimator_

cv_scores = cross_val_score(
    best_rf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n================================")
print("CROSS VALIDATION")
print("================================")

print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())