import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("titanic_cleaned.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

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


# ==========================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==========================================

X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 7. PREDICTION
# ==========================================

predictions = model.predict(
    X_test_scaled
)

probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]


# ==========================================
# 8. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ==========================================
# 9. FINAL RESULTS
# ==========================================

print("\n========================================")
print("FINAL END-TO-END PIPELINE RESULTS")
print("========================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC AUC  :", round(roc_auc, 4))

print("\nPipeline completed successfully!")