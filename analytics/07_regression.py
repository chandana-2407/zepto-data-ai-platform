import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("titanic_cleaned.csv")


# -----------------------------
# SELECT FEATURES
# -----------------------------

features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "survived"
]

X = df[features]
y = df["fare"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# SCALE FEATURES
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# LINEAR REGRESSION
# -----------------------------

model = LinearRegression()

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)


# -----------------------------
# EVALUATION
# -----------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# -----------------------------
# RESULTS
# -----------------------------

print("\n================================")
print("LINEAR REGRESSION RESULTS")
print("================================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)