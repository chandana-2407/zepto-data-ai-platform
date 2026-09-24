import pandas as pd

# Load the dataset saved from Task 1
df = pd.read_csv("titanic.csv")

print("===== MISSING VALUES BEFORE CLEANING =====")
print(df.isnull().sum())

# Fill Age with median
df["age"] = df["age"].fillna(df["age"].median())

# Fill Embarked with mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Drop columns with too many missing values
df = df.drop(columns=["deck"])

print("\n===== MISSING VALUES AFTER CLEANING =====")
print(df.isnull().sum())

print("\n===== DATASET SHAPE AFTER CLEANING =====")
print(df.shape)

# Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")