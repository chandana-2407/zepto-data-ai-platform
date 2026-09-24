import pandas as pd
import seaborn as sns

# Load Titanic dataset — only once
df = sns.load_dataset("titanic")

print("===== DATASET INFO =====")
print(df.info())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== MISSING VALUES =====")
missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage[missing_percentage > 0])

# Save cleaned/raw loaded dataset as offline fallback
df.to_csv("titanic.csv", index=False)

print("\nTitanic dataset saved successfully as titanic.csv")