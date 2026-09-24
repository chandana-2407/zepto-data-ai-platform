import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")

# -----------------------------
# 1. Age Distribution
# -----------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()


# -----------------------------
# 2. Survival by Gender
# -----------------------------

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.show()


# -----------------------------
# 3. Survival by Passenger Class
# -----------------------------

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="class", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()


# -----------------------------
# 4. Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("EDA visualizations created successfully!")