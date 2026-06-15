import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(
    r"D:\Thiranex\Exploratory Data Analysis (EDA) Project\student_performance.csv"
)

# Display Data
print("FIRST 5 RECORDS")
print(df.head())

# Dataset Information
print("\nDATASET INFO")
print(df.info())

# Statistical Summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Correlation Matrix
print("\nCORRELATION MATRIX")
print(df.corr(numeric_only=True))

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["Final_Score"], kde=True)
plt.title("Distribution of Final Scores")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Hours_Studied",
    y="Final_Score",
    data=df
)
plt.title("Hours Studied vs Final Score")
plt.show()

# Attendance vs Final Score
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Attendance",
    y="Final_Score",
    data=df
)
plt.title("Attendance vs Final Score")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")