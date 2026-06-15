import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv(
    r"D:\Thiranex\Real-world Data Project (Finance, Health, or Retail)\retail_sales.csv"
)


print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# Correlation
print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))

# Sales Trend
plt.figure(figsize=(8,5))
plt.plot(df["Month"], df["Sales"], marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# Prediction Model
X = df[["Advertising","Customers"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X,y)

predicted_sales = model.predict(X)

df["Predicted_Sales"] = predicted_sales

print("\nPredicted Sales")
print(df[["Month","Sales","Predicted_Sales"]])

# Actual vs Predicted
plt.figure(figsize=(8,5))
plt.plot(df["Month"], df["Sales"], marker="o", label="Actual")
plt.plot(df["Month"], df["Predicted_Sales"], marker="s", label="Predicted")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY")