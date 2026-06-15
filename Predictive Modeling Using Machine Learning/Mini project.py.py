# Predictive Modeling Using Machine Learning

# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# LOAD DATASET

df = pd.read_csv(r"D:\Thiranex\Predictive Modeling Using Machine Learning\student_data.csv")

print("FIRST 5 ROWS")
print(df.head())


# DATA INFORMATION

print("\nDATA INFORMATION")
print(df.info())


print("\nMISSING VALUES")
print(df.isnull().sum())


# DATA CLEANING

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)


print("\nAfter Cleaning")
print(df)


# SPLIT INPUT AND OUTPUT

X = df.drop("Result", axis=1)

y = df["Result"]


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 1. LOGISTIC REGRESSION
# =========================

log_model = LogisticRegression()

log_model.fit(
    X_train,
    y_train
)

log_prediction = log_model.predict(
    X_test
)


log_accuracy = accuracy_score(
    y_test,
    log_prediction
)


print("\nLogistic Regression Accuracy:")
print(log_accuracy)



# =========================
# 2. DECISION TREE
# =========================

tree_model = DecisionTreeClassifier()

tree_model.fit(
    X_train,
    y_train
)


tree_prediction = tree_model.predict(
    X_test
)


tree_accuracy = accuracy_score(
    y_test,
    tree_prediction
)


print("\nDecision Tree Accuracy:")
print(tree_accuracy)



# =========================
# 3. RANDOM FOREST
# =========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


rf_prediction = rf_model.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)


print("\nRandom Forest Accuracy:")
print(rf_accuracy)



# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    rf_prediction
)


plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)


plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



# CLASSIFICATION REPORT

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_prediction
    )
)



# =========================
# MODEL COMPARISON
# =========================

models = {

    "Logistic Regression":
    log_accuracy,

    "Decision Tree":
    tree_accuracy,

    "Random Forest":
    rf_accuracy

}


plt.figure(figsize=(8,5))


sns.barplot(
    x=list(models.keys()),
    y=list(models.values())
)


plt.title(
    "Model Accuracy Comparison"
)


plt.ylabel(
    "Accuracy"
)


plt.xticks(
    rotation=45
)


plt.show()



# =========================
# FEATURE IMPORTANCE
# =========================


importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)


importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)


plt.title(
    "Feature Importance"
)


plt.xlabel(
    "Importance"
)


plt.show()


print("\nPROJECT COMPLETED SUCCESSFULLY")