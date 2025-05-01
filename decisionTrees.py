#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("heart.csv")

# Check for missing values and basic info
print(df.info())
print(df.isnull().sum())

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and visualize Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

plt.figure(figsize=(18,8))
plot_tree(
    dtree,
    feature_names=list(X.columns),  # Fixed line
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    max_depth=3
)
plt.title("Decision Tree Visualization")
plt.show()

# Analyze overfitting and control tree depth
dtree_controlled = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree_controlled.fit(X_train, y_train)

# Train Random Forest and compare accuracy
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

models = {
    "Decision Tree": dtree,
    "Controlled Tree": dtree_controlled,
    "Random Forest": rf
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.3f}")

# Interpret feature importances
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Evaluate using cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\nRandom Forest 5-fold CV scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

