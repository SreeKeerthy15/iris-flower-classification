# Iris Flower Classification Project
# Author: Your Name
# Description: Classifies iris flowers into Setosa, Versicolor, or Virginica

# ==== 1. Import Libraries ====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ==== 2. Load Dataset ====
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 5 rows of the dataset:")
print(df.head())

# ==== 3. Data Visualization ====
sns.pairplot(df, hue="species")
plt.savefig("pairplot.png")  # save the plot
plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()

# ==== 4. Train-Test Split ====
X = df.iloc[:, :-1]  # features
y = df['species']    # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 5. Train Model ====
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ==== 6. Predictions ====
y_pred = model.predict(X_test)

# ==== 7. Evaluation ====
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==== 8. Save Model ====
joblib.dump(model, "iris_model.pkl")
print("\nModel saved as iris_model.pkl")
