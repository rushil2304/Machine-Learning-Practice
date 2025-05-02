import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load the data
data = pd.read_csv(r"C:\Users\Rushil\Desktop\training\Supervised\datasets\wine_quality_classification.csv")

# Display basic structure and info
print("\n" + "="*60)
print(data.head())
print("\n" + "="*60)
print(data.tail())
print("\n" + "="*60)
print(data.describe())
print("\n" + "="*60)
print(data.isnull().sum())

# Encode quality labels
le = LabelEncoder()
data['quality_encoded'] = le.fit_transform(data['quality_label'])

# Check class distribution
print("\nClass distribution:")
print(data['quality_encoded'].value_counts())

# Correlation matrix heatmap
numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Histograms of numeric columns
for col in numeric_data:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, color='skyblue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Distribution of quality labels
plt.figure(figsize=(6, 4))
sns.countplot(x='quality_label', data=data, palette='pastel')
plt.title('Distribution of Wine Quality Labels')
plt.xlabel('Quality Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Combine numeric features and the encoded target column for pairplot
numeric_data_with_target = numeric_data.copy()
numeric_data_with_target['quality_encoded'] = data['quality_encoded']

# Pairplot
sns.pairplot(numeric_data_with_target, hue='quality_encoded', corner=True)
plt.suptitle('Pair Plot of Numeric Features and Encoded Quality', y=1.02)
plt.show()

# Features and target variable
X = data[numeric_data.columns]
Y = data['quality_encoded']

# Split the data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance by using class_weight='balanced'
svm = SVC(kernel="linear", gamma=0.6, C=1.0, class_weight='balanced')

# Train the model on scaled training data
svm.fit(X_train_scaled, y_train)

# Predict on scaled test data
y_pred = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Detailed Classification Report
print('Classification Report:\n', classification_report(y_test, y_pred, target_names=le.classes_))

# Optional: Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(SVC(kernel="linear"), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best Parameters
print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)

# Train the model again with the best parameters
svm_best = grid_search.best_estimator_

# Predict and evaluate with the best model
y_pred_best = svm_best.predict(X_test_scaled)
print("\nTest Accuracy with Best Model:", accuracy_score(y_test, y_pred_best))

# Detailed Classification Report for the best model
print('Classification Report with Best Model:\n', classification_report(y_test, y_pred_best, target_names=le.classes_))
