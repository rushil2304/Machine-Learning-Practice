import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
file_path = r"C:\Users\Rushil\Desktop\training\Supervised\datasets\WineQT.csv"
data = pd.read_csv(file_path)

# Inspect the data
print("\n" + "="*60)
print(data.head())
print("\n" + "="*60)
print(data.tail())
print("\n" + "="*60)
print(data.describe())
print("\n" + "="*60)
print(data.isnull().sum())



#  Plot histograms for skewness
fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(30, 60))
for p, ax in zip(data.columns, axs.flat):
    skew = data[p].skew()
    sns.histplot(data[p], kde=False, label=f'skew = {skew:.3f}', ax=ax)
    ax.legend(loc='best', fontsize=25)
    ax.set_xlabel(p, fontsize=25)
plt.show()

#Correlation Heatmap
corr = data.corr()
fig = plt.figure(figsize=(16, 16))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Data Preprocessing
def preprocess_data(data):
    X = data.drop(columns=['quality'])  
    y = data['quality']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(data)

# Improved confusion matrix visualization
def plot_confusion_matrix(cm, model_name):
    """Plot a confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.show()

# Model Training and Evaluation
def train_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    """Train and evaluate KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print("KNN Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "KNN")

def train_svm(X_train, X_test, y_train, y_test, kernel='linear'):
    """Train and evaluate Support Vector Machine classifier."""
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    print("SVM Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "SVM")

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression classifier."""
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    print("Logistic Regression Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Logistic Regression")

# Hyperparameter Tuning to improve model accuracy

# GridSearch for KNN
def tune_knn(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best KNN Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

# GridSearch for SVM
def tune_svm(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best SVM Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

# GridSearch for Logistic Regression
def tune_logistic_regression(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best Logistic Regression Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Tune and evaluate models
best_knn = tune_knn(X_train, y_train)
train_knn(X_train, X_test, y_train, y_test, n_neighbors=best_knn.get_params()['n_neighbors'])

best_svm = tune_svm(X_train, y_train)
train_svm(X_train, X_test, y_train, y_test, kernel=best_svm.get_params()['kernel'])

best_log_reg = tune_logistic_regression(X_train, y_train)
train_logistic_regression(X_train, X_test, y_train, y_test)
