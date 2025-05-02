""""
OUTPUT:
            Model     RMSE    R2 Score
    Decision Tree  6.647545   0.994674
          XGBoost  3.696082   0.998353
Linear Regression  1.709119   0.999648

Approach:
1. Loads stock market data from a CSV file.
2. Preprocesses the data by converting the 'Date' column to datetime format and extracting day, month, and year.
3. Performs exploratory data analysis (EDA) to visualize the closing price over time and the correlation between features.
4. Splits the data into training and testing sets.
5. Defines a function to run different regression models (Decision Tree, XGBoost, Linear Regression) and evaluate their performance using RMSE and R2 score.
6. Trains and evaluates the models on the training and testing sets.
7. Displays the results in a DataFrame with model names, RMSE, and R2 scores.
8. Prints the results DataFrame without the index.    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df

# EDA function
def perform_eda(df):
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.isnull().sum())
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Closing Price Over Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

# Train/test split
def split_data(df):
    features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']
    target = 'Close'
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# Model runners
def run_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    return [name, rmse, r2]

# Main execution
if __name__ == "__main__":
    filepath = rf'C:\Users\Rushil\Desktop\training\Supervised\datasets\stocks.csv'  
    df = load_data(filepath)
    perform_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)

    results = []
    results.append(run_model("Decision Tree", DecisionTreeRegressor(), X_train, X_test, y_train, y_test))
    results.append(run_model("XGBoost", XGBRegressor(), X_train, X_test, y_train, y_test))
    results.append(run_model("Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test))

    results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'R2 Score'])
    print(results_df.to_string(index=False))  
