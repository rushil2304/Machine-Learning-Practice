# 🤖 Machine Learning Algorithms Implementation 🚀

This repository showcases Python implementations and practical applications of a diverse set of machine learning algorithms. You'll find code covering both supervised and unsupervised learning paradigms, complemented by essential data preprocessing and model evaluation techniques.

## 📂 Repository Contents

The codebase is neatly organized into the following directories and files:

* **Supervised Learning 🧠**
    * `DesTree,LinReg,XGboost.py`: Implementation of Decision Tree, Linear Regression, and XGBoost algorithms tailored for regression tasks.
    * `KNN,SVM,LogReg.py`: Implementation of K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Logistic Regression algorithms designed for classification problems.
    * `Random Forest.py`: Implementation of the Random Forest ensemble learning method for classification.
    * `SVM.py`: A dedicated implementation of Support Vector Machines (SVM) for classification, featuring hyperparameter optimization.
* **Unsupervised Learning 💡**
    * `DBSCAN.py`: Implementation of the DBSCAN density-based clustering algorithm.
    * `DBSCAN,Kmeans.py`: Comparative implementation of DBSCAN, KMeans, and KMedoids clustering algorithms.
    * `Kmeans.py`: Implementation of the KMeans and KMedoids centroid-based clustering algorithms.
* **Notebook 📒**
    * `ML Practice(all codes).ipynb`: A comprehensive Jupyter Notebook containing a collection of various machine learning practices and code snippets.

## 📜 Detailed File Descriptions

Here's an in-depth look at each file within the repository:

* **`DBSCAN,Kmeans.py`**

    * **Algorithms:** Implements DBSCAN, KMeans, and KMedoids clustering techniques.
    * **Functionality:** Includes data loading, preprocessing steps, feature selection, data scaling, Principal Component Analysis (PCA) for visualization, and Silhouette Score for evaluation.
    * **Visualization:** Generates visualizations to illustrate the resulting clusters.
    * **Key Libraries:** Leverages the power of `pandas`, `numpy`, `seaborn`, `matplotlib`, and `sklearn`.

* **`DBSCAN.py`**

    * **Algorithm:** Focuses on the implementation of the DBSCAN clustering algorithm.
    * **Workflow:** Conducts Exploratory Data Analysis (EDA), performs data preprocessing, and visualizes the data.
    * **Output:** Calculates and presents the identified clusters through visualization.
    * **Key Libraries:** Utilizes `pandas`, `numpy`, `seaborn`, `matplotlib`, and `sklearn`.

* **`DesTree,LinReg,XGboost.py`**

    * **Algorithms:** Implements Decision Tree, Linear Regression, and XGBoost for regression analysis.
    * **Data Handling:** Loads and preprocesses stock market data.
    * **Process:** Executes EDA, splits data into training and testing sets, trains the regression models, and evaluates their performance using Root Mean Squared Error (RMSE) and R-squared (R2) score.
    * **Key Libraries:** Employs `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, and `xgboost`.

* **`Kmeans.py`**

    * **Algorithms:** Implements both KMeans and KMedoids clustering methods.
    * **Methodology:** Covers data loading, preprocessing, application of the Elbow method to determine the optimal number of clusters, and visualization of the clustering outcomes.
    * **Key Libraries:** Relies on `pandas`, `numpy`, `sklearn`, `sklearn_extra`, `seaborn`, and `matplotlib`.

* **`KNN,SVM,LogReg.py`**

    * **Algorithms:** Implements K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Logistic Regression for classification tasks.
    * **Pipeline:** Includes data loading, preprocessing, model training, hyperparameter tuning using GridSearchCV, and evaluation using accuracy metrics and classification reports.
    * **Key Libraries:** Utilizes the functionalities of `pandas` and `sklearn`.

* **`ML Practice(all codes).ipynb`**

    * **Content:** A Jupyter Notebook encompassing a variety of machine learning practices, potentially including data loading, EDA, model training, and evaluation across different algorithms. (Refer to the notebook for specific details and insights).
    * **Key Libraries:** Likely draws upon a broad spectrum of libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

* **`Random Forest.py`**

    * **Algorithm:** Implements the Random Forest algorithm for classification.
    * **Steps:** Involves data loading, preprocessing, model training, and evaluation using accuracy scores, classification reports, and confusion matrices.
    * **Key Libraries:** Leverages `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

* **`SVM.py`**

    * **Algorithm:** Provides a dedicated implementation of Support Vector Machines (SVM) for classification.
    * **Features:** Includes data loading, preprocessing, model training, hyperparameter optimization techniques, and performance evaluation.
    * **Key Libraries:** Makes use of `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

## ⚙️ Setup and Installation

Get started with these easy steps:

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies:**

    It's highly recommended to set up a virtual environment to manage project-specific dependencies.

    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment (depending on your OS)
    source venv/bin/activate   # Linux/macOS
    venv\Scripts\activate.bat  # Windows (for cmd)
    venv\Scripts\Activate.ps1  # Windows (for PowerShell)
    ```

    Install the necessary Python packages using pip. While it's best to check each script for its exact requirements, a general installation command is provided below:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn scikit-learn-extra xgboost jupyter
    ```

    * **Note:** Some scripts might have specific version requirements for certain libraries. If you encounter any issues, please refer to the comments within the respective script files.

3.  **Data Configuration:**

    * The provided scripts often assume that datasets are located at specific file paths (e.g., `C:\\Users\\Rushil\\Desktop\\training\\...`). To run the code successfully, you'll need to:
        * **Option 1:** Modify these hardcoded paths within the scripts to reflect the actual locations of your datasets on your system.
        * **Option 2:** Organize your data directory structure so that it aligns with the file paths specified in the scripts.

## 🚀 Usage Instructions

* **Running Python Scripts:**

    To execute an individual Python script, open your terminal or command prompt, navigate to the repository directory, and use the following command:

    ```bash
    python <script_name>.py
    ```

    For instance, to run the Decision Tree, Linear Regression, and XGBoost script:

    ```bash
    python DesTree,LinReg,XGboost.py
    ```

* **Working with the Jupyter Notebook:**

    To launch and interact with the Jupyter Notebook, navigate to the repository directory in your terminal and run:

    ```bash
    jupyter notebook ML\ Practice(all\ codes).ipynb
    ```

    This command will open the notebook in your default web browser. You can then execute the code cells sequentially to explore the machine learning practices.

## Performances of Models

1.  SVM

    Best SVM Params: {'C': 1, 'kernel': 'rbf'}
    SVM Model Results:
    Accuracy: 0.6593886462882096
    Classification Report:

    ```
                  precision    recall  f1-score   support

               4       0.00      0.00      0.00         6
               5       0.70      0.75      0.72        96
               6       0.62      0.70      0.65        99
               7       0.71      0.38      0.50        26
               8       0.00      0.00      0.00         2

        accuracy                           0.66       229
       macro avg       0.41      0.37      0.38       229
    weighted avg       0.64      0.66      0.64       229
    ```

2.  KNN

    Best KNN Params: {'n_neighbors': 11}
    KNN Model Results:
    Accuracy: 0.6026200873362445
    Classification Report:

    ```
                  precision    recall  f1-score   support

               4       0.00      0.00      0.00         6
               5       0.66      0.78      0.72        96
               6       0.58      0.57      0.57        99
               7       0.35      0.27      0.30        26
               8       0.00      0.00      0.00         2

        accuracy                           0.60       229
       macro avg       0.32      0.32      0.32       229
    weighted avg       0.57      0.60      0.58       229
    ```

3.  Logistic Regression

    Best Logistic Regression Params: {'C': 0.1}
    Logistic Regression Model Results:
    Accuracy: 0.6375545851528385
    Classification Report:

    ```
                  precision    recall  f1-score   support

               4       0.00      0.00      0.00         6
               5       0.68      0.73      0.70        96
               6       0.63      0.64      0.63        99
               7       0.52      0.50      0.51        26
               8       0.00      0.00      0.00         2

        accuracy                           0.64       229
       macro avg       0.37      0.37      0.37       229
    weighted avg       0.62      0.64      0.63       229
    ```

4.  Decision Tree Regressor

    ```
            Model      RMSE    R2 Score
    Decision Tree  6.647545   0.994674
    ```

5.  XGBoost

    ```
        Model      RMSE    R2 Score
    XGBoost  3.696082   0.998353
    ```

6.  Linear Regression

    ```
               Model      RMSE    R2 Score
    Linear Regression  1.709119   0.999648
    ```

7.  Random Forest

    Training Accuracy: 0.8539
    Testing Accuracy : 0.8436

    Classification Report (Test Data):

    ```
                  precision    recall  f1-score   support

               0       0.84      0.92      0.88       110
               1       0.85      0.72      0.78        69

        accuracy                           0.84       179
       macro avg       0.84      0.82      0.83       179
    weighted avg       0.84      0.84      0.84       179
    ```

8.  K-means

    KMeans Silhouette Score: 0.6023

9.  DBSCAN

    DBSCAN Silhouette Score: 0.8935

10. K-medoids

    KMedoids Silhouette Score: 0.3597

## 📌 Important Considerations

* **File Paths:** Carefully review and adjust the file paths within the scripts to match your local data storage locations. Incorrect paths will lead to errors.
* **Library Dependencies:** Ensure that you have installed all the required Python libraries. The `pip install` command provided is a general guideline; always refer to the specific import statements at the beginning of each script.
* **Output Interpretation:** The scripts typically print their output directly to your console. Additionally, some scripts generate plots and visualizations, so ensure you have a suitable environment configured to display them.
* **Computational Time:** Be aware that scripts utilizing hyperparameter tuning techniques like GridSearchCV can be computationally intensive and may take a significant amount of time to complete.
* **Virtual Environment Best Practices:** Utilizing virtual environments is strongly recommended to maintain a clean and isolated environment for this project and prevent potential conflicts with other Python projects on your system.