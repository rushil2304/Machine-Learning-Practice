import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load and Preprocess Data
def load_data(train_path, test_path):
    data = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    return data, data_test

def preprocess_data(data):
    # Drop missing values for simplicity (you can handle this differently if needed)
    data = data.dropna()
    
    # Select numeric features for clustering
    df_numeric = data.select_dtypes(exclude=[object])
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    return df_numeric, df_scaled

# Elbow Method to find optimal number of clusters for KMeans
def plot_elbow_method(df_scaled):
    inertias = []
    for k in range(1, 11):  # Try cluster numbers from 1 to 10
        kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot the elbow method
    plt.figure(figsize=(8,6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

# Apply K-Means and evaluate silhouette score
def apply_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=42)
    kmeans.fit(df_scaled)
    
    # Evaluate the clustering using silhouette score
    silhouette = silhouette_score(df_scaled, kmeans.labels_)
    print(f"K-Means Silhouette Score (k={n_clusters}): {silhouette}")
    
    return kmeans

# Apply K-Medoids and evaluate silhouette score
def apply_kmedoids(df_scaled, n_clusters):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42)
    kmedoids.fit(df_scaled)
    
    silhouette = silhouette_score(df_scaled, kmedoids.labels_)
    print(f"K-Medoids Silhouette Score (k={n_clusters}): {silhouette}")
    
    return kmedoids

# Visualization: Plot 2D PCA of the clusters
def plot_pca(df_scaled, model, method='K-Means'):
    # Reduce to 2 dimensions for better visualization
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    
    # Create a DataFrame with PCA results and cluster labels
    df_pca_df = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
    df_pca_df['Cluster'] = model.labels_
    
    # 2D Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, edgecolor='black')
    plt.title(f'{method} Clusters (2D PCA-reduced)')
    plt.show()

# Heatmap of feature correlations
def plot_heatmap(df_numeric):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Main Code Execution
def main():
    train_path = r'C:\Users\Rushil\Desktop\training\Unsupervised\datasets\Train.csv'
    test_path = r'C:\Users\Rushil\Desktop\training\Unsupervised\datasets\Test.csv'
    
    # Load Data
    data, data_test = load_data(train_path, test_path)
    
    # Preprocess Data
    df_numeric, df_scaled = preprocess_data(data)
    
    # Elbow Method to find optimal k (number of clusters)
    plot_elbow_method(df_scaled)
    
    # Choose the optimal k from the Elbow method (based on the plot)
    optimal_k = 4  # Changed to 4 clusters
    
    # Apply K-Means
    kmeans = apply_kmeans(df_scaled, n_clusters=optimal_k)
    
    # Apply K-Medoids
    kmedoids = apply_kmedoids(df_scaled, n_clusters=optimal_k)
    
    # Plot Results
    plot_pca(df_scaled, kmeans, method='K-Means')
    plot_pca(df_scaled, kmedoids, method='K-Medoids')
    plot_heatmap(df_numeric)

if __name__ == "__main__":
    main()
