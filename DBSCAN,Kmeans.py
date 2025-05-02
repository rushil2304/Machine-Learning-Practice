'''DBSCAN Silhouette Score: 0.8935
   KMeans Silhouette Score: 0.6023
   KMedoids Silhouette Score: 0.3597
'''
"""
Aproach:
1. Load the datasets and merge them.
2. Perform exploratory data analysis (EDA) to understand the data.
3. Visualize the correlation between numerical features using a heatmap.
4. Generate a pairplot to visualize relationships between numerical features.
5. Select features for clustering and handle skewness in the data.
6. Standardize the features using StandardScaler.
7. Apply PCA to reduce dimensionality for visualization.
8. Implement DBSCAN clustering and evaluate the results using silhouette score.
9. Visualize the DBSCAN clustering results in a 2D PCA view.
10. Visualize the K-Medoids clustering results in a 2D PCA view.
11. Implement KMeans clustering and evaluate the results using silhouette score."""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids  

# Load datasets
df_orders = pd.read_csv(rf'C:\Users\Rushil\Desktop\training\Unsupervised\datasets\List of Orders.csv')
df_order_details = pd.read_csv(rf'C:\Users\Rushil\Desktop\training\Unsupervised\datasets\Order Details.csv')

# Merge datasets
df = pd.merge(df_orders, df_order_details, on='Order ID')

# EDA
print("\n" + "="*60)
print(df.head())
print("\n" + "="*60)
print(df.isnull().sum())
print("\n" + "="*60)
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Numerical Features')
plt.tight_layout()
plt.show()

# Generates Pairplot
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Pairwise Relationships Between Numerical Columns', y=1.02)
    plt.show()

# Features for clustering
features = ['Quantity', 'Profit', 'Amount']
df_cluster = df[features].dropna().copy()

# Handle left skewness: reflect + log transform
for col in ['Profit', 'Quantity', 'Amount']:
    max_val = df_cluster[col].max()
    df_cluster[col] = np.log1p(max_val + 1 - df_cluster[col])

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def apply_dbscan(X_scaled, X_pca):
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    unique_clusters = np.unique(dbscan_labels)
    cluster_map = {old_label: idx for idx, old_label in enumerate(unique_clusters)}
    adjusted_clusters = np.array([cluster_map[label] for label in dbscan_labels])

    df['DBSCAN_Cluster'] = -1
    df.loc[df_cluster.index, 'DBSCAN_Cluster'] = adjusted_clusters

    if len(set(adjusted_clusters)) > 1:
        score = silhouette_score(X_scaled, adjusted_clusters)
        print(f"\nDBSCAN Silhouette Score: {score:.4f}")
    else:
        print("\nDBSCAN Silhouette Score: Not applicable (only one cluster or noise)")

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set1", len(np.unique(adjusted_clusters)))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=adjusted_clusters, palette=palette)
    plt.title('DBSCAN Clustering Results (2D PCA View)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n DBSCAN Cluster Distribution ")
    print(df['DBSCAN_Cluster'].value_counts())

def apply_kmeans(X_scaled, X_pca, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    df['KMeans_Cluster'] = -1
    df.loc[df_cluster.index, 'KMeans_Cluster'] = kmeans_labels

    score = silhouette_score(X_scaled, kmeans_labels)
    print(f"\nKMeans Silhouette Score: {score:.4f}")

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_clusters)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette=palette)
    plt.title(f'KMeans Clustering Results (k={n_clusters}, 2D PCA View)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n KMeans Cluster Distribution ")
    print(df['KMeans_Cluster'].value_counts())

def apply_kmedoids(X_scaled, X_pca, n_clusters=3, random_state=42):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state)
    kmedoids_labels = kmedoids.fit_predict(X_scaled)

    df['KMedoids_Cluster'] = -1
    df.loc[df_cluster.index, 'KMedoids_Cluster'] = kmedoids_labels

    score = silhouette_score(X_scaled, kmedoids_labels)
    print(f"\nKMedoids Silhouette Score: {score:.4f}")

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set3", n_clusters)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmedoids_labels, palette=palette)
    plt.title(f'KMedoids Clustering Results (k={n_clusters}, 2D PCA View)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n KMedoids Cluster Distribution")
    print(df['KMedoids_Cluster'].value_counts())

# Apply clustering algorithms
apply_dbscan(X_scaled, X_pca)
apply_kmeans(X_scaled, X_pca, n_clusters=3)
apply_kmedoids(X_scaled, X_pca, n_clusters=3)
