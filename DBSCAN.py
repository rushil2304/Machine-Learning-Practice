import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load the datasets
df_orders = pd.read_csv(rf'C:\Users\Rushil\Desktop\training\Unsupervised\datasets\List of Orders.csv')
df_order_details = pd.read_csv(rf'C:\Users\Rushil\Desktop\training\Unsupervised\datasets\Order Details.csv')

# Merge both datasets using 'Order ID' so we can access all relevant columns
df = pd.merge(df_orders, df_order_details, on='Order ID')

# Perform basic exploratory data analysis (EDA)

print("\n--- Preview of the Data ---")
print(df.head())

print("\n--- Missing Values in Each Column ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

# Show correlation heatmap between numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Numerical Features')
plt.tight_layout()
plt.show()

# Visualize pairwise relationships for numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Pairwise Relationships Between Numerical Columns', y=1.02)
    plt.show()

# Choose the features we want to cluster on and drop rows with missing values
features = ['Quantity', 'Profit','Amount']
X = df[features].dropna()  # Drop rows with missing values in selected columns

# Scale the data so all features contribute equally to distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions using PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply DBSCAN clustering algorithm
dbscan = DBSCAN(eps=1.7, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)


# Find the unique cluster labels and map them to start from 0
unique_clusters = np.unique(clusters)
cluster_map = {old_label: idx for idx, old_label in enumerate(unique_clusters)}
adjusted_clusters = np.array([cluster_map[label] for label in clusters])

# Add adjusted cluster labels to the DataFrame
df['Cluster'] = adjusted_clusters

# Step 11: Evaluate how well the clustering performed
if len(set(adjusted_clusters)) > 1:
    score = silhouette_score(X_scaled, adjusted_clusters)
    print(f"Silhouette Score: {score:.4f}")
else:
    print("Silhouette Score: Not applicable (only one cluster or mostly noise)")

# Step 12: Visualize clusters in PCA-reduced 2D space
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

# Step 13: Print cluster distribution
print("\n--- Cluster Distribution ---")
print(df['Cluster'].value_counts())
