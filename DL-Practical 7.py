
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the dataset
print("Loading the Iris Dataset")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
print(f"Dataset Loaded Successfully with {X.shape[0]} samples and {X.shape[1]} features.")

# Step 3: Standardize the dataset (very important for PCA)
print("Standardizing the dataset")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data successfully standardized (mean=0, variance=1).")

# Step 4: Apply PCA
print("Applying PCA to reduce to 2 components")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Step 6: Visualize PCA result
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='Set2', s=80)
plt.title('PCA - Dimensionality Reduction of Iris Dataset', fontsize=14)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
plt.legend(labels=target_names, title="Iris Species")
plt.grid(True)
plt.show()

# Step 7: Analyze the explained variance
print("Explained Variance Analysis:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  • PC{i+1}: {ratio*100:.2f}% variance explained")
print(f"Total Variance Retained by first 2 components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# Step 8: Scree Plot - visualize how much variance each component explains
pca_full = PCA().fit(X_scaled)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(7,5))
plt.plot(range(1, len(explained_var)+1), explained_var, 'o-', color='purple', label='Individual Variance')
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'o--', color='green', label='Cumulative Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.title("Scree Plot (Variance Explained by Principal Components)")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Summary
print("Summary:")
print("PCA successfully reduced the dataset from 4D → 2D.")
print(f"The first two principal components capture {np.sum(pca.explained_variance_ratio_)*100:.2f}% of the total variance.")
print("The scatter plot shows clear clustering of Iris species, demonstrating PCA’s effectiveness for visualization.")
