from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # True labels

# Convert to a DataFrame for better visualization
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y

# Visualize pairwise relationships
sns.pairplot(iris_df, hue="species", palette="viridis")
plt.show()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a Gaussian Mixture Model with modified parameters
gmm = GaussianMixture(n_components=4,  # Trying 4 clusters instead of 3
                      covariance_type='diag',  # Diagonal covariance for simpler cluster shapes
                      n_init=10,  # Increase the number of initializations
                      init_params='random',  # Random initialization
                      random_state=0)
gmm.fit(X_scaled)

# Predict clusters
predicted_clusters = gmm.predict(X_scaled)

# Plot predicted clusters against true labels
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=predicted_clusters, cmap='viridis', marker='o', label='Predicted Clusters')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='plasma', marker='x', alpha=0.6, label='True Labels')
plt.xlabel("Feature 1 (standardized)")
plt.ylabel("Feature 2 (standardized)")
plt.title("GMM Clustering vs True Labels on Iris Dataset")
plt.legend(["Predicted Clusters", "True Labels"], loc="upper right")
plt.show()

# Evaluate the model using Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(y, predicted_clusters)
print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")

# Optional: Fine-tune the model using BIC with new parameters
components = np.arange(1, 7)
bics = []
for n in components:
    gmm = GaussianMixture(n_components=n, covariance_type='diag', init_params='random', n_init=10, random_state=0)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))

# Plot BIC to find the optimal number of components
plt.figure(figsize=(8, 6))
plt.plot(components, bics, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.title("BIC for Different Number of Components with Modified Parameters")
plt.show()
