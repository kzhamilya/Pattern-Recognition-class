from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target (species)

# Scale the features
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Check the number of components selected
n_components = X_pca.shape[1]

# Plot based on the number of components
if n_components == 2:
    # 2D Plot if there are only two components
    plt.figure(figsize=(8, 6))
    for i, target_name in zip([0, 1, 2], iris.target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name, alpha=0.7)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("PCA of Iris Dataset (2D)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
elif n_components == 3:
    # 3D Plot if there are three components
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i, target_name, color in zip([0, 1, 2], iris.target_names, colors):
        ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=target_name, c=color, alpha=0.7)
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_zlabel("Third Principal Component")
    plt.title("PCA of Iris Dataset (3D)")
    plt.legend()
    plt.show()
else:
    print(f"PCA selected {n_components} components, which is outside the expected 2 or 3 range.")
