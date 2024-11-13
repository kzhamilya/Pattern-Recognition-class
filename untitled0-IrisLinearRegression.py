import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import *
#WE SHOULD USE ONE-VS-ALL BTW, `CUZ LINEAR REGRESSION, POOKIE <3

data = load_iris()
features = data.data
labels = data.target
target_names = data.target_names

# Display the shape of the feature set and the labels
print("Shape of feature set:", features.shape)
print("Labels:", labels)

# Create and train the Perceptron classifier
perceptron = Perceptron(verbose=3)
perceptron.fit(features, labels)

# Display the model coefficients
print("Model coefficients:", perceptron.coef_)

# Test with a new input sample
sample = np.array([[2.5, 3.6, 1.9, 0.3]])
print("Test sample:", sample)
print("Shape of test sample:", sample.shape)

# Predict the class for the test sample
prediction = perceptron.predict(sample)
print("Prediction:", target_names[prediction])

# Visualization of the iris dataset classification
plt.title('IRIS Dataset Classification')
formatter = plt.FuncFormatter(lambda i, _: target_names[int(i)])
scatter = plt.scatter(features[:, 2], features[:, 3], c=labels, cmap='RdYlBu', edgecolor='k')
plt.colorbar(scatter, ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()