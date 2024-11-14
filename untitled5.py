# Step 1: Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Step 3: Preprocess the data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to improve model convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train the MLP model
# Initialize the MLP classifier with modified parameters
mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, activation='tanh', solver='sgd', learning_rate_init=0.01, random_state=42)

# Train the model on the training data
mlp.fit(X_train, y_train)

# Step 5: Evaluate the model
# Test the model on the test data
y_pred = mlp.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)