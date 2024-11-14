import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
data = pd.read_csv("D:/Жамиля0/UNIVERSITY/Pattern Recognition/diabetes.csv")

# Step 2: Preprocess the data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Set up the SVM classifier
svm = SVC()

# Step 4: Define the hyperparameter grid
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],  # Different kernel types
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

# Step 5: Perform Grid Search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Step 6: Get the best parameters and evaluate the model
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Step 7: Evaluate the model on the test set
y_pred = grid_search.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))