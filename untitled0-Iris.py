import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

iris = load_iris()
#feature selection: the sepal length (first column of the dataset)
#choosing more different features allows us to classify the flower more accurately
X = iris.data[:, 0] 
y = (iris.target == 2).astype(int)
#threshold classification: if the sepal length is >5 it's classified as iris-virginica
threshold = 5.0
y_pred = (X > threshold).astype(int)
#train/test split: 30% of the data is used for testing, 70% is used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#accuracy calculation
accuracy = accuracy_score(y_test, (X_test > threshold).astype(int))
print(f'Accuracy at threshold {threshold}: {accuracy:.2f}')

y_scores = X_test  
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()