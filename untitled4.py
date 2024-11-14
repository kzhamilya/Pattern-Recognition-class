# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binarize labels for ROC analysis (required for multi-class)
y_binarized = label_binarize(y, classes=[0, 1, 2])
n_classes = y_binarized.shape[1]

# Initializing classifiers
lda = LinearDiscriminantAnalysis()
log_reg = LogisticRegression(max_iter=200)
nb = GaussianNB()

# Training each model and predicting
lda.fit(X, y)
log_reg.fit(X, y)
nb.fit(X, y)

y_pred_lda = lda.predict(X)
y_pred_log = log_reg.predict(X)
y_pred_nb = nb.predict(X)

# Evaluating each classifier
accuracy_lda = accuracy_score(y, y_pred_lda)
accuracy_log = accuracy_score(y, y_pred_log)
accuracy_nb = accuracy_score(y, y_pred_nb)

f1_lda = f1_score(y, y_pred_lda, average="weighted")
f1_log = f1_score(y, y_pred_log, average="weighted")
f1_nb = f1_score(y, y_pred_nb, average="weighted")

# ROC analysis (using OneVsRest approach for multiclass classifiers)
lda_proba = lda.predict_proba(X)
log_reg_proba = log_reg.predict_proba(X)
nb_proba = nb.predict_proba(X)

# Calculate ROC and AUC for each class and plot them
fpr_lda, tpr_lda, roc_auc_lda = {}, {}, {}
fpr_log, tpr_log, roc_auc_log = {}, {}, {}
fpr_nb, tpr_nb, roc_auc_nb = {}, {}, {}

for i in range(n_classes):
    # LDA ROC
    fpr_lda[i], tpr_lda[i], _ = roc_curve(y_binarized[:, i], lda_proba[:, i])
    roc_auc_lda[i] = auc(fpr_lda[i], tpr_lda[i])
    # Logistic Regression ROC
    fpr_log[i], tpr_log[i], _ = roc_curve(y_binarized[:, i], log_reg_proba[:, i])
    roc_auc_log[i] = auc(fpr_log[i], tpr_log[i])
    # Naive Bayes ROC
    fpr_nb[i], tpr_nb[i], _ = roc_curve(y_binarized[:, i], nb_proba[:, i])
    roc_auc_nb[i] = auc(fpr_nb[i], tpr_nb[i])

# Macro-average AUC for each classifier
auc_lda = roc_auc_score(y_binarized, lda_proba, average="macro")
auc_log = roc_auc_score(y_binarized, log_reg_proba, average="macro")
auc_nb = roc_auc_score(y_binarized, nb_proba, average="macro")

# Results Summary
results = {
    "LDA": {"Accuracy": accuracy_lda, "F1 Score": f1_lda, "AUC": auc_lda},
    "Logistic Regression": {"Accuracy": accuracy_log, "F1 Score": f1_log, "AUC": auc_log},
    "Naive Bayes": {"Accuracy": accuracy_nb, "F1 Score": f1_nb, "AUC": auc_nb},
}

print(results)
