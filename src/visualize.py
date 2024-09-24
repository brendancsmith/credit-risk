from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def correlation_matrix(corr):
    # Plot heatmap
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    _, _ = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def correlations(corr):
    # Plot correlation values
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlations with Loan Status')
    plt.show()

def class_dist(df):
    # Plot class distribution
    sns.countplot(x='loan_status', data=df)
    plt.title('Class Distribution')
    plt.show()

def roc_curve(y_test, y_proba, roc_auc=None):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    label = f"AUC = {roc_auc:.4f}" if roc_auc is not None else None
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_rfe(rfe, n_features_to_select=None):
    plt.figure()
    plt.title("RFE with cross-validation")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(rfe.cv_results_["n_features"], rfe.cv_results_["mean_test_score"])
    cv_score = rfe.cv_results_["mean_test_score"][rfe.n_features_ - 1]
    plt.scatter(rfe.n_features_, cv_score, marker="|", s=100)
    plt.show()
