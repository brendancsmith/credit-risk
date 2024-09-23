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
    _, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
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
