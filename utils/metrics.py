import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, accuracy_score

def get_classification_metrics(y_true, y_pred, y_prob):
    """
    y_true: Ground-truth labels, shape=(N,)
    y_pred: Predicted labels, shape=(N,)
    y_prob: Predicted probabilities, shape=(N, C)
    """
    # Normalize in float64; AMP probabilities may not sum to exactly 1.
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = y_prob / (y_prob.sum(axis=1, keepdims=True) + 1e-12)
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Macro-F1'] = f1_score(y_true, y_pred, average='macro')
    metrics['Macro-AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics["Cohen's Kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics
