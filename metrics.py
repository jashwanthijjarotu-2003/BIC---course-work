"""Classification metrics from scratch: accuracy, precision, recall, F1.

Binary case, positive class = 1 (a forged note).
"""
import numpy as np


def binary_metrics(y_true, y_pred):
    """Return accuracy, precision, recall and F1 for binary 0/1 labels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))   # predicted 1, actually 1
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))   # predicted 0, actually 0
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))   # predicted 1, actually 0
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))   # predicted 0, actually 1
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0  # of predicted-positives, how many correct
    recall = tp / (tp + fn) if (tp + fn) else 0.0     # of actual-positives, how many caught
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)
