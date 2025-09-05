import numpy as np
from collections import Counter
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def count_mismatch_by_class(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mismatches = y_true != y_pred
    mismatch_classes = y_true[mismatches]
    
    return dict(Counter(mismatch_classes))