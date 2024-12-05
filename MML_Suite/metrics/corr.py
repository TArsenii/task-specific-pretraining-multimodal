from numpy import corrcoef, ndarray

def pearson(y_true: ndarray, y_pred: ndarray) -> float:
    return corrcoef(y_true, y_pred)[0][1]