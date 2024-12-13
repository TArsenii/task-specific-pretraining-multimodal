import traceback
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def msa_binarize(preds: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary_truth = (labels == 1).astype(int)  # 1 for Neutral, 0 otherwise
    binary_preds = (preds == 1).astype(int)  # 1 for Neutral, 0 otherwise

    # Indices of Non-Neutral samples (Negative and Positive)
    non_zero_indices = np.where(labels != 1)[0]

    # For Non0 metrics, distinguish Positive (2) vs Negative (0)
    # Here, Positive is 1, Negative is 0
    non_zero_binary_truth = (labels[non_zero_indices] == 2).astype(int)  # 1 for Positive, 0 for Negative
    non_zero_binary_preds = (preds[non_zero_indices] == 2).astype(int)  # 1 for Positive, 0 for Negative

    return (
        binary_preds,
        binary_truth,
        non_zero_indices,
        non_zero_binary_preds,
        non_zero_binary_truth,
    )


def __multiclass_acc(y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


def confusion_matrix_from_logits(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, **kwargs)


def msa_binary_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    binary_preds, binary_truth, non_zeros, non_zeros_binary_preds, non_zeros_binary_truth = msa_binarize(y_pred, y_true)

    non_zeros_accuracy = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)

    non_zeros_f1_weighted_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="weighted")
    non_zeros_f1_macro_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="macro")
    non_zeros_f1_micro_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="micro")
    non_zeros_recall_weighted_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="weighted")
    non_zeros_recall_macro_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="macro")
    non_zeros_recall_micro_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="micro")
    non_zeros_precision_weighted_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="weighted")
    non_zeros_precision_macro_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="macro")
    non_zeros_precision_micro_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="micro")

    binary_accuracy = accuracy_score(binary_preds, binary_truth)
    binary_f1_weighted_score = f1_score(binary_truth, binary_preds, average="weighted")
    binary_f1_macro_score = f1_score(binary_truth, binary_preds, average="macro")
    binary_f1_micro_score = f1_score(binary_truth, binary_preds, average="micro")
    binary_recall_weighted_score = f1_score(binary_truth, binary_preds, average="weighted")
    binary_recall_macro_score = f1_score(binary_truth, binary_preds, average="macro")
    binary_recall_micro_score = f1_score(binary_truth, binary_preds, average="micro")
    binary_precision_weighted_score = f1_score(binary_truth, binary_preds, average="weighted")
    binary_precision_macro_score = f1_score(binary_truth, binary_preds, average="macro")
    binary_precision_micro_score = f1_score(binary_truth, binary_preds, average="micro")

    return {
        "Non0_Accuracy": round(non_zeros_accuracy, 4),
        "Non0_F1_weighted": round(non_zeros_f1_weighted_score, 4),
        "Non0_F1_macro": round(non_zeros_f1_macro_score, 4),
        "Non0_F1_micro": round(non_zeros_f1_micro_score, 4),
        "Non0_Recall_weighted": round(non_zeros_recall_weighted_score, 4),
        "Non0_Recall_macro": round(non_zeros_recall_macro_score, 4),
        "Non0_Recall_micro": round(non_zeros_recall_micro_score, 4),
        "Non0_Precision_weighted": round(non_zeros_precision_weighted_score, 4),
        "Non0_Precision_macro": round(non_zeros_precision_macro_score, 4),
        "Non0_Precision_micro": round(non_zeros_precision_micro_score, 4),
        "Has0_Accuracy": round(binary_accuracy, 4),
        "Has0_F1_weighted": round(binary_f1_weighted_score, 4),
        "Has0_F1_macro": round(binary_f1_macro_score, 4),
        "Has0_F1_micro": round(binary_f1_micro_score, 4),
        "Has0_Recall_weighted": round(binary_recall_weighted_score, 4),
        "Has0_Recall_macro": round(binary_recall_macro_score, 4),
        "Has0_Recall_micro": round(binary_recall_micro_score, 4),
        "Has0_Precision_weighted": round(binary_precision_weighted_score, 4),
        "Has0_Precision_macro": round(binary_precision_macro_score, 4),
        "Has0_Precision_micro": round(binary_precision_micro_score, 4),
    }


def old_mosei_regression(y_true, y_pred):
    import warnings

    warnings.filterwarnings("error")

    try:
        test_preds = y_pred
        test_truth = y_true

        test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
        test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
        test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
        test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)
        # test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        # test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        # corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = __multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = __multiclass_acc(test_preds_a5, test_truth_a5)
        # mult_a3 = __multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = test_truth[non_zeros] > 0
        non_zeros_binary_preds = test_preds[non_zeros] > 0

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="weighted")

        binary_truth = test_truth >= 0
        binary_preds = test_preds >= 0
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average="weighted")

        eval_results = {
            "Has0_Acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_Acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_Acc_5": round(mult_a5, 4),
            "Mult_Acc_7": round(mult_a7, 4),
            # "Corr": round(corr, 4),
        }
    except RuntimeWarning as e:
        print(traceback.format_exc())
        raise e
    finally:
        warnings.filterwarnings("ignore")
    return eval_results
