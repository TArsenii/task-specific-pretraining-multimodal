import numpy as np
from sklearn.metrics import accuracy_score, f1_score



def __multiclass_acc( y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

def mosei_regression(y_pred, y_true):
    import warnings
    warnings.filterwarnings("error")
    
    try:
        test_preds = y_pred
        test_truth = y_true

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        # test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        # test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = __multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = __multiclass_acc(test_preds_a5, test_truth_a5)
        # mult_a3 = __multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')
        
        eval_results = {
            "Has0_Acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_Acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_Acc_5": round(mult_a5, 4),
            "Mult_Acc_7": round(mult_a7, 4),
            "Corr": round(corr, 4)
        }
    except RuntimeWarning as e:
        print(e)
    finally:
        warnings.filterwarnings("ignore")
    return eval_results