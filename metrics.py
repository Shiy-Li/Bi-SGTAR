from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from utils import sort_matrix


def metrics(score_matrix, roc_circrna_disease_matrix):
    sorted_circrna_disease_matrix, sorted_score_matrix, sort_index = sort_matrix(score_matrix,
                                                                                 roc_circrna_disease_matrix)
    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    F1_list = []
    for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
        P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
        N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        F1 = (2 * TP) / (2 * TP + FP + FN)
        if (2 * TP + FP + FN) == 0:
            F1 = 0
        F1_list.append(F1)
        accuracy_list.append(accuracy)

    # Here are the counts of predicted accuracy for top50, top100, and top200
    top_list = [50, 100, 200]
    for num in top_list:
        P_matrix = sorted_circrna_disease_matrix[0:num, :]
        N_matrix = sorted_circrna_disease_matrix[num:sorted_circrna_disease_matrix.shape[0] + 1, :]
        top_count = np.sum(P_matrix == 1)
        # print("top" + str(num) + ": " + str(top_count))

    tpr_arr_epoch = np.array(tpr_list)
    fpr_arr_epoch = np.array(fpr_list)
    recall_arr_epoch = np.array(recall_list)
    precision_arr_epoch = np.array(precision_list)
    accuracy_arr_epoch = np.array(accuracy_list)
    F1_arr_epoch = np.array(F1_list)
    auc_epoch = np.trapz(tpr_arr_epoch, fpr_arr_epoch)
    aupr_epoch = np.trapz(precision_arr_epoch, recall_arr_epoch)

    return tpr_list, fpr_list, recall_list, precision_list, accuracy_list, F1_list


def calculate_performace(y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    num = len(y_prob)
    y_pred = np.where(y_prob >= 0.5, 1., 0.)
    for index in range(num):
        if y_test[index] == 1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / num
    try:
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1_score = float((2 * precision * recall) / (precision + recall))
        MCC = float(tp * tn - fp * fn) / (np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
        sens = tp / (tp + fn)
        spec = tn / (tn + tp)
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision = recall = f1_score = sens = MCC = spec = 100
    AUC = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    return tp, fp, tn, fn, acc, precision, sens, f1_score, MCC, AUC, auprc, spec
