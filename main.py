import math

import numpy as np
import torch
from sklearn.metrics import roc_curve, average_precision_score, auc
from train import train
from metrics import metrics, calculate_performace
from utils import load_association, build_model
import random


def main(args):
    circrna_disease_matrix = load_association(args)

    print('Now Load Dataset ' + str(args.data))
    args.rna_num = circrna_disease_matrix.shape[0]
    args.dis_num = circrna_disease_matrix.shape[1]
    print('rna_num', args.rna_num)
    print('dis_num', args.dis_num)
    n_fold = 5
    if args.data == 8:
        n_fold = 10
    index_tuple = (np.where(circrna_disease_matrix == 1))
    index_tuple_0 = (np.where(circrna_disease_matrix == 0))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    zero_list = list(zip(index_tuple_0[0], index_tuple_0[1]))
    rnd_state = random.Random(100)
    rnd_state.shuffle(one_list)
    rnd_state.shuffle(zero_list)

    split = math.ceil(len(one_list) / n_fold)
    split_0 = math.ceil(len(zero_list) / n_fold)
    print('split: ', split)
    print('split_0: ', split_0)
    # Evaluation Option-1
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # Evaluation Option-2
    # y_prob = []
    # y_test = []
    # 5-fold start
    for i in range(n_fold):
        test_index = one_list[i * split:(i + 1) * split]
        test_index_0 = zero_list[i * split_0:(i + 1) * split_0]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()

        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix
        circnum = rel_matrix.shape[0]
        disnum = rel_matrix.shape[1]
        rel_matrix_tensor = torch.tensor(np.array(rel_matrix).astype(np.float32))

        model_init = build_model(args.model_type)
        model = model_init(args)
        if args.cuda:
            model = model.cuda()
            rel_matrix_tensor = rel_matrix_tensor.cuda()

        smooth_factor = args.para
        norm_rel = smooth_factor + (1 - 2 * smooth_factor) * rel_matrix_tensor
        resi = train(model, norm_rel, args, args.alpha, i, rel_matrix_tensor)
        if args.cuda:
            ymat = resi.cpu().detach().numpy()
        else:
            ymat = resi.detach().numpy()

        ###--------------------------------Evaluation Option 2------------------------------------------------------###
        # S = ymat
        # for index_0 in test_index_0:
        #     y_prob.append(S[index_0[0],index_0[1]])
        #     y_test.append(circrna_disease_matrix[index_0[0],index_0[1]])
        # for index in test_index:
        #     y_prob.append(S[index[0],index[1]])
        #     y_test.append(circrna_disease_matrix[index[0],index[1]])

    # y_prob = np.array(y_prob)
    # y_test = np.array(y_test)
    # fpr, tpr, threshold = roc_curve(y_test, y_prob)
    # auc_val = auc(fpr, tpr)
    # aupr_val = average_precision_score(y_test, y_prob)
    # print('Final: \n  auc_val = \t'+ str(auc_val)+'\n  avpr_val = \t'+ str(aupr_val))
    # print('-' * 200)
    # return auc_val

        ###--------------------------------Evaluation Option 1------------------------------------------------------###
        S = ymat
        prediction_matrix = S
        zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
        score_matrix_temp = prediction_matrix.copy()
        score_matrix = score_matrix_temp + zero_matrix
        minvalue = np.min(score_matrix)
        score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 100
        tpr_list, fpr_list, recall_list, precision_list, accuracy_list, F1_list = metrics(score_matrix,
                                                                                          roc_circrna_disease_matrix)
        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_F1.append(F1_list)

    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)

    mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)
    mean_cross_accuracy = np.mean(accuracy_arr, axis=0)

    # Average
    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    #
    print("The average values")
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))
    #
    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))
    print('-' * 200)
    return roc_auc
