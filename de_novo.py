import numpy as np
import torch
from utils import load_dict, load_association, build_model
from train import train, find_key


def denovo(args):
    circrna_disease_matrix = load_association(args)

    print('Now Load Dataset ' + str(args.data))
    args.rna_num = circrna_disease_matrix.shape[0]
    args.dis_num = circrna_disease_matrix.shape[1]
    print('rna_num', args.rna_num)
    print('dis_num', args.dis_num)
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # Six diseases that require special consideration are recorded in the dictionary
    cancer_dict = load_dict(args.data)
    for i in range(circrna_disease_matrix.shape[1]):
        # association matrix for training
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        # association matrix used to calculate indicators
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        # Checks if the current column is not all zeros, and replaces the other column if it is all zeros.
        if not (False in (new_circrna_disease_matrix[:, i] == 0)):
            continue
        # Replace the current column with 0
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix

        args.rna_num = rel_matrix.shape[0]
        args.dis_num = rel_matrix.shape[1]

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

        S = ymat
        prediction_matrix = S
        aa = prediction_matrix.shape
        bb = roc_circrna_disease_matrix.shape
        zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))

        sort_index = np.argsort(-prediction_matrix[:, i], axis=0)
        sorted_circrna_disease_row = roc_circrna_disease_matrix[:, i][sort_index]

        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []

        for cutoff in range(1, rel_matrix.shape[0] + 1):
            P_vector = sorted_circrna_disease_row[0:cutoff]
            N_vector = sorted_circrna_disease_row[cutoff:]
            TP = np.sum(P_vector == 1)
            FP = np.sum(P_vector == 0)
            TN = np.sum(N_vector == 0)
            FN = np.sum(N_vector == 1)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            F1_list.append(F1)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            accuracy_list.append(accuracy)

        # Judgment call on i to determine if we've reached the six diseases that need to be considered separately
        if i in cancer_dict.values():
            top_count_list = [10, 20, 50, 100, 200]
            top_count = []
            for count in top_count_list:
                p_vector = sorted_circrna_disease_row[:count]
                top_count.append(np.sum(p_vector == 1))

            print("Number is：" + str(i) + " Dis Name：" + find_key(i, cancer_dict) + " top results as follow： \n")
            for j in range(len(top_count)):
                print("top_" + str(top_count_list[j]) + " results：", top_count[j])

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

    mean_denovo_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    mean_denovo_fpr = np.mean(fpr_arr, axis=0)
    mean_denovo_recall = np.mean(recall_arr, axis=0)
    mean_denovo_precision = np.mean(precision_arr, axis=0)
    mean_denovo_accuracy = np.mean(accuracy_arr, axis=0)
    # The average
    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

    roc_auc = np.trapz(mean_denovo_tpr, mean_denovo_fpr)
    AUPR = np.trapz(mean_denovo_precision, mean_denovo_recall)
    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))
