import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import minmax_scale
import scipy.sparse as sp
import numpy as np

from model import MLP, MLPQC, AE, AECDA, AEMLP


def normalize_adj(mx, r):
    mx = torch.clamp(mx, min=0)
    mx = mx - torch.diag(torch.diag(mx))
    A = F.normalize(mx, p=1, dim=1)
    D = torch.diag(torch.sum(A, dim=1))
    D_inv_sqrt = torch.diag(torch.rsqrt(torch.diag(D)))
    L = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
    return L


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # if torch.cuda.is_available():
    #   return torch.sparse.FloatTensor(indices, values, shape).cuda()
    return torch.sparse.FloatTensor(indices, values, shape)


def neighborhood(feat, k):
    # print("This is neighborhood...")
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def normalized(wmat):
    # print("This is normalized...")
    # a = np.eye(wmat.shape[0])
    deg = np.diag(np.sum(wmat, axis=0))
    # print('deg',deg)
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    # print("This is norm_adj...")
    C = neighborhood(feat.T, k=10)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def GKL(data):
    # circ-disea邻接矩阵
    circR_disease = np.array(data)
    m, n = np.shape(circR_disease)

    # 计算circ参数
    normValueList_C = []
    for i in range(m):
        temp = np.linalg.norm(circR_disease[i], ord=2)
        normValueList_C.append(temp * temp)
    segamac = m / (np.sum(normValueList_C))

    # 计算dise参数
    normValueList_D = []
    for j in range(n):
        tempd = np.linalg.norm(circR_disease[:, j], ord=2)
        normValueList_D.append(tempd * tempd)
    segamad = n / (np.sum(normValueList_D))

    # circRNA高斯谱核相似性矩阵
    cicRNA_result = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            tempcirc = np.linalg.norm(circR_disease[i] - circR_disease[j], ord=2)
            cicRNA_result[i][j] = np.exp(-segamac * (tempcirc * tempcirc))

    # 计算disease高斯谱核相似性矩阵
    disease_result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tempdisea = np.linalg.norm(circR_disease[:, i] - circR_disease[:, j], ord=2)
            disease_result[i][j] = np.exp(-segamad * (tempdisea * tempdisea))
    return cicRNA_result, disease_result


def build_model(model_type):
    if model_type == 'MLP':
        return MLP
    elif model_type == 'MLPQC':
        return MLPQC
    elif model_type == 'AE':
        return AE
    elif model_type == 'AECDA':
        return AECDA
    elif model_type == 'AEMLP':
        return AEMLP


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def sort_matrix(score_matrix, interact_matrix):
    sort_index = np.argsort(-score_matrix, axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted, sort_index


def criterion(output, target, msg, n_nodes, mu, logvar):
    if msg == 'disease':
        cost = F.binary_cross_entropy(output, target)
        # cost = F.mse_loss(output, target)

    else:
        cost = F.binary_cross_entropy(output, target)

    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL


def calculate_performace(y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    num = len(y_prob)
    # print('y_prob',y_prob)
    y_pred = np.where(y_prob >= 0.5, 1., 0.)
    # print('y_pred',y_pred[y_pred == 1.])
    # print('y_test',y_test[y_test == 1.])
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
