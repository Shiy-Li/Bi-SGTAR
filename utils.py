import torch
import numpy as np
from model import BiSGTAR


def load_dict(data):
    if data == 1:
        cancer_dict = {'glioma': 7, 'bladder cancer': 9, 'breast cancer': 10, 'cervical cancer': 53,
                       'cervical carcinoma': 64, 'colorectal cancer': 11, 'gastric cancer': 19}
    elif data == 2:
        cancer_dict = {'glioma': 23, 'bladder cancer': 2, 'breast cancer': 4, 'cervical cancer': 6,
                       'colorectal cancer': 12, 'gastric cancer': 20}
    elif data == 3:
        cancer_dict = {'glioma': 20, 'bladder cancer': 19, 'breast cancer': 6, 'cervical cancer': 16,
                       'colorectal cancer': 1, 'gastric cancer': 0}
    elif data == 4:
        # circ2Traits
        cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
                       'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6,
                       'gastric cancer': 15}
    elif data == 5:
        # circad
        cancer_dict = {'bladder cancer': 94, 'breast cancer': 53, 'triple-negative breast cancer': 111, 'gliomas': 56,
                       'glioma': 76,
                       'cervical cancer': 65, 'colorectal cancer': 143, 'gastric cancer': 28}
    else:
        cancer_dict = {}
    return cancer_dict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_model(model_type):
    if model_type == 'BiSGTAR':
        return BiSGTAR


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sort_matrix(score_matrix, interact_matrix):
    sort_index = np.argsort(-score_matrix, axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted, sort_index


def load_association(args):
    if args.data == 5:
        circrna_disease_matrix = np.loadtxt('./data/Dataset5/1265_151_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 4:
        circrna_disease_matrix = np.loadtxt('./data/Dataset4/923_104_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 3:
        circrna_disease_matrix = np.loadtxt('./data/Dataset3/312_40_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 2:
        circrna_disease_matrix = np.loadtxt('./data/Dataset2/514_62_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 1:
        circrna_disease_matrix = np.loadtxt('./data/Dataset1/533_89_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 6:
        circrna_disease_matrix = np.loadtxt('./data/KGET-Dataset1/330_79_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 7:
        circrna_disease_matrix = np.loadtxt('./data/KGET-Dataset2/561_190_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 8:
        circrna_disease_matrix = np.loadtxt('./data/Dataset8/l_d2.csv', delimiter=',')

    elif args.data == 9:
        circrna_disease_matrix = np.loadtxt('./data/Dataset9/C_D2.csv', delimiter=',')
    else:
        print('No data available...')
        return ''
    return circrna_disease_matrix
