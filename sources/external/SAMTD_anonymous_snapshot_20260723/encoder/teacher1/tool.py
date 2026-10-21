import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"

    score_matrix = np.dot(embeddings, embeddings.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    # print(preds_all, labels_all )

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


def mi_loss_jsd(pos, neg):
    e_pos = torch.mean(sp_func(-pos))
    e_neg = torch.mean(torch.mean(sp_func(neg), 0))
    return e_pos + e_neg


def compute_ppr(a, alpha=0.2, self_loop=True):
    N = a.size()[0]
    I = torch.eye(N).to(a.device)
    # a = a + I
    eps = 2.2204e-16
    deg_inv_sqrt = (a.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    a = deg_inv_sqrt.unsqueeze(-1) * a * deg_inv_sqrt.unsqueeze(-2)

    return alpha * torch.inverse(I - (1 - alpha) * a)  # a(I_n-(1-a)A~)^-1


def sp_func(arg):
    return torch.log(1 + torch.exp(arg))


def reconstruct_loss(pre, gnd):
    nodes_n = gnd.shape[0]
    edges_n = np.sum(gnd) / 2
    weight1 = (nodes_n * nodes_n - edges_n) * 1.0 / edges_n
    weight2 = nodes_n * nodes_n * 1.0 / (nodes_n * nodes_n - edges_n)
    gnd = torch.FloatTensor(gnd).cuda()
    temp1 = gnd * torch.log(pre + (1e-10)) * (-weight1)
    temp2 = (1 - gnd) * torch.log(1 - pre + (1e-10))
    return torch.mean(temp1 - temp2) * weight2


