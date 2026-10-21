import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from time import perf_counter as t
from tqdm import tqdm
from scipy.sparse import diags
from torch_geometric.datasets import CitationFull, Planetoid, Amazon
from scipy.sparse import coo_matrix
from .models import LogReg, SUGRL_Fast
from torch_geometric.utils import degree
import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint

def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])


def get_args(model_name, dataset_name, seed, device, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "utils/args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--task-type", default="", type=str)
    parser.add_argument("--perf-type", default="accuracy", type=str)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--verbose", default=2)
    parser.add_argument("--save-plot", default=False)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--device", default = device, type = str)

    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--data-sampling-size", default=None, type=int, nargs="+")
    parser.add_argument("--data-sampling-num-hops", default=None, type=int)
    parser.add_argument("--data-num-splits", default=1, type=int)
    parser.add_argument("--data-sampler", default=None, type=str)

    # only Test
    parser.add_argument("--pretrain", default=True, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--loss", default=None, type=str)
    parser.add_argument('--lr2', '--learning-rate2', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate2', dest='lr2')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    # Early stop
    parser.add_argument("--use-early-stop", default=False, type=bool)
    parser.add_argument("--early-stop-patience", default=-1, type=int)
    parser.add_argument("--early-stop-queue-length", default=100, type=int)
    parser.add_argument("--early-stop-threshold-loss", default=-1.0, type=float)
    parser.add_argument("--early-stop-threshold-perf", default=-1.0, type=float)



    # Baseline
    parser.add_argument("--is-link-gnn", default=False, type=bool)
    parser.add_argument("--link-lambda", default=0., type=float)

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args

def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def get_dataset(args, dataset_kwargs):
    if args.dataset_name in ['Cora', 'CiteSeer', 'Pubmed', 'dblp', 'Photo', 'Computers']:
        if args.dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/utils', 'data')
            dataset = Planetoid(path, args.dataset_name)

        elif args.dataset_name in ['dblp', 'Photo', 'Computers']:
            if args.dataset_name == 'dblp':
                dataset = CitationFull('./data', 'dblp')
            elif args.dataset_name in ['Photo', 'Computers']:
                path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/utils', 'data', args.dataset_name)
                dataset = Amazon('./data', args.dataset_name) #, pre_transform=None)  # transform=T.ToSparseTensor(),
        data = dataset[0]

        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        adj = coo_matrix(
            (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(data.num_nodes, data.num_nodes))
        nb_nodes = data.num_nodes
        I = coo_matrix((np.ones(nb_nodes), (np.arange(0, nb_nodes, 1), np.arange(0, nb_nodes, 1))),
                       shape=(nb_nodes, nb_nodes))
        adj_I = adj + I  # coo_matrix(sp.eye(adj.shape[0]))
        adj_I = row_normalize(adj_I)
        A_I_nomal = sparse_mx_to_torch_sparse_tensor(adj_I).float()
        lable = data.y
        nb_feature = data.num_features
        nb_classes = int(lable.max() - lable.min()) + 1

    return data, [A_I_nomal], [data.x], [lable, nb_feature, nb_classes, nb_nodes]

def run_GCN(args, gpu_id=None, **kwargs):
    # ===================================================#
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # ===================================================#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = args.device #"cpu" if gpu_id is None \
        #else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    # cprint("## Loading Dataset ##", "yellow")
    training = 0
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list = get_dataset(args, dataset_kwargs)
    lable = nb_list[0]
    nb_feature = nb_list[1]
    nb_classes = nb_list[2]
    nb_nodes = nb_list[3]
    feature_X = x_list[0].to(running_device)
    A_I_nomal = adj_list[0].to(running_device)
    # cprint("## Done ##", "yellow")
    # ===================================================#
    model = SUGRL_Fast(nb_feature, cfg=args.cfg,
                       dropout=args.dropout)

    if training == 1:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.to(running_device)
        A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
        edge_index = A_I_nomal._indices()[1]
        # ===================================================#
        my_margin = args.margin1  # args.margin1 = 0.8
        my_margin_2 = my_margin + args.margin2 # args.margin2 = 0.2
        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
        num_neg = args.NN
        lbl_z = torch.tensor([0.]).to(running_device)
        deg_list_2 = []
        deg_list_2.append(0)
        for i in range(nb_nodes):
            deg_list_2.append(deg_list_2[-1] + A_degree[i])
        idx_p_list = []
        for j in range(1, 101):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)

        loss_best = 1000
        patience = 100
        cnt_wait = 0
        start = t()
        for current_iter, epoch in enumerate(tqdm(range(10000))): #args.start_epoch, args.start_epoch + args.epochs + 1))):
            model.train()
            optimiser.zero_grad()
            idx_list = []
            for i in range(num_neg):
                idx_0 = np.random.permutation(nb_nodes)
                idx_list.append(idx_0)

            h_a, h_p = model(feature_X, A_I_nomal)

            h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
                idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                         idx_p_list[(epoch + 8) % 100]]) / 5
            s_p = F.pairwise_distance(h_a, h_p)
            s_p_1 = F.pairwise_distance(h_a, h_p_1)
            s_n_list = []
            for h_n in idx_list:
                s_n = F.pairwise_distance(h_a, h_a[h_n])
                s_n_list.append(s_n)
            margin_label = -1 * torch.ones_like(s_p)

            loss_mar = 0
            loss_mar_1 = 0
            mask_margin_N = 0
            for s_n in s_n_list:
                loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
                loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
                mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
            mask_margin_N = mask_margin_N / num_neg

            loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3

            if loss < loss_best:
                loss_best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'SUGRL_model.pth')
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                # print('Early stopping!')
                break

            loss.backward()
            optimiser.step()
        model.load_state_dict(torch.load('SUGRL_model.pth'))


    else:
        model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + '/parameters/' + 'SUGRL_' + args.dataset_name + '_model.pth'))

    model.eval()
    h_a, h_p = model.embed(feature_X, A_I_nomal)
    embs = h_p
    return embs.detach()


def get_embed1(args_from_main):
    main_args = get_args(
        model_name="SUGRL",  # GCN SUGRL
        # Planetoid, WikiCS, MyCitationFull, MyAmazon, Crocodile, MyCitationFull PygNodePropPredDataset
        dataset_name=args_from_main.dataset_name,  # Cora, CiteSeer, Pubmed, Photo, Computers
        custom_key="classification",  # classification, link, clu,
        seed=args_from_main.random_seed,
        device=args_from_main.device
    )
    embs_SUGRL = run_GCN(main_args)
    return embs_SUGRL
