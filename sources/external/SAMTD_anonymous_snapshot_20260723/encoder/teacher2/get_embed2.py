import numpy
import argparse
import os.path as osp
import random
import nni
import os
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from .simple_param.sp import SimpleParam
from .pGRACE.model import Encoder, GRACE, GCNNet
from .pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from .pGRACE.eval import log_regression, MulticlassEvaluator
from .pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from .pGRACE.dataset import get_dataset

def train(model, data, args, param, feature_weights):
    model.train()

    def drop_edge(idx: int):
        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset_name == 'Coauthor-Phy' else None)

    return loss


def test(model, dataset, data, args, split, use_nni, final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset_name == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, args.seed, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        # acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']
        acc = log_regression(z, dataset, args.dataset_name, evaluator, args.seed, split='rand:0.1', num_epochs=3000, preload_split=split,
                             noise_ratio=args.noise_ratio,noise_type=args.noise_type)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc

def main(args_from_main):
    training = 0
    #torch.cuda.set_device(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=args_from_main.device)
    # parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--dataset_name', type=str, default=args_from_main.dataset_name)
    parser.add_argument('--param', type=str, default='default')# ['default', 'local:CS.json']
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--noise_ratio', type=float, default=0)
    parser.add_argument('--noise_type', type=str, default='uniform')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 128,
        'num_proj_hidden': 128 if args_from_main.dataset_name== 'Computers' else 32,
        'activation': 'rrelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 10000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/utils', 'data')
    # path = '/data/liuyujing/lyjtorch/idea2/idea_bi_level/data/Cora'
    dataset = get_dataset(path, args.dataset_name)

    data = dataset[0]
    data = data.to(device)
    if args.dataset_name in ['Computers']:
        encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                          base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    else:
        encoder = GCNNet(dataset.num_features)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    # print(param['num_hidden'], param['num_proj_hidden'], param['tau'])
    if training == 1:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )

        global drop_weights
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data.edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        elif param['drop_scheme'] == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        else:
            drop_weights = None

        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            if args.dataset_name == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            node_pr = compute_pr(data.edge_index)
            if args.dataset_name == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        elif param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(data)
            if args.dataset_name == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)

        loss_best = 1000
        patience = 100
        cnt_wait = 0
        for epoch in tqdm(range(1, param['num_epochs'] + 1)):
            optimizer.zero_grad()
            loss = train(model, data, args, param, feature_weights)
            if loss < loss_best:
                loss_best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'GCA_model.pth')
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                break

            loss.backward()
            optimizer.step()
        model.load_state_dict(torch.load('GCA_model.pth'))

    else:
        model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + '/parameters/' + 'GCA_' + args.dataset_name + '_model.pth'))
    model.eval()
    embs = model(data.x, data.edge_index)
    return embs.detach()


def get_embed2(args_from_main):
    embs_GCA = main(args_from_main)
    return embs_GCA