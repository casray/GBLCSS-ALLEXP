import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
import os
from .models import DGI, LogReg
from . import process
import argparse
from tqdm import tqdm
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training params
def arg_parse(args_from_main, noise_rate):
    parser = argparse.ArgumentParser(description='FewGraph arguments.')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=int, help='learning rate')
    parser.add_argument('--l2_coef', type=int)
    parser.add_argument('--nb_epochs', type=int, help='max epoch')
    parser.add_argument('--patience', type=int, help='How many epochs no progress then break training')
    parser.add_argument('--noisy_rate', type=float, help='noisy rate')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--drop_prob', type=int)
    parser.add_argument('--device', type=str, help='cuda or cpu')
    parser.add_argument('--dataset_name', type=str,default='Cora', help='Cora/CiteSeer/Pubmed/cs/Computers/Photo/dblp')
    parser.add_argument('--hid_units', type=int)
    parser.add_argument('--nonlinearity', type=str)

    parser.set_defaults(batch_size = 1,
                        lr = 0.01,
                        l2_coef = 0,
                        nb_epochs = 10000,
                        patience = 100,
                        noisy_rate = args_from_main.noisy_rate,
                        random_seed = 678,
                        drop_prob = 0.0,
                        device = args_from_main.device,
                        dataset_name = args_from_main.dataset_name,
                        hid_units = 512,
                        nonlinearity = 'prelu'
                       )
    return parser.parse_args()

def main(args_from_main):
    training = 0
    arg = arg_parse(args_from_main, 0)
    setup_seed(arg.random_seed)
    if arg.dataset_name in ['Photo', 'Computers']:
        arg.lr = 0.001

    out_channel = 512
    dataset, data = process.load_data(arg)
    features = data.x
    adj_0 = data.edge_index
    nb_nodes = features.shape[0]

    adj_value = torch.ones(adj_0.size(1))
    adj = coo_matrix((adj_value, (adj_0[0].tolist(), adj_0[1].tolist())))
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj).to(arg.device)

    model = DGI(dataset, out_channel)
    if training == 1:
        optimiser = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2_coef)

        if torch.cuda.is_available():
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        cnt_wait = 0
        best = 1e9
        for epoch in tqdm(range(arg.nb_epochs)):
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[idx, :]

            lbl_1 = torch.ones(arg.batch_size, nb_nodes)
            lbl_2 = torch.zeros(arg.batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()

            logits = model(features, shuf_fts, adj, None, None, None)

            loss = b_xent(logits, lbl)
            # print(epoch, loss)
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'DGI_model.pth')
            else:
                cnt_wait += 1
            if cnt_wait == arg.patience:
                break
            loss.backward()
            optimiser.step()
        model.load_state_dict(torch.load('DGI_model.pth'))
    else:
        model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + '/parameters/' + 'DGI_' + arg.dataset_name + '_model.pth'))

    model.eval()
    embeds, _ = model.embed(features, adj, None)
    return embeds.detach()

def get_embed3(args_from_main):
    embs_DGI= main(args_from_main)
    return embs_DGI