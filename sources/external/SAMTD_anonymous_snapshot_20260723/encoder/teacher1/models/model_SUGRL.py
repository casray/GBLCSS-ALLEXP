import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.)+eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A =  deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def add_self_loops(edge_index, num_nodes=None):
        print("进入self_loops")
        loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                                  device=edge_index.device)
        print(loop_index)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        print(loop_index)

        edge_index = torch.cat([edge_index, loop_index], dim=1)
        print(edge_index)
        print("出self_loops")
        # 原来的edge_index为[[1, 2, 3],
        #                   [0, 0, 0]]
        #  这样一来，就在原来的边连接关系edge_index的基础上增加了self_loop的关系.
        #  torch.cat([edge_index, loop_index], dim=1)
        #      tensor([[1, 2, 3, 0, 1, 2, 3],
        #              [0, 0, 0, 0, 1, 2, 3]])

        return edge_index

def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)

class SUGRL_Fast(nn.Module):
    def __init__(self, n_in ,cfg = None, dropout = 0.2):
        super(SUGRL_Fast, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.A = None
        self.sparse = True
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj=None):
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, 0.2, training=self.training)
        if self.sparse:
            h_p = torch.spmm(adj, h_p_0)
        else:
            h_p = torch.mm(adj, h_p_0)
        return h_a, h_p

    def embed(self,  seq_a , adj=None ):
        h_a = self.MLP(seq_a.to('cpu')).to('cuda')
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()

