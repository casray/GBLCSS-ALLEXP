import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNet(nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 512)
        #self.conv2 = GCNConv(128, dataset.num_classes)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)

        out = torch.mm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias

        x = self.act(out)
        return x

