import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


def scatter_mean(src, index, dim_size):
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    out = src.new_zeros((dim_size, src.shape[-1]))
    count = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones((src.shape[0], 1), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


class TFRDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, decoder_dropout=0.5, att_softmax=False):
        super().__init__()
        self.att_layer = TFRATTLayer(in_channels, hidden_channels, att_softmax)
        self.lin_transform = nn.Linear(hidden_channels, out_channels)
        self.dropout = decoder_dropout

    def forward(self, x, edge_index):
        x, confidence = self.att_layer(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_transform(x)
        return x, confidence


class TFRATTLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, att_softmax=False):
        super().__init__(aggr="mean")
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_att = nn.Linear(2 * out_channels, 1, bias=False)
        self.lin_2hh = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.att_softmax = att_softmax
        self.confidence = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_att.reset_parameters()
        self.lin_2hh.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        return out, self.confidence

    def message(self, x_i, x_j, edge_index, size_i):
        x_i1 = self.lin_src(x_i)
        x_j1 = self.lin_dst(x_j)
        alpha = self.lin_att(torch.cat([F.relu(x_i1), F.relu(x_j1)], dim=-1))
        alpha = torch.sigmoid(alpha)
        self.confidence = scatter_mean(alpha, edge_index[1], int(size_i)).detach()
        if self.att_softmax:
            alpha = softmax(alpha, edge_index[1], num_nodes=size_i)
        x_ij = self.lin_2hh(torch.cat([x_i, x_j], dim=-1))
        return alpha * x_ij

    def aggregate(self, msg, index, dim_size):
        return scatter_mean(msg, index, int(dim_size))
