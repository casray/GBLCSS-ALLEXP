import torch
import torch.nn as nn
from ..layers import GCNNet, GCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, dataset, out_channel):
        super(DGI, self).__init__()
        # self.gcn = GCNNet(dataset)
        self.gcn = GCN(dataset.num_features, out_channel, act='prelu').to('cuda')
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(out_channel)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, msk):
        h_1 = self.gcn(seq, adj)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

