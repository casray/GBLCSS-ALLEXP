import torch
import torch.nn as nn
import torch.nn.functional as F

class classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(ft_in, nb_classes)
        # self.fc2 = nn.Linear(512, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embs):
        # z = F.elu(self.fc1(embs))
        pre = self.fc1(embs)
        return pre