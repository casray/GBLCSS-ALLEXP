import torch
from torch import nn
import torch.nn.functional as F

class weight_matrix(nn.Module):
    def __init__(self, num_class, num_teachers):
        super(weight_matrix, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_teachers, num_class), requires_grad=True)

    def forward(self, pre_SUGRL, pre_GCA, pre_DGI):
        multi_teacher_pre = torch.stack((pre_SUGRL, pre_GCA, pre_DGI), dim=1).detach()
        teacher_pre = (multi_teacher_pre * self.weight).sum(dim = 1)
        teacher_pre = F.softmax(teacher_pre, dim=1)
        return teacher_pre
