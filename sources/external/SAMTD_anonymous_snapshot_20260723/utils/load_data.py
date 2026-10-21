import os
import torch
import numpy as np
from torch_geometric.datasets import CitationFull, Planetoid, Amazon
import torch_geometric.transforms as T

def load_data(dataset_name, device):
    if dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        dataset = Planetoid(path, dataset_name, pre_transform=T.NormalizeFeatures())
        data = dataset[0]
        data = data.to(device)
        idx_train = torch.where(data.train_mask == True)[0].cpu().numpy()

    elif dataset_name in ['dblp', 'Photo', 'Computers']:
        if dataset_name == 'dblp':
            dataset = CitationFull('./data', 'dblp')
        elif dataset_name in ['Photo', 'Computers']:
            dataset = Amazon('./data', dataset_name)
        data = dataset[0]
        data = data.to(device)
        labels = dataset.data.y.numpy()
        idx = np.arange(len(labels))  # https://www.jianshu.com/p/bf1ccfb8c518
        np.random.shuffle(idx)
        idx_test = idx[:int(0.6 * len(labels))]
        idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
        idx_train = idx[int(0.9 * len(labels)):int((0.9 + 0.05) * len(labels))]
        data.train_mask = torch.zeros(data.num_nodes, dtype=bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=bool)
        for i in idx_train:
            data.train_mask[i] = 1
        for i in idx_val:
            data.val_mask[i] = 1
        for i in idx_test:
            data.test_mask[i] = 1
    return dataset, data, idx_train