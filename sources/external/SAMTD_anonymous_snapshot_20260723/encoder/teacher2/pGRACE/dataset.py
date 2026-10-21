import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(path, name):
    # assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
    #                 'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    assert name in ['Cora', 'CiteSeer', 'Pubmed', 'dblp',
                    'Computers', 'Photo','CS']
    name = 'dblp' if name == 'DBLP' else name
    # root_path = osp.expanduser('~/datasets')

    # if name == 'Coauthor-CS':
    #     return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    #
    # if name == 'Coauthor-Phy':
    #     return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())
    #
    # if name == 'WikiCS':
    #     return WikiCS(root=path, transform=T.NormalizeFeatures())
    #
    # if name == 'Amazon-Computers':
    #     return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    #
    # if name == 'Amazon-Photo':
    #     return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    #
    # if name.startswith('ogbn'):
    #     return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())
    #
    if name in ['Photo','Computers']:
        # print(name)
        return Amazon(path, name=name, transform=T.NormalizeFeatures())

    if name in ['Cora','Pubmed','CiteSeer']:
        # print(name)
        return Planetoid(path, name=name, transform=T.NormalizeFeatures())

    if name in ['dblp']:
        # print(name)
        return CitationFull(root=path, name=name, transform=T.NormalizeFeatures())

    if name in ['CS']:
        # print(name)
        return WikiCS(root=path+'/CS', transform=T.NormalizeFeatures())
    # return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'Pubmed']:
        return base_path
    else:
        return osp.join(base_path, name)
