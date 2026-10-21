from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np

from .model import LogReg
# from model import LogReg
from numpy.testing import assert_array_almost_equal

def get_idx_split(dataset, dataset_name,split, preload_split):

    if dataset_name in ['Cora','PubMed','CiteSeer','dblp','Photo','Computers','CS']:

        if dataset_name in ['CS']:
            data = dataset.data
        else:
            data = dataset[0]
        # n_nodes = dataset[0].x.size(0)
        n_nodes = data.x.size(0)
        # data = dataset[0]
        if dataset_name in ['Cora', 'CiteSeer']:  # ,'PubMed'
            # train_lbls = label[data.train_mask]
            # train_lbls = noisy_label[data.train_mask]
            train_mask = data.train_mask
            idx_train = torch.where(train_mask == True)[0]
            # test_lbls = label[data.test_mask]
            test_mask = data.test_mask
            idx_test = torch.where(test_mask == True)[0]
            val_mask = data.val_mask
            # val_lbls = label[data.val_mask]
            idx_val = torch.where(val_mask == True)[0]
        elif dataset_name in ['PubMed', 'dblp']:
            # randomly select 0.1 * num_nodes training set
            # print(label.size()[0],n_nodes)
            print("n_nodes:", n_nodes)
            random_split = np.random.permutation(n_nodes)

            # train_index = random_split[:int(n_nodes * 0.01)]
            if dataset_name in ['Photo', ]:
                train_index = random_split[:int(n_nodes * 0.03)]
                # test_index = random_split[int(n_nodes * 0.2):]
            elif dataset_name in ['Computers']:
                train_index = random_split[:int(n_nodes * 0.05)]
            else:
                train_index = random_split[:int(n_nodes * 0.1)]

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            train_mask[train_index] = True
            idx_train = torch.where(train_mask == True)[0]
            val_index = random_split[int(n_nodes * 0.1):int(n_nodes * 0.2)]
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask[val_index] = True
            idx_val = torch.where(val_mask == True)[0]
            test_index = random_split[int(n_nodes * 0.2):]
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask[test_index] = True
            idx_test = torch.where(test_mask == True)[0]

        elif dataset_name in ['CS']:  # ,'PubMed'
            # train_lbls = label[data.train_mask]
            # train_lbls = noisy_label[data.train_mask]
            train_mask = data.train_mask[:,0]
            idx_train = torch.where(train_mask == True)[0]
            # test_lbls = label[data.test_mask]
            test_mask = data.test_mask
            idx_test = torch.where(test_mask == True)[0]
            val_mask = data.val_mask[:,0]
            # val_lbls = label[data.val_mask]
            idx_val = torch.where(val_mask == True)[0]

        elif dataset_name in ['Photo', 'Computers', 'ogbn-arxiv', 'ogbn-products', 'ogbn-mag']:  # ,'PubMed','dblp'
            # randomly select 0.1 * num_nodes training set
            # print(label.size()[0],n_nodes)
            random_split = np.random.permutation(n_nodes)
            train_index = random_split[:int(n_nodes * 0.03)]
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            train_mask[train_index] = True
            idx_train = torch.where(train_mask == True)[0]
            val_index = random_split[int(n_nodes * 0.1):int(n_nodes * 0.2)]
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask[val_index] = True
            idx_val = torch.where(val_mask == True)[0]
            test_index = random_split[int(n_nodes * 0.2):]
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask[test_index] = True
            idx_test = torch.where(test_mask == True)[0]

        return {
                # 'train': indices[:train_size],
                # 'val': indices[train_size:2 * train_size],
                # 'test': indices[2 * train_size:]
            'train': idx_train,
            'val': idx_val,
            'test': idx_test
            }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def log_regression(z,
                   dataset,
                   dataset_name,
                   evaluator,
                   seed,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None,
                   noise_ratio = 0.2,
                   noise_type = 'uniform'):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = dataset[0].y.view(-1).to(test_device)
    num_classes = dataset[0].y.max().item() + 1

    noisy_label, trans = noisify_p(y.cpu().numpy().flatten(), n_class=num_classes,
                                   noise_ratio=noise_ratio, random_state=seed,
                                   noise_type=noise_type)
    noisy_label = torch.from_numpy(noisy_label).to(test_device)

    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    split = get_idx_split(dataset, dataset_name,split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(z[split['train']])
        # loss = nll_loss(f(output), y[split['train']])
        loss = nll_loss(f(output), noisy_label[split['train']])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}

def uniform_trans(n_class, noise_ratio):
    # uniform transition matrix
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.0)
    trans = np.float64(noise_ratio) / np.float64(n_class - 1) * np.ones((n_class,n_class))
    np.fill_diagonal(trans,(np.float64(1) - np.float64(noise_ratio))*np.ones(n_class))
    # print(trans.sum(0))
    diag_idx = np.arange(n_class)
    # make sure that sum of every row is 0
    trans[diag_idx,diag_idx] = trans[diag_idx,diag_idx] + 1.0 - trans.sum(0)
    # assert_array_almost_equal(a,b,decimal=6)
    assert_array_almost_equal(trans.sum(axis=1),1,1)
    # print(diag_idx)
    return trans

def pair_trans(n_class, noise_ratio):
    # pair_transition matrix
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.0)
    trans = (1.0 - np.float64(noise_ratio)) * np.eye(n_class)
    for i in range(n_class):
        trans[i-1,i] = np.float64(noise_ratio)
    assert_array_almost_equal(trans.sum(axis=1), 1, 1)
    return  trans

def inter_class_noisify(labels, trans,random_state=0):
    # flip classes according to transition matrix
    assert trans.shape[0] == trans.shape[1]
    assert np.max(labels) < trans.shape[0]
    # assert torch.max(labels) < trans.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(trans.sum(axis=1), np.ones(trans.shape[1]))
    assert (trans >= 0.0).all()

    # trans = torch.from_numpy(trans)
    m = labels.shape[0]
    new_labels = labels.copy()
    # new_labels = torch.clone(labels)
    # random number generator
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        # i ranges from 0 to n_class-1
        i = labels[idx]
        flipped = flipper.multinomial(1, trans[i, :], 1)[0]
        # sample_label = torch.multinomial(trans[i,:],1)[0]
        new_labels[idx] = np.where(flipped == 1)[0]
        # new_labels[idx] = sample_label

    return new_labels

def noisify_p(labels,n_class,noise_ratio,random_state=None,noise_type='uniform'):
    if noise_ratio > 0.0:
        if noise_type == 'uniform':
            print("Uniform noise")
            trans = uniform_trans(n_class,noise_ratio)
        elif noise_type == 'pair':
            print("Pair noise")
            trans = pair_trans(n_class,noise_ratio)
        else:
            print("Noise type not implemented")

        noisy_labels = inter_class_noisify(labels,trans,random_state)
        actual_noise_ratio = (noisy_labels != labels).mean()
        assert actual_noise_ratio > 0.0
        print("Actual noise ratio:{:.2f}".format(actual_noise_ratio))
        labels = noisy_labels
    else:
        print("Actual noise ratio:0")
        trans = np.eye(n_class)

    return labels,trans