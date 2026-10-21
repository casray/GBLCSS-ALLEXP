from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/simmlp'


class SparseConv(nn.Module):
    """Tensor-edge path of official nonparametric GCN, using sparse.mm."""
    def __init__(self, in_channels, out_channels, aggr_norm='gcn', **kwargs):
        super().__init__()
        self.norm = aggr_norm

    def forward(self, x, edge_index, edge_weight=None):
        edges, weights = add_remaining_self_loops(
            edge_index, edge_weight, 1., x.shape[0])
        if weights is None:
            weights = torch.ones(edges.shape[1], device=x.device, dtype=x.dtype)
        src, dst = edges
        degree = x.new_zeros(x.shape[0]).scatter_add_(0, dst, weights)
        if self.norm == 'gcn':
            inv = degree.pow(-.5).masked_fill(degree == 0, 0)
            weights = inv[src] * weights * inv[dst]
        elif self.norm in ('row', 'col'):
            inv = degree.reciprocal().masked_fill(degree == 0, 0)
            weights = inv[src] * weights
        else:
            raise ValueError(self.norm)
        matrix = torch.sparse_coo_tensor(torch.stack([dst, src]), weights,
                                         (len(x), len(x))).coalesce()
        return torch.sparse.mm(matrix, x)


def author_objects():
    # Execute only selected unchanged definitions; unavailable binary imports stay out.
    ns = dict(torch=torch, nn=nn, F=F, GConv_wo_param=SparseConv)
    selections = {
        'model.py': {'get_normalize', 'MLP', 'GNN', 'Model'},
        'loss.py': {'Loss', 'Bootstrap'},
        'augment.py': {'dropout_edge', 'mask_feature', 'Augment'},
        'eval.py': {'LogisticRegression'},
        'main.py': {'train'},
    }
    from abc import ABC, abstractmethod
    from torch.distributions import Beta
    ns.update(ABC=ABC, abstractmethod=abstractmethod, Beta=Beta)
    for filename, names in selections.items():
        tree = ast.parse((VENDOR / filename).read_text(encoding='utf-8'))
        selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in selected} != names:
            raise ValueError(f'Missing author definitions in {filename}')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(VENDOR / filename), 'exec'), ns)
    return ns


def effective_config(name, smoke):
    preset = {'cora': 'cora', 'citeseer': 'citeseer', 'corafull': 'cora',
              'coauthorcs': 'co-cs', 'amazonpho': 'amazon-photo'}[name]
    cfg = json.loads((VENDOR / f'param/node/{preset}.json').read_text())
    cfg.update(author_preset=preset, dataset=name, seed=0, epochs=2 if smoke else 200,
               lr_epochs=2 if smoke else 200, hidden_dim=64, pred_dim=64,
               enc_drop=.5, learning_rate=.01, weight_decay=.0005,
               lr_learning_rate=.01, lr_weight_decay=.0005,
               pretrain_selection='fixed final epoch; no label-based selection',
               probe_selection='clean validation accuracy; last tie',
               feature_input='raw main-experiment attributes', full_graph=True)
    return cfg


def build_model(data, cfg):
    obj = author_objects()
    mlp = obj['MLP']
    h = cfg['hidden_dim']
    encoder = mlp(data.x.shape[1], h, h, nn.PReLU, cfg['enc_layers'],
                  cfg['res_enc'], cfg['enc_norm'], cfg['enc_drop'])
    aggregator = obj['GNN'](h, h, h, nn.PReLU, cfg['proj_layers'],
                            cfg['proj_norm'], cfg['proj_drop'], cfg['aggr_norm'])
    predictor = mlp(h, cfg['pred_dim'], h, nn.PReLU, cfg['pred_layers'],
                    False, cfg['pred_norm'], cfg['pred_drop'])
    decoder = mlp(h, h, data.x.shape[1], nn.PReLU, 2)
    model = obj['Model'](encoder, aggregator, predictor,
                         obj['Augment'](cfg['edge_mask'], cfg['feature_mask']), decoder)
    return model.to(data.x.device), obj


def run(data, cfg, outdir):
    if data.x.device.type != 'cuda':
        raise ValueError('GPU required')
    if (data.y[~data.train_mask] != -1).any():
        raise ValueError('Nontraining labels must be hidden')
    model, obj = build_model(data, cfg)
    criterion = obj['Bootstrap'](cfg['eta'], aux_pos_ratio=cfg['aux_pos_ratio'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 + math.cos(epoch * math.pi / cfg['epochs'])) * .5
    ) if cfg['use_scheduler'] else None
    with (outdir / 'epochs.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, cfg['epochs'] + 1):
            loss = obj['train'](model, criterion, data, None, optimizer, scheduler,
                                 cfg['aug_rounds'], cfg['recon_lambda'])
            if not math.isfinite(loss):
                raise ValueError('Nonfinite pretraining loss')
            record = dict(stage='pretrain', epoch=epoch, loss=loss)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if epoch % 20 == 0 or epoch == cfg['epochs']:
                print(record, flush=True)
        model.eval()
        with torch.no_grad():
            embedding = model.encode(data.x).detach()
        classifier = obj['LogisticRegression'](cfg['hidden_dim'], data.num_classes).to(data.x.device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg['lr_learning_rate'],
                                       weight_decay=cfg['lr_weight_decay'])
        best_val, best_epoch, best_state = -1., 0, None
        for epoch in range(1, cfg['lr_epochs'] + 1):
            classifier.train()
            logits = classifier(embedding[data.train_index])
            loss = F.cross_entropy(logits, data.y[data.train_index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            classifier.eval()
            with torch.no_grad():
                pred = classifier(embedding[data.val_index]).argmax(1)
                valid = float((pred == data.val_y).double().mean())
            if valid >= best_val:
                best_val, best_epoch = valid, epoch
                best_state = copy.deepcopy(classifier.state_dict())
            record = dict(stage='probe', epoch=epoch, loss=float(loss.detach()), valid=valid)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if epoch % 20 == 0 or epoch == cfg['lr_epochs']:
                print(record, flush=True)
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        probabilities = classifier(model.encode(data.x)).softmax(1).cpu()
    torch.save(probabilities, outdir / 'probabilities.pt')
    torch.save(dict(model=model.state_dict(), classifier=classifier.state_dict(), config=cfg,
                    best_epoch=best_epoch), outdir / 'checkpoint.pt')
    return dict(best_val=best_val, best_epoch=best_epoch,
                parameters=sum(p.numel() for p in model.encoder.parameters()) +
                           sum(p.numel() for p in classifier.parameters()),
                pretrain_parameters=sum(p.numel() for p in model.parameters()))
