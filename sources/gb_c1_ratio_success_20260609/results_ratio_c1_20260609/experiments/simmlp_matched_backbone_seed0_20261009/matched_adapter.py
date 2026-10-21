from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / 'simmlp_tined_audit_seed0_20261009'))
sys.path.insert(0, str(WORKSPACE / 'GB-LCSS_GitHub/src'))
from simmlp_adapter import author_objects, effective_config as previous_config
from gblcss.models import MLP


class HiddenEncoder(nn.Module):
    """Exactly the first hidden block of the common deployed MLP."""
    def __init__(self, backbone):
        super().__init__()
        self.layer = backbone.layers[0]
        self.normalizer = backbone.batch_norms[0]
        self.dropout = backbone.dropout

    def forward(self, x):
        hidden = F.relu(self.layer(x), inplace=True)
        hidden = self.normalizer(hidden)
        return F.dropout(hidden, p=self.dropout, training=self.training)


def effective_config(dataset, smoke):
    cfg = previous_config(dataset, smoke)
    cfg.update(method='simmlp_matched_backbone', enc_layers=1, res_enc=False,
               enc_norm='batch', activation='ReLU', output_l2_normalization=False,
               deployed_layers=2, optimizer='Adam', use_scheduler=False,
               probe_selection='clean validation accuracy; first tie',
               head_initialization='common MLP head, initialized before pretraining',
               probe_encoder='frozen parameters and BatchNorm statistics; eval mode',
               adaptations=['common two-FC deployed MLP', 'Adam in both stages',
                            'fixed learning rate', 'first validation tie'])
    return cfg


def build_model(data, cfg):
    obj = author_objects()
    h = cfg['hidden_dim']
    backbone = MLP(data.x.shape[1], h, data.num_classes, 2, cfg['enc_drop'])
    encoder = HiddenEncoder(backbone)
    head = backbone.layers[-1]
    mlp = obj['MLP']
    aggregator = obj['GNN'](h, h, h, nn.PReLU, cfg['proj_layers'],
                            cfg['proj_norm'], cfg['proj_drop'], cfg['aggr_norm'])
    predictor = mlp(h, cfg['pred_dim'], h, nn.PReLU, cfg['pred_layers'],
                    False, cfg['pred_norm'], cfg['pred_drop'])
    decoder = mlp(h, h, data.x.shape[1], nn.PReLU, 2)
    model = obj['Model'](encoder, aggregator, predictor,
                         obj['Augment'](cfg['edge_mask'], cfg['feature_mask']), decoder)
    return model.to(data.x.device), head.to(data.x.device), obj


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(data, cfg, outdir):
    if data.x.device.type != 'cuda':
        raise ValueError('GPU required')
    if (data.y[~data.train_mask] != -1).any():
        raise ValueError('Nontraining labels must be hidden')
    model, head, obj = build_model(data, cfg)
    criterion = obj['Bootstrap'](cfg['eta'], aux_pos_ratio=cfg['aux_pos_ratio'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    # The author training routine receives neither labels nor evaluation masks.
    ssl_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    with (outdir / 'epochs.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, cfg['epochs'] + 1):
            loss = obj['train'](model, criterion, ssl_data, None, optimizer, None,
                                cfg['aug_rounds'], cfg['recon_lambda'])
            if not math.isfinite(loss):
                raise ValueError('Nonfinite pretraining loss')
            record = dict(stage='pretrain', epoch=epoch, loss=loss)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if epoch % 20 == 0 or epoch == cfg['epochs']:
                print(record, flush=True)
        model.eval()
        model.encoder.requires_grad_(False)
        frozen_hash = state_hash(model.encoder)
        with torch.no_grad():
            embedding = model.encode(data.x).detach()
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg['lr_learning_rate'],
                                     weight_decay=cfg['lr_weight_decay'])
        best_val, best_epoch, best_state = -1., 0, None
        for epoch in range(1, cfg['lr_epochs'] + 1):
            head.train()
            logits = head(embedding[data.train_index])
            loss = F.cross_entropy(logits, data.y[data.train_index])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite probe loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            head.eval()
            with torch.no_grad():
                pred = head(embedding[data.val_index]).argmax(1)
                valid = float((pred == data.val_y).double().mean())
            if valid > best_val:
                best_val, best_epoch = valid, epoch
                best_state = copy.deepcopy(head.state_dict())
            record = dict(stage='probe', epoch=epoch, loss=float(loss.detach()), valid=valid)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if epoch % 20 == 0 or epoch == cfg['lr_epochs']:
                print(record, flush=True)
    if state_hash(model.encoder) != frozen_hash:
        raise ValueError('Frozen encoder changed during probe')
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probabilities = head(model.encode(data.x)).softmax(1).cpu()
    torch.save(probabilities, outdir / 'probabilities.pt')
    torch.save(dict(model=model.state_dict(), classifier=head.state_dict(), config=cfg,
                    best_epoch=best_epoch, frozen_encoder_hash=frozen_hash), outdir / 'checkpoint.pt')
    parameters = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in head.parameters())
    common_parameters = sum(p.numel() for p in MLP(data.x.shape[1], 64, data.num_classes, 2, .5).parameters())
    if parameters != common_parameters:
        raise ValueError('Deployed parameter count differs from common MLP')
    return dict(best_val=best_val, best_epoch=best_epoch, parameters=parameters,
                common_mlp_parameters=common_parameters, frozen_encoder_hash=frozen_hash,
                pretrain_parameters=sum(p.numel() for p in model.parameters()))
