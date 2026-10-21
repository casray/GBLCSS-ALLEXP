from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'simmlp_matched_backbone_seed0_20261009'
sys.path.insert(0, str(BASE))
from matched_adapter import build_model, effective_config, state_hash
from protocol import set_seed


def build_probe(data, seed):
    folder = BASE / 'runs' / data.name / 'simmlp_matched'
    source_hashes = json.loads((folder / 'source_hashes.json').read_text())
    for path, expected in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Cached SimMLP source has changed: ' + path)
    checkpoint_path = folder / 'checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location=data.x.device, weights_only=False)
    cfg = effective_config(data.name, False)
    if checkpoint['config'] != cfg:
        raise RuntimeError('Cached SimMLP configuration mismatch')
    cfg.update(seed=seed, pretrain_seed=0, probe_init_seed=seed,
               representation_policy='fixed label-free seed0 encoder across downstream seeds',
               pretrained_checkpoint=str(checkpoint_path),
               pretrained_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
    set_seed(seed)
    model, head, _ = build_model(data, cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.encoder.requires_grad_(False)
    if state_hash(model.encoder) != checkpoint['frozen_encoder_hash']:
        raise RuntimeError('Cached frozen encoder hash mismatch')
    return model, head, cfg


def run(data, seed, outdir, smoke=False):
    model, head, cfg = build_probe(data, seed)
    cfg['smoke'] = smoke
    if smoke:
        cfg['lr_epochs'] = 2
    frozen_hash = state_hash(model.encoder)
    with torch.no_grad():
        embedding = model.encode(data.x).detach()
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg['lr_learning_rate'],
                                 weight_decay=cfg['lr_weight_decay'])
    best_val, best_epoch, best_state = -1., 0, None
    with (outdir / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, cfg['lr_epochs'] + 1):
            head.train()
            loss = F.cross_entropy(head(embedding[data.train_index]), data.y[data.train_index])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SimMLP probe objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError('Nonfinite SimMLP gradient')
            optimizer.step()
            head.eval()
            with torch.no_grad():
                valid = float((head(embedding[data.val_index]).argmax(1) == data.val_y).double().mean())
            selected = valid > best_val
            if selected:
                best_val, best_epoch, best_state = valid, epoch, copy.deepcopy(head.state_dict())
            log.write(json.dumps(dict(stage='probe', epoch=epoch, loss=float(loss.detach()),
                                      valid=valid, selected=selected), allow_nan=False) + '\n')
    if state_hash(model.encoder) != frozen_hash:
        raise RuntimeError('Encoder changed during frozen probe')
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probabilities = head(model.encode(data.x)).softmax(1).cpu()
    torch.save(probabilities, outdir / 'probabilities.pt')
    torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()},
                    classifier={k: v.cpu() for k, v in head.state_dict().items()}, config=cfg,
                    best_epoch=best_epoch, frozen_encoder_hash=frozen_hash), outdir / 'checkpoint.pt')
    return dict(best_val=best_val, best_epoch=best_epoch, config=cfg,
                parameters=sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in head.parameters()),
                frozen_encoder_hash=frozen_hash)
