from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / 'new_graph_mlp_seed0_20261009'))
sys.path.insert(0, str(WORKSPACE / 'GB-LCSS_GitHub/src'))
import tined_adapter as previous
from tined_adapter import (SparseGraph, dirichlet_energy, _log_ratio, _OfficialSequence,
                           _seed, _state, _step, _val, _de_inputs, _student_logits,
                           de_loss, validate_data, _cpu_state)
from gblcss.models import GCN, MLP


class GCNGraph(SparseGraph):
    def __init__(self, edge_index, num_nodes, dtype=torch.float32):
        super().__init__(edge_index, num_nodes, dtype)
        # Match Table 8's coalesced native sparse adjacency and its orientation.
        self.adjacency = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.shape[1], dtype=dtype, device=edge_index.device),
            (num_nodes, num_nodes)).coalesce()
        self.normalized, _ = gcn_norm(self.adjacency, num_nodes=num_nodes,
                                     add_self_loops=True, dtype=dtype)

    def propagate(self, x):
        return torch.sparse.mm(self.normalized, x)


class GCNTeacher(nn.Module):
    def __init__(self, dims, dropout=.5):
        super().__init__()
        self.dims = list(dims)
        self.backbone = GCN(dims[0], dims[1], dims[2], 2, dropout)
        self.energy_degenerate_layers = []

    def forward(self, graph, x):
        return self.backbone(x, graph.adjacency)

    def forward_with_targets(self, graph, x):
        if self.training:
            raise ValueError('Extract GCN targets in evaluation mode')
        h, targets, degenerates = x, [], []
        for index, layer in enumerate(self.backbone.layers):
            operations = ('FT', 'GP') if layer.in_channels > layer.out_channels else ('GP', 'FT')
            for operation in operations:
                before = h
                h = F.linear(h, layer.lin.weight) if operation == 'FT' else graph.propagate(h)
                ein, eout = dirichlet_energy(graph, before), dirichlet_energy(graph, h)
                targets.append(_log_ratio(ein, eout, 0.))
                degenerates.append(bool((ein == 0) | (eout == 0)))
            # Official TINED excludes the deferred teacher bias from its FT/GP traces.
            h = h + layer.bias
            if index != len(self.backbone.layers) - 1:
                h = F.relu(h)
        self.energy_degenerate_layers = degenerates
        reference = self(graph, x)
        torch.testing.assert_close(h, reference, rtol=1e-4, atol=2e-5)
        return reference, torch.stack(targets).detach()


class TINEDGCNStudent(_OfficialSequence):
    def __init__(self, teacher, config):
        shapes = [(layer.in_channels, layer.out_channels) for layer in teacher.backbone.layers]
        ga_shapes = [(min(din, dout), min(din, dout)) for din, dout in shapes]
        args = SimpleNamespace(load_disabled=False, DE_sampling_ratio=config['DE_sampling_ratio'],
                               DE_regularization=False)
        super().__init__(len(shapes), shapes, ga_shapes, teacher.dims[0], 0,
                         config['hidden'], teacher.dims[-1], config['dropout'],
                         norm_type=config['norm_type'], from_MLP_mode='same_as_teacher', args=args)
        reference = teacher.backbone.layers[0].lin.weight
        self.to(device=reference.device, dtype=reference.dtype)
        self.register_buffer('DE_targets', reference.new_zeros(len(self.layers)))
        with torch.no_grad():
            for source, target in zip(teacher.backbone.layers, self.learned_MLP_layers):
                target.weight.copy_(source.lin.weight)
                target.bias.copy_(source.bias)
            if config['GA_init_type'] == 'identity':
                for layer in self.graph_aggregation_layers:
                    layer.weight.copy_(torch.eye(layer.out_features, layer.in_features,
                                                 device=reference.device, dtype=reference.dtype))
                    layer.bias.zero_()

    set_de_targets = previous.TINEDStudent.set_de_targets
    get_appoximated_DE = previous.TINEDStudent.get_appoximated_DE


def effective_config(dataset, smoke):
    # Keep the predeclared method knobs, but restore the author's student dropout.
    cfg = previous.resolve_config({'epochs': 2 if smoke else 200}, dataset)
    records = json.loads((previous.VENDOR / 'best_records/best_records.json').read_text())
    preset = records['SAGE-transductive']['ours'][cfg['author_preset']]
    cfg.update(method='tined_gcn_adapted', teacher='GCN', teacher_dropout=.5,
               dropout=float(preset['dropout_ratio']), optimizer='Adam', use_scheduler=False,
               teacher_training='common two-layer GCN on native coalesced sparse adjacency',
               teacher_initialization='common MLP initialization followed by common GCN, as Table 8 CE routes',
               teacher_validation_selection='clean validation accuracy; last tie',
               validation_selection='clean validation accuracy; first tie',
               algorithm_preset_family='SAGE-transductive',
               gcn_specific_author_best_preset_available=False,
               adaptations=['common GCN teacher instead of SAGE-gcn',
                            'author student dropout restored', 'first student validation tie'],
               smoke=smoke)
    return cfg


def select_checkpoint(model, best, epoch, val, last_tie=False):
    if val > best['val'] or (last_tie and val == best['val']):
        return _state(model), {'epoch': epoch, 'val': val}
    return None, best


def fit_teacher(model, graph, data, config, log):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    best, state = {'epoch': 0, 'val': -1.}, None
    for epoch in range(1, config['teacher_epochs'] + 1):
        model.train()
        logits = model(graph, data.x)
        loss = F.cross_entropy(logits[data.train_index], data.y[data.train_index])
        _step(optimizer, loss)
        model.eval()
        with torch.no_grad():
            val, val_loss = _val(model(graph, data.x)[data.val_index], data)
        selected, best = select_checkpoint(model, best, epoch, val, last_tie=True)
        if selected is not None:
            state = selected
        log(dict(phase='teacher', epoch=epoch, loss=float(loss.detach()),
                 val=val, val_loss=val_loss, selected=selected is not None))
    return state, best


def fit_student(model, graph, data, teacher_logp, config, log):
    injected = list(model.learned_MLP_layers.parameters())
    injected_ids = {id(p) for p in injected}
    other = [p for p in model.parameters() if id(p) not in injected_ids]
    optimizer = torch.optim.Adam([
        {'params': other, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},
        {'params': injected, 'lr': config['learning_rate'] * config['learned_MLP_lr_ratio'],
         'weight_decay': config['weight_decay']}])
    best, state = {'epoch': 0, 'val': -1.}, None
    size, lamb = config['batch_size'], config['lamb']
    input_energy = dirichlet_energy(graph, data.x).detach() if config['DE_sampling_ratio'] == 1 else None
    _seed()
    for epoch in range(1, config['student_epochs'] + 1):
        model.train()
        ce_values, kd_values = [], []
        # Preserve the author's separate CE, KL and DE steps, including zero-weight CE.
        for indices in data.train_index.split(size):
            ce = F.cross_entropy(model(data.x[indices])[1], data.y[indices])
            _step(optimizer, (1 - lamb) * ce)
            ce_values.append(float(ce.detach()))
        for start in range(0, data.x.shape[0], size):
            logits = model(data.x[start:start + size])[1]
            kl = F.kl_div(logits.log_softmax(1), teacher_logp[start:start + size],
                          reduction='batchmean', log_target=True)
            _step(optimizer, lamb * kl)
            kd_values.append(float(kl.detach()))
        de_graph, de_x = _de_inputs(graph, data.x, config['DE_sampling_ratio'])
        ratios = torch.stack(model.get_appoximated_DE(de_x, de_graph, input_energy))
        energy_loss = de_loss(ratios, model.DE_targets, config)
        _step(optimizer, energy_loss)
        model.eval()
        with torch.no_grad():
            val, val_loss = _val(_student_logits(model, data.x[data.val_index], size), data)
        selected, best = select_checkpoint(model, best, epoch, val)
        if selected is not None:
            state = selected
        ce, kl, de = sum(ce_values)/len(ce_values), sum(kd_values)/len(kd_values), float(energy_loss.detach())
        log(dict(phase='student', epoch=epoch, loss=(1-lamb)*ce+lamb*kl+de,
                 ce=ce, kl=kl, de=de, val=val, val_loss=val_loss,
                 log_de_ratios=ratios.detach().cpu().tolist(), selected=selected is not None))
    return state, best


def run(data, config, outdir):
    validate_data(data)
    directory = Path(outdir)
    if any((directory / name).exists() for name in ['checkpoint.pt', 'probabilities.pt', 'epochs.jsonl']):
        raise FileExistsError('Use a fresh TINED output directory')
    _seed()
    graph = GCNGraph(data.edge_index, len(data.x), data.x.dtype)
    dims = [data.x.shape[1], config['hidden'], data.num_classes]
    # Match the random-number consumption of the common CE teacher routes.
    unused_common_student = MLP(dims[0], dims[1], dims[-1], 2, .5)
    del unused_common_student
    teacher = GCNTeacher(dims, config['teacher_dropout']).to(device=data.x.device, dtype=data.x.dtype)
    with (directory / 'epochs.jsonl').open('x', encoding='utf-8') as handle:
        def log(row):
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            if row['epoch'] % 20 == 0 or row['epoch'] == config[row['phase'] + '_epochs']:
                print({key: row[key] for key in ['phase', 'epoch', 'loss', 'val']}, flush=True)

        teacher_state, teacher_best = fit_teacher(teacher, graph, data, config, log)
        teacher.load_state_dict(teacher_state)
        teacher.eval()
        with torch.no_grad():
            logits, targets = teacher.forward_with_targets(graph, data.x)
            teacher_logp = logits.log_softmax(1).detach()
        teacher.requires_grad_(False)
        _seed()
        student = TINEDGCNStudent(teacher, config)
        student.set_de_targets(targets)
        student_state, student_best = fit_student(student, graph, data, teacher_logp, config, log)
    student.load_state_dict(student_state)
    student.eval()
    with torch.no_grad():
        probabilities = _student_logits(student, data.x, config['batch_size']).softmax(1)
        selected_val = float((probabilities[data.val_index].argmax(1) == data.val_y).double().mean())
    if selected_val != student_best['val'] or not torch.isfinite(probabilities).all():
        raise RuntimeError('Selected student probability consistency failure')
    deploy_parameters = (sum(p.numel() for p in student.layers.parameters()) +
                         sum(p.numel() for norm in student.norms[:-1] for p in norm.parameters()))
    diagnostics = dict(student_fc_layers=len(student.layers), sequence=student.transformation_types_pseudo,
                       teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                       student_deploy_parameters=deploy_parameters,
                       teacher_log_de_targets=targets.cpu().tolist(),
                       teacher_zero_energy_layers=teacher.energy_degenerate_layers,
                       ft_shapes_in_out=[[l.in_features, l.out_features] for l in student.learned_MLP_layers],
                       gp_shapes_in_out=[[l.in_features, l.out_features] for l in student.graph_aggregation_layers],
                       test_labels_read=False, student_kl_nodes=len(data.x),
                       implementation='common PyG GCN + official AST student + sparse energy port',
                       direct_official_execution=False, deployment_input='raw x only')
    checkpoint = dict(teacher_state=_cpu_state(teacher_state), student_state=_cpu_state(student_state),
                      config=config, dims=dims, teacher_best=teacher_best,
                      student_best=student_best, diagnostics=diagnostics)
    torch.save(checkpoint, directory / 'checkpoint.pt')
    torch.save(probabilities.cpu(), directory / 'probabilities.pt')
    torch.save(teacher_logp.exp().cpu(), directory / 'teacher_probabilities.pt')
    return dict(best_epoch=student_best['epoch'], best_val=student_best['val'], parameters=deploy_parameters,
                teacher_best_epoch=teacher_best['epoch'], teacher_best_val=teacher_best['val'],
                config=config, diagnostics=diagnostics)
