"""TINED official-core adaptation: sparse torch SAGE + AST official student.

Public run() requires CUDA (including ROCm's CUDA API). No graph is needed
by TINEDStudent.forward(x); graphs are used only for teacher training and DE.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parent / 'vendor' / 'tined'


def _load_official_sequence():
    source = VENDOR / 'models.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'MLP_from_sequence_of_layers']
    if len(classes) != 1:
        raise RuntimeError('Expected exactly one official TINED sequence class')
    namespace = {'__name__': __name__, 'torch': torch, 'nn': nn, 'F': F,
                 'np': np, 'Path': Path}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['MLP_from_sequence_of_layers']


_OfficialSequence = _load_official_sequence()


class SparseGraph:
    """Raw directed edge multiset: no symmetrization, filtering, or dense NxN."""
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, dtype=torch.float32):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        src, dst = edge_index
        values = torch.ones(src.numel(), dtype=dtype, device=src.device)
        self.adjacency = torch.sparse_coo_tensor(
            torch.stack((dst, src)), values, (num_nodes, num_nodes)).coalesce()
        self.degree = torch.bincount(dst, minlength=num_nodes).to(dtype)
        # The incidence operator computes edge differences without cancellation
        # in a Laplacian quadratic form, or storing an E x raw_feature_dim tensor.
        keep = src != dst
        src, dst = src[keep], dst[keep]
        rows = torch.arange(src.numel(), device=src.device)
        self.incidence = torch.sparse_coo_tensor(
            torch.stack((torch.cat((rows, rows)), torch.cat((src, dst)))),
            torch.cat((torch.ones_like(src, dtype=dtype), -torch.ones_like(dst, dtype=dtype))),
            (src.numel(), num_nodes)).coalesce()
        self.incidence_t = self.incidence.transpose(0, 1).coalesce()

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        # Official SAGEConv(gcn): (sum of incoming neighbors + self)/(degree+1).
        return (torch.sparse.mm(self.adjacency, x) + x) / (self.degree[:, None] + 1)


class _SparseEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, incidence, incidence_t, feature_chunk):
        norm = feature.norm(dim=1, keepdim=True).detach()
        norm = torch.where((norm == 0) | torch.isnan(norm), torch.ones_like(norm), norm)
        energy = feature.new_zeros(())
        for start in range(0, feature.shape[1], feature_chunk):
            z = feature[:, start:start + feature_chunk] / norm
            diff = torch.sparse.mm(incidence, z)
            energy += diff.square().sum()
        ctx.save_for_backward(feature, norm, incidence, incidence_t)
        ctx.feature_chunk = feature_chunk
        return energy / feature.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        feature, norm, incidence, incidence_t = ctx.saved_tensors
        gradient = torch.empty_like(feature)
        scale = grad_output * (2.0 / feature.shape[0])
        for start in range(0, feature.shape[1], ctx.feature_chunk):
            z = feature[:, start:start + ctx.feature_chunk] / norm
            diff = torch.sparse.mm(incidence, z)
            gradient[:, start:start + ctx.feature_chunk] = (
                torch.sparse.mm(incidence_t, diff) / norm) * scale
        return gradient, None, None, None


def dirichlet_energy(graph: SparseGraph, feature: torch.Tensor) -> torch.Tensor:
    feature = feature.reshape(feature.shape[0], -1)
    return _SparseEnergy.apply(feature, graph.incidence, graph.incidence_t, 64)


def _log_ratio(energy_in, energy_out, zero_value):
    tiny = torch.finfo(energy_in.dtype).tiny
    value = energy_out.clamp_min(tiny).log() - energy_in.clamp_min(tiny).log()
    return torch.where((energy_in == 0) | (energy_out == 0),
                       value.new_tensor(zero_value), value)


class _SAGEGCN(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.fc_neigh = nn.Linear(din, dout, bias=False)
        self.bias = nn.Parameter(torch.zeros(dout))
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=nn.init.calculate_gain('relu'))


class SAGETeacher(nn.Module):
    def __init__(self, dims, dropout=.5):
        super().__init__()
        self.dims = list(dims)
        self.layers = nn.ModuleList([_SAGEGCN(din, dout)
                                    for din, dout in zip(dims, dims[1:])])
        self.dropout = nn.Dropout(dropout)
        self.energy_degenerate_layers = []

    def _forward(self, graph, x, targets=False):
        h, ratios, degenerates = x, [], []
        for i, layer in enumerate(self.layers):
            din, dout = layer.fc_neigh.in_features, layer.fc_neigh.out_features
            operations = ('FT', 'GP') if din > dout else ('GP', 'FT')
            for operation in operations:
                before = h
                h = F.linear(h, layer.fc_neigh.weight) if operation == 'FT' else graph.propagate(h)
                if targets:
                    ein, eout = dirichlet_energy(graph, before), dirichlet_energy(graph, h)
                    degenerates.append(bool((ein == 0) | (eout == 0)))
                    ratios.append(_log_ratio(ein, eout, 0.0))
            # Official teacher layer_info excludes this deferred SAGE bias.
            h = h + layer.bias
            if i != len(self.layers) - 1:
                h = self.dropout(F.relu(h))
        if targets:
            self.energy_degenerate_layers = degenerates
            return h, torch.stack(ratios).detach()
        return h

    def forward(self, graph, x):
        return self._forward(graph, x)

    def forward_with_targets(self, graph, x):
        return self._forward(graph, x, targets=True)


class TINEDStudent(_OfficialSequence):
    """Official constructor/forward; replace file I/O and DGL DE only."""
    def __init__(self, teacher: SAGETeacher, config: dict):
        shapes = [(l.fc_neigh.in_features, l.fc_neigh.out_features) for l in teacher.layers]
        ga_shapes = [(min(din, dout), min(din, dout)) for din, dout in shapes]
        args = SimpleNamespace(load_disabled=False, DE_sampling_ratio=config['DE_sampling_ratio'],
                               DE_regularization=False)
        super().__init__(len(shapes), shapes, ga_shapes, teacher.dims[0], 0,
                         config['hidden'], teacher.dims[-1], config['dropout'],
                         norm_type=config['norm_type'], from_MLP_mode='same_as_teacher', args=args)
        reference = teacher.layers[0].fc_neigh.weight
        self.to(device=reference.device, dtype=reference.dtype)
        self.register_buffer('DE_targets', reference.new_zeros(len(self.layers)))
        # copy_ preserves Parameter registration/device and prevents teacher aliasing.
        with torch.no_grad():
            for source, target in zip(teacher.layers, self.learned_MLP_layers):
                target.weight.copy_(source.fc_neigh.weight)
                target.bias.copy_(source.bias)
            if config['GA_init_type'] == 'identity':
                for layer in self.graph_aggregation_layers:
                    layer.weight.copy_(torch.eye(layer.out_features, layer.in_features,
                                                 device=reference.device, dtype=reference.dtype))
                    layer.bias.zero_()

    def set_de_targets(self, targets):
        if targets.shape != self.DE_targets.shape or not torch.isfinite(targets).all():
            raise ValueError('Teacher DE targets must match the complete FT/GP sequence')
        self.DE_targets.copy_(targets.detach())

    def get_appoximated_DE(self, feats, graph, input_energy=None):
        h, ratios = feats, []
        for i, layer in enumerate(self.layers):
            before = h
            h = layer(h)
            ein = input_energy if i == 0 and input_energy is not None else dirichlet_energy(graph, before)
            eout = dirichlet_energy(graph, h)
            # Preserve official student's degenerate log-ratio=1 branch on-device.
            ratios.append(_log_ratio(ein, eout, 1.0))
            if i != len(self.layers) - 1:
                h = self.dropout(F.relu(self.norms[i](h)))
        return ratios


def resolve_config(config: dict, name: str) -> dict:
    aliases = {'amazonpho': 'a-photo', 'amazon-photo': 'a-photo', 'amazonphoto': 'a-photo'}
    dataset = aliases.get(name.lower(), name.lower())
    records = json.loads((VENDOR / 'best_records' / 'best_records.json').read_text(encoding='utf-8'))
    presets = records['SAGE-transductive']['ours']
    preset_name = dataset if dataset in presets and presets[dataset] else 'cora'
    preset = presets[preset_name]
    algorithm = {'lamb': float(preset['lamb']),
                 'DE_regularization_rate': float(preset['DE_regularization_rate']),
                 'DE_sampling_ratio': float(preset.get('DE_sampling_ratio', 1)),
                 'DE_log': preset['DE_log'].lower() == 'true',
                 'sqrt_DER': True, 'learned_MLP_lr_ratio': float(preset['learned_MLP_lr_ratio']),
                 'GA_init_type': preset['GA_init_type'], 'norm_type': preset['norm_type']}
    fixed = {'seed': 0, 'hidden': 64, 'dropout': .5, 'learning_rate': .01, 'weight_decay': .0005}
    for key, value in {**fixed, **algorithm}.items():
        if key in config and config[key] != value:
            raise ValueError(f'{key} is fixed to {value!r} under this protocol')
    for alias, canonical in {'hidden_dim': 'hidden', 'dropout_ratio': 'dropout',
                             'lr': 'learning_rate', 'num_layers': 'teacher_layers', 'layers': 'teacher_layers'}.items():
        if alias in config and config[alias] != (2 if canonical == 'teacher_layers' else fixed[canonical]):
            raise ValueError(f'{alias} conflicts with the fixed main protocol')
    for key in ['dw', 'position_encoding', 'positional_encoding', 'feat_distill', 'adv', 'pca']:
        if config.get(key, False) not in [False, None, 'none', 'False', 'false', 0]:
            raise ValueError(f'{key} is disabled for attribute-only TINED (not TINED+)')
    for key in ['from_learned_MLP', 'DE_regularization']:
        if key in config and config[key] not in [True, 'True', 'true', 1]:
            raise ValueError(f'{key} is required by TINED')
    for key in ['from_MLP_mode', 'DE_mode', 'teacher', 'teacher_layers', 'batch_size']:
        required = {'from_MLP_mode': 'same_as_teacher', 'DE_mode': 'same_as_teacher',
                    'teacher': 'SAGE', 'teacher_layers': 2, 'batch_size': int(preset['batch_size'])}[key]
        if key in config and config[key] != required:
            raise ValueError(f'{key} is fixed to {required!r}')
    requested_device = str(config.get('device', 'cuda'))
    if not requested_device.startswith('cuda'):
        raise ValueError('CUDA required; CPU fallback is disabled')
    base_epochs = config.get('epochs', 200)
    teacher_epochs = config.get('teacher_epochs', base_epochs)
    student_epochs = config.get('student_epochs', base_epochs)
    for value in [base_epochs, teacher_epochs, student_epochs]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Epoch budgets must be positive integers')
    effective = {**config, **fixed, **algorithm, 'teacher': 'SAGE', 'teacher_layers': 2,
                 'teacher_epochs': teacher_epochs, 'student_epochs': student_epochs,
                 'epochs': base_epochs, 'batch_size': int(preset['batch_size']),
                 'teacher_norm_type': 'none', 'author_preset': preset_name,
                 'preset_fallback': preset_name != dataset, 'dataset': name,
                 'from_MLP_mode': 'same_as_teacher', 'DE_mode': 'same_as_teacher',
                 'from_learned_MLP': True, 'DE_regularization': True,
                 'dw': False, 'position_encoding': False, 'feat_distill': False, 'adv': False,
                 'validation_selection': 'clean validation accuracy; last tie',
                 'cpu_fallback': False, 'early_stopping': False,
                 'teacher_training': 'full-neighbor sparse SAGE-gcn (not sampled fan_out=5,5)',
                 'student_updates': 'ordered CE batches, ordered all-node KL batches, separate DE step',
                 'non_main_budget': teacher_epochs != 200 or student_epochs != 200,
                 'author_original_base': {key: preset[key] for key in
                                          ['learning_rate', 'dropout_ratio', 'weight_decay', 'max_epoch', 'patience']}}
    json.dumps(effective, allow_nan=False)
    return effective


def validate_data(data, require_cuda=True):
    x, edges, y = data.x, data.edge_index, data.y
    if require_cuda and x.device.type != 'cuda':
        raise ValueError('CUDA data required; CPU fallback is disabled')
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or x.dtype not in [torch.float32, torch.float64]:
        raise ValueError('x must be a nonempty float32/float64 raw attribute matrix')
    if not torch.isfinite(x).all():
        raise ValueError('Non-finite raw attributes')
    n = x.shape[0]
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError('edge_index must be long with shape [2,E]')
    if edges.device != x.device or y.device != x.device:
        raise ValueError('All data tensors must already be on the same device')
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Invalid edge node index')
    if y.shape != (n,) or y.dtype != torch.long:
        raise ValueError('y must be a length-N long tensor')
    if isinstance(data.num_classes, bool) or not isinstance(data.num_classes, int) or data.num_classes < 2:
        raise ValueError('num_classes must be an explicit integer >= 2')
    counts = torch.zeros(n, dtype=torch.long, device=x.device)
    for split in ['train', 'val', 'test']:
        index, mask = getattr(data, f'{split}_index'), getattr(data, f'{split}_mask')
        if index.device != x.device or mask.device != x.device:
            raise ValueError('Indices and masks must be on the input device')
        if index.ndim != 1 or index.dtype != torch.long or not index.numel():
            raise ValueError(f'{split}_index must be a nonempty ordered long vector')
        if index.min() < 0 or index.max() >= n or index.unique().numel() != index.numel():
            raise ValueError(f'Invalid or duplicate {split}_index')
        if mask.shape != (n,) or mask.dtype != torch.bool:
            raise ValueError(f'Invalid {split}_mask')
        expected = torch.zeros_like(mask)
        expected[index] = True
        if not torch.equal(mask, expected):
            raise ValueError(f'{split} mask/index mismatch')
        counts[index] += 1
    # Historical percentage splits floor each size; leftovers stay unlabeled.
    if (counts > 1).any():
        raise ValueError('Parent train/val/test splits must be disjoint')
    if not (y[~data.train_mask] == -1).all():
        raise ValueError('All non-training labels must be -1; no clean test labels accepted')
    train_y = y[data.train_index]
    if train_y.min() < 0 or train_y.max() >= data.num_classes:
        raise ValueError('Training labels outside explicit num_classes')
    val_y = data.val_y
    if val_y.device != x.device or val_y.dtype != torch.long or val_y.shape != data.val_index.shape:
        raise ValueError('val_y must match ordered val_index on the input device')
    if val_y.min() < 0 or val_y.max() >= data.num_classes:
        raise ValueError('Validation labels outside explicit num_classes')


def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(0)


def _state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def select_checkpoint(model, best, epoch, val):
    if val >= best['val']:
        return _state(model), {'epoch': epoch, 'val': val}
    return None, best


def _step(optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite TINED objective; no silent fallback')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Non-finite TINED gradient')
    optimizer.step()


def _val(logits, data):
    return (float((logits.argmax(1) == data.val_y).double().mean()),
            float(F.cross_entropy(logits, data.val_y)))


def fit_teacher(model, graph, data, config, log: Callable):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, state = {'epoch': 0, 'val': -1.0}, None
    for epoch in range(1, config['teacher_epochs'] + 1):
        model.train()
        logits = model(graph, data.x)
        loss = F.cross_entropy(logits[data.train_index], data.y[data.train_index])
        _step(optimizer, loss)
        model.eval()
        with torch.no_grad():
            val, val_loss = _val(model(graph, data.x)[data.val_index], data)
        selected, best = select_checkpoint(model, best, epoch, val)
        if selected is not None:
            state = selected
        log({'phase': 'teacher', 'epoch': epoch, 'loss': float(loss.detach()),
             'ce': float(loss.detach()), 'val': val, 'val_loss': val_loss,
             'selected': selected is not None})
    return state, best


def de_loss(log_ratios, targets, config):
    power = .5 if config['sqrt_DER'] else 1.0
    actual, target = power * log_ratios, power * targets
    if not config['DE_log']:
        actual, target = actual.exp(), target.exp()
    return (actual - target).square().sum() * config['DE_regularization_rate']


def _de_inputs(graph, x, ratio):
    if ratio == 1:
        return graph, x
    count = int(graph.edge_index.shape[1] * ratio)
    if count < 1:
        raise ValueError('Fixed author DE sampling ratio yields zero edges')
    chosen = torch.randperm(graph.edge_index.shape[1], device=x.device)[:count]
    edges = graph.edge_index[:, chosen]
    nodes = torch.unique(edges.flatten(), sorted=True)
    remap = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    remap[nodes] = torch.arange(nodes.numel(), device=x.device)
    return SparseGraph(remap[edges], nodes.numel(), x.dtype), x[nodes]


def _student_logits(model, x, batch_size):
    return torch.cat([model(x[start:start + batch_size])[1]
                      for start in range(0, x.shape[0], batch_size)], dim=0)


def fit_student(model, graph, data, teacher_logp, config, log: Callable):
    injected = list(model.learned_MLP_layers.parameters())
    injected_ids = {id(p) for p in injected}
    other = [p for p in model.parameters() if id(p) not in injected_ids]
    optimizer = torch.optim.Adam([
        {'params': other, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},
        {'params': injected, 'lr': config['learning_rate'] * config['learned_MLP_lr_ratio'],
         'weight_decay': config['weight_decay']}])
    best, state = {'epoch': 0, 'val': -1.0}, None
    size, lamb = config['batch_size'], config['lamb']
    # Raw attributes and the full DE graph never change. Reuse their scalar
    # energy, not graph-derived features; this is exactly the same objective.
    input_energy = (dirichlet_energy(graph, data.x).detach()
                    if config['DE_sampling_ratio'] == 1 else None)
    _seed()
    for epoch in range(1, config['student_epochs'] + 1):
        model.train()
        ce_values, kd_values = [], []
        # Even lambda=1 keeps official zero-weight CE optimizer steps, including
        # Adam weight decay/momentum effects. Do not combine these three stages.
        for indices in data.train_index.split(size):
            logits = model(data.x[indices])[1]
            ce = F.cross_entropy(logits, data.y[indices])
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
            logits = _student_logits(model, data.x[data.val_index], size)
            val, val_loss = _val(logits, data)
        selected, best = select_checkpoint(model, best, epoch, val)
        if selected is not None:
            state = selected
        ce, kl, de = sum(ce_values) / len(ce_values), sum(kd_values) / len(kd_values), float(energy_loss.detach())
        log({'phase': 'student', 'epoch': epoch, 'loss': (1 - lamb) * ce + lamb * kl + de,
             'ce': ce, 'kl': kl, 'de': de, 'val': val, 'val_loss': val_loss,
             'log_de_ratios': ratios.detach().cpu().tolist(), 'selected': selected is not None})
    return state, best


def _provenance():
    paths = ['models.py', 'utils_gnn_property.py', 'train_and_eval.py', 'train_student.py',
             'tran.conf.yaml', 'best_records/best_records.json', 'tasks/3_run_all_TINED.py']
    return {name: hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() for name in paths}


def _cpu_state(state):
    return {key: value.detach().cpu() for key, value in state.items()}


def run(data, config, outdir) -> dict:
    validate_data(data)
    effective = resolve_config(dict(config), str(data.name))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is disabled')
    directory = Path(outdir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = [directory / name for name in ['probabilities.pt', 'checkpoint.pt', 'epochs.jsonl']]
    if any(path.exists() for path in artifacts):
        raise FileExistsError('TINED output already contains artifacts; use a fresh outdir')
    _seed()
    graph = SparseGraph(data.edge_index, data.x.shape[0], data.x.dtype)
    dims = [data.x.shape[1], effective['hidden'], data.num_classes]
    teacher = SAGETeacher(dims, effective['dropout']).to(device=data.x.device, dtype=data.x.dtype)
    with (directory / 'epochs.jsonl').open('x', encoding='utf-8') as handle:
        def log(row):
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()

        teacher_state, teacher_best = fit_teacher(teacher, graph, data, effective, log)
        teacher.load_state_dict(teacher_state)
        teacher.eval()
        with torch.no_grad():
            logits, targets = teacher.forward_with_targets(graph, data.x)
            teacher_logp = logits.log_softmax(1).detach()
        teacher.requires_grad_(False)
        _seed()
        student = TINEDStudent(teacher, effective)
        student.set_de_targets(targets)
        student_state, student_best = fit_student(student, graph, data, teacher_logp, effective, log)
    student.load_state_dict(student_state)
    student.eval()
    with torch.no_grad():
        probabilities = _student_logits(student, data.x, effective['batch_size']).softmax(1)
        selected_val = float((probabilities[data.val_index].argmax(1) == data.val_y).double().mean())
    if selected_val != student_best['val'] or not torch.isfinite(probabilities).all():
        raise RuntimeError('Selected checkpoint probability/validation consistency failure')
    if not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-5):
        raise RuntimeError('Invalid node probabilities')
    teacher_parameters = sum(p.numel() for p in teacher.parameters())
    all_student_parameters = sum(p.numel() for p in student.parameters())
    deploy_parameters = (sum(p.numel() for p in student.layers.parameters())
                         + sum(p.numel() for norm in student.norms[:-1] for p in norm.parameters()))
    diagnostics = {'implementation': 'torch sparse SAGE port + AST official sequence constructor/forward',
                   'direct_official_execution': False, 'source_sha256': _provenance(),
                   'device': str(data.x.device), 'torch': str(torch.__version__),
                   'teacher_best_epoch': teacher_best['epoch'], 'teacher_best_val': teacher_best['val'],
                   'teacher_parameters': teacher_parameters, 'student_parameters_including_unused_heads': all_student_parameters,
                   'student_deploy_parameters': deploy_parameters, 'student_fc_layers': len(student.layers),
                   'sequence': student.transformation_types_pseudo,
                   'ft_shapes_in_out': [[l.in_features, l.out_features] for l in student.learned_MLP_layers],
                   'gp_shapes_in_out': [[l.in_features, l.out_features] for l in student.graph_aggregation_layers],
                   'teacher_log_de_targets': targets.cpu().tolist(),
                   'teacher_zero_energy_layers': teacher.energy_degenerate_layers,
                   'author_preset': effective['author_preset'], 'preset_fallback': effective['preset_fallback'],
                   'train_supervision': 'noisy y[train_index] only; val_y is selection/evaluation only',
                   'student_kl_nodes': data.x.shape[0], 'test_labels_read': False,
                   'deployment_input': 'raw x only', 'graph_dense_n_by_n': False,
                   'energy_feature_chunk': 64, 'no_cpu_fallback': True,
                   'raw_input_energy_cached': effective['DE_sampling_ratio'] == 1,
                   'sampling_rng': 'torch GPU randperm instead of upstream numpy choice',
                   'budget': {'teacher_epochs': effective['teacher_epochs'], 'student_epochs': effective['student_epochs']}}
    checkpoint = {'teacher_state': _cpu_state(teacher_state), 'student_state': _cpu_state(student_state),
                  'state': _cpu_state(student_state), 'config': effective,
                  'teacher_config': {'dims': dims, 'dropout': effective['dropout'], 'norm_type': 'none'},
                  'student_config': {**effective, 'input_dim': dims[0], 'output_dim': dims[-1],
                                     'sequence': student.transformation_types_pseudo},
                  'teacher_best': teacher_best, 'student_best': student_best, 'diagnostics': diagnostics}
    torch.save(checkpoint, directory / 'checkpoint.pt')
    torch.save(probabilities.cpu(), directory / 'probabilities.pt')
    return {'best_epoch': student_best['epoch'], 'best_val': student_best['val'],
            'parameters': deploy_parameters, 'config': effective,
            'probabilities_file': str(directory / 'probabilities.pt'),
            'checkpoint_file': str(directory / 'checkpoint.pt'), 'diagnostics': diagnostics}
