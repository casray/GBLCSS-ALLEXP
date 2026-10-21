"""Common deployed MLP with frozen GSSC auxiliary objectives and training core."""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
OLD = ROOT.parent / 'new_graph_mlp_seed0_20261009'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(WORKSPACE / 'GB-LCSS_GitHub/src'))
import gssc_adapter as core
from gblcss.models import MLP

CORE_SHA256 = 'bd9ed5ee5e5f43e08f91633f7f3f99a5494447c77f10f547570701c5bf685754'
if hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() != CORE_SHA256:
    raise RuntimeError('Frozen GSSC core changed; review before running')
ORIGINAL_CLASS = core.NeighContrast


def effective_config(dataset, smoke=False, seed=0):
    if isinstance(seed, bool) or seed not in range(5):
        raise ValueError('Seed must be 0..4')
    cfg = core.resolve_config(dataset, dict(epochs=1 if smoke else 100,
                              ep_epochs=1 if smoke else 100,
                              cl_epochs=1 if smoke else 100, smoke=smoke))
    cfg.update(method='gssc_matched', seed=seed, deployed_fc_layers=2,
               validation_selection='clean validation accuracy; first tie',
               validation_candidate_phases=['classifier_pretrain', 'main'],
               adaptations=['common two-FC MLP: Linear-ReLU-BN-dropout-Linear',
                            '100 warmup + 100 main supervised epochs',
                            'first validation maximum over warmup and main'],
               auxiliary_head='native BN-LeakyReLU-dropout-Linear; training only',
               core_sha256=CORE_SHA256,
               compute_equivalence=False)
    return cfg


class MatchedContrast(nn.Module):
    def __init__(self, in_dim, hidden, classes, config):
        super().__init__()
        self.backbone = MLP(in_dim, hidden, classes, 2, config['dropout'])
        self.inf = core._fc(hidden, classes, config['dropout'])

    def encode(self, x):
        return self.backbone.layers[0](x)

    def output(self, h):
        # Non-inplace ReLU leaves the native auxiliary head's input unchanged.
        h = self.backbone.batch_norms[0](F.relu(h))
        h = F.dropout(h, p=self.backbone.dropout, training=self.training)
        return self.backbone.layers[-1](h)

    def forward(self, x):
        h = self.encode(x)
        return self.output(h), self.inf(h)

    def update(self, x):
        return self.output(self.encode(x)).argmax(1)

    def inference_parameters(self):
        return sum(p.numel() for p in self.backbone.parameters())


def checked_config(name, config):
    expected = effective_config(name, bool(config.get('smoke', False)), config.get('seed', 0))
    if config != expected:
        raise ValueError('Configuration differs from the approved fixed adaptation')
    return dict(expected)


class _AdaptRun(ast.NodeTransformer):
    def __init__(self):
        self.counts = dict(first_tie=0, inference_diagnostic=0, inference_return=0)

    def visit_Compare(self, node):
        if ast.unparse(node) == 'val >= best_val':
            node.ops = [ast.Gt()]
            self.counts['first_tie'] += 1
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == 'inference_parameters':
            assert ast.unparse(node.value) == "parameters_by_module['model']"
            node.value = ast.parse('model.inference_parameters()', mode='eval').body
            self.counts['inference_diagnostic'] += 1
        elif node.arg == 'parameters':
            assert ast.unparse(node.value) == 'sum(parameters_by_module.values())'
            node.value = ast.parse('model.inference_parameters()', mode='eval').body
            self.counts['inference_return'] += 1
        return self.generic_visit(node)


# Isolated namespace: historical globals/source stay untouched. Losses/loops are reused.
tree = ast.parse(inspect.getsource(core.run))
transform = _AdaptRun()
tree = ast.fix_missing_locations(transform.visit(tree))
TRANSFORM_COUNTS = transform.counts
if TRANSFORM_COUNTS != dict(first_tie=1, inference_diagnostic=1, inference_return=1):
    raise RuntimeError('Unexpected core transformation boundaries')
TRAINING_SOURCE = ast.unparse(tree)
namespace = dict(vars(core))
namespace.update(NeighContrast=MatchedContrast, resolve_config=checked_config)
exec(compile(tree, str(ROOT / 'adapted_training_core.py'), 'exec'), namespace)
run = namespace['run']
