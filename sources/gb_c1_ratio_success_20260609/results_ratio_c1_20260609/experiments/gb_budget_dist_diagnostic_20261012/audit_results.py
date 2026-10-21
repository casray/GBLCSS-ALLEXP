from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import weighted_dist as wd


def main():
    spec = importlib.util.spec_from_file_location('budget_diagnostic_runner', ROOT/'diagnose.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.verify()
    runner.knowledge.verify()
    torch.manual_seed(12)
    features = torch.randn(41, 8, dtype=torch.float64)
    head = torch.nn.Linear(8, 5, dtype=torch.float64)
    teacher = torch.randn(41, 5, dtype=torch.float64).softmax(1)
    weights = torch.full((41,), 1/41, dtype=torch.float64)
    logits = head(features)
    value = wd.weighted_dist(logits, teacher, weights, 41)
    reference = torch.autograd.grad(value, (head.weight, head.bias))
    reference = torch.cat([r.flatten() for r in reference])
    _, calculated = runner.head_gradient(logits, teacher, weights, features, 41)
    torch.testing.assert_close(calculated, reference, rtol=1e-12, atol=1e-12)
    with (ROOT/'replicates.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14400
    cases = {(r['dataset'], r['rate'], r['seed']) for r in rows}
    assert len(cases) == 20
    identities = [tuple(r[f] for f in ('dataset', 'rate', 'seed', 'state', 'fraction', 'arm', 'repeat')) for r in rows]
    assert len(set(identities)) == len(rows)
    for row in rows:
        for field in ('loss', 'full_loss', 'loss_error', 'grad_relative_mse', 'gradient_cosine'):
            assert math.isfinite(float(row[field]))
        assert float(row['grad_relative_mse']) >= 0
        assert -1.000001 <= float(row['gradient_cosine']) <= 1.000001
        assert math.isclose(float(row['loss_error']), float(row['loss'])-float(row['full_loss']), abs_tol=1e-7)
    allocations = {}
    with (ROOT/'allocations.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[f] for f in ('dataset', 'rate', 'seed', 'fraction', 'arm'))
            nodes, count = int(row['nodes']), int(row['allocated'])
            assert 1 <= count <= nodes
            allocations.setdefault(key, [0, 0])
            allocations[key][0] += nodes
            allocations[key][1] += count
    assert len(allocations) == 360
    for key, (n, count) in allocations.items():
        assert count == int(n*float(key[3]))
    for row in rows:
        key = tuple(row[f] for f in ('dataset', 'rate', 'seed', 'fraction', 'arm'))
        assert int(row['nodes_sampled']) == allocations[key][1]
    summary = json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    assert not summary['proceed_training'] and not any(summary['gates'].values())
    result = dict(status='passed', metric_rows=len(rows), protocol_cases=len(cases),
                  allocation_configurations=len(allocations),
                  inputs_and_frozen_code_unchanged=True,
                  head_gradient_matches_autograd=True,
                  gradient_max_abs_error=float((calculated-reference).abs().max()),
                  audit_scope='finite values, unique rows, budgets, head-gradient chain rule, input hashes')
    (ROOT/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for directory in (ROOT, runner.knowledge.ROOT):
        manifest = {str(path.relative_to(directory)): runner.knowledge.sha(path)
                    for path in sorted(directory.rglob('*'))
                    if path.is_file() and '__pycache__' not in path.parts
                    and path.name != 'output_manifest.json'}
        runner.knowledge.write(directory/'output_manifest.json', manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
