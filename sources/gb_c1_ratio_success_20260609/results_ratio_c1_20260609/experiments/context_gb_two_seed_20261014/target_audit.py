"""Independently recompute frozen LOO targets; no training modules imported."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit import Auditor, fingerprint

ROOT = Path(__file__).resolve().parent


def loo_targets(labels, assignment, train, classes):
    """Vectorized histogram formula, independent of the training implementation."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    train = torch.as_tensor(train, dtype=torch.long)
    assignment = torch.as_tensor(assignment, dtype=torch.long)
    targets = F.one_hot(labels, classes).float()
    alpha = torch.zeros(len(labels))
    if len(train) == 0:
        return targets, alpha
    _, group = torch.unique(assignment, sorted=True, return_inverse=True)
    hist = torch.zeros((int(group.max()) + 1, classes), dtype=torch.long)
    hist.index_add_(0, group[train], F.one_hot(labels[train], classes))
    counts = hist[group[train]] - F.one_hot(labels[train], classes)
    total = counts.sum(1)
    maximum = counts.max(1).values
    denominator = total.clamp_min(1).float()
    strength = ((maximum / denominator - 1 / classes) / (1 - 1 / classes)).clamp(0, 1)
    strength = torch.where(total >= 2, strength, 0.)
    distribution = counts.float() / denominator[:, None]
    tied = (counts == maximum[:, None]).sum(1) > 1
    majority = F.one_hot(counts.argmax(1), classes).float()
    local = torch.where(tied[:, None], distribution, .5 * (distribution + majority))
    targets[train] = (1 - strength[:, None]) * targets[train] + strength[:, None] * local
    alpha[train] = strength
    return targets, alpha


def clean_evidence(labels, assignment, train, classes):
    """Return per-node OTHER training-voter counts and clean-class margins."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    train = torch.as_tensor(train, dtype=torch.long)
    _, group = torch.unique(torch.as_tensor(assignment), sorted=True, return_inverse=True)
    onehot = F.one_hot(labels, classes)
    hist = torch.zeros((int(group.max()) + 1, classes), dtype=torch.long)
    hist.index_add_(0, group[train], onehot[train])
    counts = hist[group[train]] - onehot[train]
    total = counts.sum(1)
    own = counts.gather(1, labels[train, None]).flatten()
    counts.scatter_(1, labels[train, None], -1)
    margin = (own-counts.max(1).values) / total.clamp_min(1)
    support_out, margin_out = torch.zeros(len(labels), dtype=torch.long), torch.zeros(len(labels))
    support_out[train], margin_out[train] = total, margin
    return support_out, margin_out


def run(partial=False):
    torch.set_num_threads(4)
    output = ROOT / 'runs'
    auditor = Auditor(output, partial=partial)
    if not auditor._load_config():
        raise ValueError(auditor.errors)
    checked = 0
    max_error = 0.
    errors = []
    evidence_rows = []
    for dataset in auditor.datasets:
        for seed in auditor.seeds:
            info = auditor._graph_info(dataset, seed)
            if info is None:
                continue
            train = torch.as_tensor(info['train'])
            classes = int(info['clean'].max()) + 1
            for noise in auditor.noise_rates:
                noisy, _ = auditor._noise_for(info, dataset, seed, noise)
                for space in auditor.spaces:
                    for budget in auditor.budgets:
                        folder = output/'targets'/dataset/f'seed{seed}'/noise/space/f'B{budget}'
                        path = folder/'targets.pt'
                        if not path.is_file():
                            if not partial:
                                errors.append(f'missing target: {path}')
                            continue
                        partition_dir = output/'partitions'/dataset/f'seed{seed}'
                        assignment = np.load(partition_dir/space/f'B{budget}'/'assignment.npy')
                        artifact = torch.load(path, map_location='cpu', weights_only=True)
                        if space in ('context_r', 'pingda_mean_r'):
                            arm = 'pingda_context' if space == 'context_r' else 'pingda'
                            base_space = 'context_h' if space == 'context_r' else 'pingda_h'
                            rep = torch.load(output/'pretrain'/dataset/f'seed{seed}'/arm/'representation.pt',
                                             map_location='cpu', weights_only=True)
                            valid = rep['structural_valid'].bool() & (rep['h'].norm(dim=1)>1e-8) & (rep['c'].norm(dim=1)>1e-8)
                            partition_info = json.loads((partition_dir/space/f'B{budget}'/'summary.json').read_text(encoding='utf-8'))
                            if partition_info['validity_hash'] != fingerprint(valid):
                                errors.append(f'partition validity hash mismatch: {folder}')
                            base = np.load(partition_dir/base_space/f'B{budget}'/'assignment.npy')
                            expected, alpha = loo_targets(noisy, base, train, classes)
                            support, margin = clean_evidence(info['clean'], base, train, classes)
                            indices = torch.nonzero(valid).flatten()
                            train_mask = torch.zeros(len(noisy), dtype=torch.bool)
                            train_mask[train] = True
                            local_train = torch.nonzero(train_mask[indices]).flatten()
                            local, local_alpha = loo_targets(noisy[indices], assignment, local_train, classes)
                            expected[indices], alpha[indices] = local, local_alpha
                            local_support, local_margin = clean_evidence(info['clean'][indices], assignment, local_train, classes)
                            support[indices], margin[indices] = local_support, local_margin
                        else:
                            expected, alpha = loo_targets(noisy, assignment, train, classes)
                            support, margin = clean_evidence(info['clean'], assignment, train, classes)
                        error = max(float((expected-artifact['targets']).abs().max()),
                                    float((alpha-artifact['alpha']).abs().max()))
                        max_error = max(max_error, error)
                        if error > 2e-7:
                            errors.append(f'LOO target/alpha mismatch ({error}): {folder}')
                        brier = float((artifact['targets'][train].double()-F.one_hot(info['clean'][train],classes)).square().sum(1).mean())
                        metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
                        if abs(brier-metrics['soft_target_brier']) > 1e-10:
                            errors.append(f'Brier disagrees with saved target: {folder}')
                        evidence_rows.append(dict(dataset=dataset, seed=seed, noise=noise, space=space, budget=budget,
                            soft_target_brier=brier, voting_support_coverage=float((support[train]>=2).float().mean()),
                            positive_clean_vote_margin_coverage=float(((support[train]>=2)&(margin[train]>0)).float().mean()),
                            mean_other_train_votes=float(support[train].float().mean()),
                            fallback_nodes=metrics['fallback_nodes'], scope='exact voting sets including full-H fallback'))
                        checked += 1
    expected_count = len(auditor.datasets)*len(auditor.seeds)*len(auditor.noise_rates)*len(auditor.spaces)*len(auditor.budgets)
    errors.extend(auditor.errors)
    result = dict(partial=partial, checked_targets=checked, expected_targets=expected_count,
                  coverage_complete=checked==expected_count, max_absolute_error=max_error, errors=errors)
    dest = output/'audit'/'target_formula_report.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if evidence_rows:
        with (dest.parent/'exact_evidence.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(evidence_rows[0]))
            writer.writeheader(); writer.writerows(evidence_rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    raise SystemExit(run(parser.parse_args().partial))
