"""Posthoc label-free view shift diagnostics; no parameter or target updates."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
import run_experiment as run
import core
from pingda import build_generators, sample_augmentation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'corafull'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    for path in sorted((ROOT/'runs/pretrain').glob('*/seed*/*/complete.json')):
        dataset, seed, arm = path.parents[2].name, int(path.parents[1].name[4:]), path.parent.name
        if dataset not in args.datasets or arm == 'grace':
            continue
        config = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))['base']
        state = torch.load(path.parent/'checkpoint.pt', map_location='cpu', weights_only=True)
        model = core.WeightedGRACE(run.load_graph(dataset, seed).x.shape[1], config['hidden_dim'],
                                  config['projection_dim'], config['temperature'], config['activation'])
        model.load_state_dict(state['model']); model.eval()
        graph = run.load_graph(dataset, seed)
        x, edges = run._row_normalize_features(graph.x), graph.edge_index
        edge_gen, feat_gen = build_generators(x.shape[1], 'cpu')
        edge_gen.load_state_dict(state['edge_generator']); feat_gen.load_state_dict(state['feature_generator'])
        head = None
        if arm == 'pingda_context':
            head = nn.Sequential(nn.Linear(config['hidden_dim'],128), nn.ReLU(), nn.Linear(128,128))
            head.load_state_dict(state['head']); head.eval()
        torch.manual_seed(190014)
        with torch.no_grad():
            h0 = model(x, edges)
            xa, weights = sample_augmentation(x, edges, edge_gen, feat_gen)
            ha = model(xa, edges, weights)
            c0, valid = core.neighbor_mean(h0, edges)
            ca, _ = core.neighbor_mean(ha, edges)
            if head is not None:
                c0, ca = head(c0), head(ca)
            g0, ga = core.geometry(h0), core.geometry(ha)
            v0, va = core.geometry(c0[valid]), core.geometry(ca[valid])
            row = dict(dataset=dataset, seed=seed, arm=arm,
                       h_clean_aug_cosine=float(F.cosine_similarity(h0,ha).mean()),
                       h_clean_std=g0['mean_feature_std'], h_aug_std=ga['mean_feature_std'],
                       h_clean_rank=g0['effective_rank'], h_aug_rank=ga['effective_rank'],
                       c_clean_std=v0['mean_feature_std'], c_aug_std=va['mean_feature_std'],
                       c_clean_rank=v0['effective_rank'], c_aug_rank=va['effective_rank'],
                       c_clean_aug_cosine=float(F.cosine_similarity(c0[valid],ca[valid]).mean()),
                       scope='one fixed stochastic augmented view, full-neighbor readout; posthoc no labels, not a causal intervention')
        rows.append(row)
    # Preserve other datasets when incrementally adding diagnostics.
    dest = ROOT/'runs/view_diagnostics.json'
    previous = json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else []
    rows = [row for row in previous if row['dataset'] not in args.datasets] + rows
    run.write(dest, rows)
    run.csv_write(ROOT/'runs/view_diagnostics.csv', rows)
    print(json.dumps(rows, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
