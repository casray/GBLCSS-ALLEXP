from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from run import ROOT, DATASETS, load_data, audit, final_metrics, write_json


def main():
    torch.set_num_threads(4)
    rows, audits = [], {}
    for dataset in DATASETS:
        folder = ROOT / 'runs' / dataset / 'uniform_0.4' / 'seed0'
        record = json.loads((folder / 'result.json').read_text())
        if record['status'] != 'ok' or record['smoke'] or record['seed'] != 0 or record['noise_rate'] != .4:
            raise RuntimeError('Unexpected screening identity')
        data, protocol = load_data(dataset, 0, .4)
        audits[dataset] = audit(folder, data, record, protocol)
        saved = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        metrics = final_metrics(dataset, 0, saved)
        if metrics['test'] != record['test'] or metrics['valid'] != record['valid']:
            raise RuntimeError('Final metrics mismatch')
        row = dict(dataset=dataset, noise_type='uniform', noise_rate=.4, seed=0,
                   test_percent=100*record['test'], valid_percent=100*record['valid'],
                   best_epoch=record['best_epoch'], best_phase=record['diagnostics']['best_phase'],
                   inference_parameters=record['parameters'],
                   wall_seconds=record['wall_seconds'], peak_allocated_gib=record['peak_allocated_gib'])
        rows.append(row)
        del data, saved
        torch.cuda.empty_cache()
    with (ROOT / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / 'artifact_audit.json', audits)
    names = dict(cora='Cora', citeseer='CiteSeer', corafull='CoraFull',
                 coauthorcs='Coauthor-CS', amazonpho='Amazon-Photo')
    lines = ['# GSSC common-backbone adaptation: single-seed screening', '',
             '## Protocol and changes', '',
             'Five main datasets; uniform noise 0.4; split/noise/downstream seed 0;20% train,10% validation,70% test. Raw features and original graph. Ordered splits/noise match historical main fingerprints and original NoisyGL noise generation. Clean test labels enter only final evaluation.',
             '', 'Deployed backbone: exact common two-FC MLP, hidden64, ReLU-BN-dropout0.5. Native separate auxiliary BN-LeakyReLU-dropout-linear head remains training-only. Adam lr0.01, weight decay0.0005; native mixup lr0.0001.',
             '', '100 label-free edge pretraining +100 classifier warmup +100 main epochs. First clean-validation maximum over the 200 supervised epochs, including warmup. Native per-edge minibatches, class-center extra steps and sampler/mixup training remain: matched supervised epochs/backbone is not matched total optimizer steps or training cost.',
             '', 'Preserved: edge sampler, straight-through gradients, virtual-node mixup, positive/negative self-contrast, noisy-train-only CE and class weights, class-center loss and first-order homophily updates. Preset mapping remains Cora/CiteSeer/Coauthor-CS native; CoraFull/Amazon-Photo Cora fallback. No test-based search.',
             '', 'Frozen historical core is unchanged. An isolated AST copy changes only last-tie to first-tie and two inference-parameter counts; adapter records the source hash and asserts transformation counts. Emitted adapted_training_core.py is saved per run for inspection.',
             '', '## Results', '',
             '| Dataset | Test accuracy (%) | Best clean validation (%) | Selected supervised epoch | Phase | Time (s) |',
             '|---|---:|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {names[row['dataset']]} | {row['test_percent']:.2f} | {row['valid_percent']:.2f} | {row['best_epoch']} | {row['best_phase']} | {row['wall_seconds']:.1f} |")
    lines += ['', '## Verification and conclusion', '',
              'Four new adaptation tests and thirteen prior GSSC core tests passed. The optional old synthetic GPU test was skipped; the new real-Cora short GPU smoke passed. Every formal run completed all300 phase epochs (200 supervised), selected the first validation maximum, and reloaded its deployed checkpoint with zero absolute probability difference. Artifact/source hashes and split/noise identities were rechecked after training. Non-training optimization labels are masked to -1.',
              '', 'Conclusion: GSSC now runs under the approved common deployed backbone and supervised-epoch budget, retaining its training mechanisms. These five numbers are single-seed screening results, not five-seed mean/std; they should not directly replace a main-table five-seed row. This experiment verifies the configured port and integration, not universal bug-freedom or bitwise parity with the old official DGL/Pyro environment. No other methods or GB-LCSS runs were repeated; no manuscript edits.',
              '', 'Decision: screening complete; retain all outcomes. A main-table five-seed extension requires explicit authorization.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
