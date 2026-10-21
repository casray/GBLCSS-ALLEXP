from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'new_graph_mlp_seed0_20261009'))
from common import DATASETS, final_metrics, load_data, write_json
from matched_adapter import build_model, state_hash


def main():
    torch.set_num_threads(4)
    rows, audits = [], []
    for dataset in DATASETS:
        folder = ROOT / 'runs' / dataset / 'simmlp_matched'
        result = json.loads((folder / 'result.json').read_text())
        cfg = json.loads((folder / 'config.json').read_text())
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
        probabilities = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        data, protocol = load_data(dataset, device='cpu')
        model, head, _ = build_model(data, cfg)
        model.load_state_dict(checkpoint['model'])
        head.load_state_dict(checkpoint['classifier'])
        model.eval()
        head.eval()
        with torch.no_grad():
            reloaded = head(model.encode(data.x)).softmax(1)
        torch.testing.assert_close(reloaded, probabilities, rtol=1e-4, atol=2e-6)
        metrics = final_metrics(dataset, probabilities)
        reloaded_metrics = final_metrics(dataset, reloaded)
        for key in ('valid', 'test'):
            if abs(metrics[key] - result[key]) > 1e-10 or metrics[key] != reloaded_metrics[key]:
                raise ValueError(f'{dataset}: metric reload mismatch')
        if state_hash(model.encoder) != checkpoint['frozen_encoder_hash']:
            raise ValueError(f'{dataset}: frozen encoder hash mismatch')
        records = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
        for stage, epochs in [('pretrain', cfg['epochs']), ('probe', cfg['lr_epochs'])]:
            stage_records = [record for record in records if record['stage'] == stage]
            if [r['epoch'] for r in stage_records] != list(range(1, epochs + 1)):
                raise ValueError(f'{dataset}: incomplete {stage} log')
        probe = [record for record in records if record['stage'] == 'probe']
        best = max(probe, key=lambda record: record['valid'])
        if best['epoch'] != result['best_epoch'] or best['epoch'] != checkpoint['best_epoch']:
            raise ValueError(f'{dataset}: first validation maximum mismatch')
        for key in ('split_hash', 'noise_hash'):
            if protocol[key] != result[key]:
                raise ValueError(f'{dataset}: {key} mismatch')
        hashes = json.loads((folder / 'source_hashes.json').read_text())
        for filename, digest in hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{dataset}: changed source {filename}')
        old = json.loads((ROOT.parent / 'simmlp_tined_audit_seed0_20261009' /
                          'runs' / dataset / 'simmlp/result.json').read_text())
        rows.append(dict(dataset=dataset, seed=0, noise='uniform 0.4',
                         test_percent=100*result['test'], valid_percent=100*result['valid'],
                         previous_test_percent=100*old['test'],
                         change_points=100*(result['test']-old['test']),
                         best_epoch=result['best_epoch'], deployed_parameters=result['parameters'],
                         previous_deployed_parameters=old['parameters'],
                         wall_seconds=result['wall_seconds'],
                         peak_allocated_gib=result['peak_allocated_gib']))
        audits.append(dict(dataset=dataset, status='passed',
                           checkpoint_reload=True, first_validation_maximum=True,
                           source_hashes=True, main_protocol_hashes=True,
                           common_parameter_count=result['parameters']==result['common_mlp_parameters'],
                           frozen_encoder_hash=True,
                           max_probability_reload_error=float((reloaded-probabilities).abs().max())))
    with (ROOT / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / 'artifact_audit.json', audits)
    print(json.dumps(dict(results=rows, audit=audits), indent=2), flush=True)


if __name__ == '__main__':
    main()
