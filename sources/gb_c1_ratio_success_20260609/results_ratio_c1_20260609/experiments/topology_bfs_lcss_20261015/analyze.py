"""Apply preregistered training gate using frozen training-target diagnostics only."""
from pathlib import Path
import json
import pandas as pd
import run_screen as screen

ROOT = screen.ROOT


def main():
    audit = screen.read(ROOT/'audit/report.json')
    assert audit['status'] == 'passed' and len(audit['cases']) == 15
    paired = pd.read_csv(ROOT/'paired_target_deltas.csv')
    gates = []
    for dataset in screen.NAMES:
        checks = []
        for noise in ['uniform', 'pair']:
            for reference in ['CE', 'RANDOM']:
                values = paired[(paired.dataset == dataset) & (paired.noise == noise)
                                & (paired.reference == reference)].brier_delta
                assert len(values) == 5
                mean = float(values.mean())
                improved = int((values < 0).sum())
                checks.append(dict(noise=noise, reference=reference, mean_brier_delta=mean,
                    sample_std=float(values.std(ddof=1)), improved_seeds=improved,
                    passed=mean <= -.005 and improved >= 4))
        gates.append(dict(dataset=dataset, proceed_to_students=all(c['passed'] for c in checks), checks=checks))
    decision = dict(stage='target_screen', protocol_audit='passed', datasets=gates,
        selected_datasets=[g['dataset'] for g in gates if g['proceed_to_students']],
        selection_uses='training clean-label diagnostic only; no test accuracy',
        threshold_mean_brier=-.005, threshold_improved_seeds=4,
        analyzer_source_hash=screen.d.sha(__file__), plan_hash=screen.d.sha(ROOT/'PLAN.md'))
    screen.d.write(ROOT/'gate.json', decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
