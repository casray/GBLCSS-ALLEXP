"""Check the complete matrix and export paired supplementary tables."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from run_experiment import HERE, arms, snapshot, validate_pairs


def main():
    out = HERE/'gpu_reference_matched'
    cfg = json.loads((HERE/'config.json').read_text(encoding='utf8'))
    source, _ = snapshot(cfg)
    frame = pd.read_csv(out/'per_seed_results.csv')
    expected = {(d, s, a['name']) for d in cfg['datasets'] for s in cfg['seeds'] for a in arms(cfg)}
    observed = set(frame[['dataset', 'seed', 'arm']].itertuples(index=False, name=None))
    assert observed == expected and len(frame) == len(expected), 'incomplete or duplicated matrix'
    validate_pairs(frame.to_dict('records'))
    assert frame.status.eq('complete').all()
    assert frame.device.eq('cuda:0').all() and frame.gpu_peak_allocated_mib.gt(0).all()
    assert frame.epochs.eq(200).all()
    assert np.allclose(frame.brier_gain, frame.corrupted_gain-frame.clean_cost, atol=1e-7)
    reference = frame[frame.arm.eq('default')]
    assert reference.default_archive_brier_delta.abs().max() < 2e-5
    same_region = frame[frame.arm.isin(['default','beta_0','beta_0.25','beta_0.75','beta_1',
                                      'min_train_per_ball_5','min_train_per_ball_10'])]
    assert same_region.groupby(['dataset','seed']).assignment_hash.nunique().eq(1).all()
    for (d, s), rows in frame.groupby(['dataset','seed']):
        r = rows.set_index('arm')
        assert r.loc['raw_x','target_hash'] == r.loc['a2x','target_hash']
        assert r.loc['raw_x','test_accuracy'] == r.loc['a2x','test_accuracy']
        assert r.loc['raw_x','num_balls'] == r.loc['a2x','num_balls'] == 1
        for name in rows.arm:
            p = out/'runs'/d/f'seed{s}'/(name+'.npz')
            with np.load(p) as arrays:
                assert np.isfinite(arrays['targets']).all()
                assert np.allclose(arrays['targets'].sum(1),1,atol=1e-6)
            reuse = r.loc[name,'reused_from']
            if pd.notna(reuse):
                assert r.loc[name,'target_hash'] == r.loc[reuse,'target_hash']
            else:
                loss = pd.read_csv(p.with_name(p.stem+'_loss.csv'))
                assert len(loss) == 200 and np.isfinite(loss.loss).all()
                assert p.with_name(p.stem+'_predictions.npz').exists()

    archive = pd.read_csv(source/'results/main_tables/table2_per_seed.csv')
    archive = archive[(archive.noise_type=='uniform') & (archive.noise_rate==.4)]
    check = frame[frame.arm.isin(['default','mlp_ce'])].copy()
    check['method'] = check.arm.map({'default':'GB-LCSS','mlp_ce':'MLP+CE'})
    check = check.merge(archive[['dataset','seed','method','test_accuracy']],
                        on=['dataset','seed','method'],suffixes=('', '_archived'),validate='one_to_one')
    check['archive_delta_pp'] = 100*(check.test_accuracy-check.test_accuracy_archived)
    check[['dataset','seed','method','test_accuracy','test_accuracy_archived','archive_delta_pp']].to_csv(
        out/'archive_comparison.csv',index=False)

    paired = pd.read_csv(out/'paired_results.csv')
    summary = pd.read_csv(out/'summary.csv')
    signs = paired.groupby(['dataset','arm']).delta_pp.agg(
        positive=lambda x:int((x>1e-6).sum()),negative=lambda x:int((x < -1e-6).sum()),
        ties=lambda x:int((x.abs()<=1e-6).sum())).reset_index()
    summary = summary.merge(signs,on=['dataset','arm'],validate='one_to_one')
    summary['accuracy_percent'] = 100*summary.accuracy_mean
    summary['sd_percent'] = 100*summary.accuracy_sd
    summary.to_csv(out/'summary_verified.csv',index=False,encoding='utf-8-sig')
    reference.groupby('dataset').agg(
        noisy_brier=('noisy_brier','mean'), target_brier=('target_brier','mean'),
        corrupted_gain=('corrupted_gain','mean'), clean_cost=('clean_cost','mean'),
        net_gain=('brier_gain','mean'), true_margin_fraction=('positive_clean_margin_fraction','mean'),
        target_accuracy=('target_accuracy','mean'), corrected_corrupt=('corrected_corrupt','mean'),
        damaged_clean=('damaged_clean','mean'), mean_balls=('num_balls','mean'),
        mean_other_votes=('mean_other_votes','mean'), coverage=('alpha_coverage','mean')).to_csv(
            out/'theory_diagnostics.csv',encoding='utf-8-sig')

    cells = summary.copy()
    cells['cell'] = cells.apply(lambda r: f'{r.accuracy_percent:.2f} +/- {r.sd_percent:.2f} ({r.delta_pp:+.2f})',axis=1)
    wide = cells.pivot(index='arm',columns='dataset',values='cell').reindex([a['name'] for a in arms(cfg)])
    wide = wide[cfg['datasets']]
    lines = ['# GPU Supplementary Results', '',
             'Accuracy: mean +/- sample SD (%); parentheses: paired mean change from the freshly rerun default (percentage points).',
             '', '| Arm | Cora | CiteSeer | CoraFull |', '|---|---:|---:|---:|']
    for name, values in wide.iterrows():
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '## Checks', '', f'- Logical evaluations: {len(frame)}.',
              f'- Actual GPU training runs: {int(frame.reused_from.isna().sum())}.',
              f'- Exact-target reuse: {int(frame.reused_from.notna().sum())}; not independent repeats.',
              f'- Maximum default Brier deviation from archive: {reference.default_archive_brier_delta.abs().max():.3g}.',
              '- All devices, pair fingerprints, stored targets, losses and predictions passed checks.',
              '- All 15 raw-X / A^2X pairs have one ball and identical targets.',
              '- Original algorithm/config files still match the pre-run snapshot.',
              '- Accuracy differences from archived training runs are listed separately in archive_comparison.csv.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))
    print('\nTheory diagnostics:\n'+pd.read_csv(out/'theory_diagnostics.csv').to_string(index=False))
    print('\nArchived accuracy comparison:\n'+check[['dataset','seed','method','archive_delta_pp']].to_string(index=False))


if __name__ == '__main__':
    main()
