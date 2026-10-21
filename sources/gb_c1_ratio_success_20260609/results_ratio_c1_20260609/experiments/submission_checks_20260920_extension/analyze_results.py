"""Validate the extension and join it to the completed three-dataset matrix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_experiment import HERE, OLD, runner, check_sources


def verify(folder):
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf8'))
    out = folder / ('gpu_reference_matched_v2' if (folder / 'gpu_reference_matched_v2').exists()
                    else 'gpu_reference_matched')
    frame = pd.read_csv(out / 'per_seed_results.csv')
    expected = {(d, s, a['name']) for d in cfg['datasets'] for s in cfg['seeds']
                for a in runner.arms(cfg)}
    assert set(frame[['dataset', 'seed', 'arm']].itertuples(index=False, name=None)) == expected
    assert len(frame) == len(expected)
    runner.validate_pairs(frame.to_dict('records'))
    assert frame.status.eq('complete').all() and frame.device.eq('cuda:0').all()
    assert frame.gpu_peak_allocated_mib.gt(0).all() and frame.epochs.eq(200).all()
    assert np.allclose(frame.brier_gain, frame.corrupted_gain-frame.clean_cost, atol=1e-7)
    default_delta = frame[frame.arm.eq('default')].default_archive_brier_delta.abs()
    if cfg.get('enforce_archive_brier', True):
        assert default_delta.max() < 2e-5
    for (dataset, seed), rows in frame.groupby(['dataset', 'seed']):
        rows = rows.set_index('arm')
        fixed = ['default', 'beta_0', 'beta_0.25', 'beta_0.75', 'beta_1',
                 'min_train_per_ball_5', 'min_train_per_ball_10']
        assert rows.loc[fixed, 'assignment_hash'].nunique() == 1
        for arm, row in rows.iterrows():
            p = out / 'runs' / dataset / f'seed{seed}' / (arm+'.npz')
            with np.load(p) as arrays:
                assert np.isfinite(arrays['targets']).all()
                assert np.allclose(arrays['targets'].sum(1), 1, atol=1e-6)
            if pd.notna(row.reused_from):
                assert row.target_hash == rows.loc[row.reused_from, 'target_hash']
                assert row.test_accuracy == rows.loc[row.reused_from, 'test_accuracy']
            else:
                losses = pd.read_csv(p.with_name(p.stem+'_loss.csv'))
                assert len(losses) == 200 and np.isfinite(losses.loss).all()
                assert p.with_name(p.stem+'_predictions.npz').exists()
    return frame, cfg


def export(frame, out, datasets, cfg):
    out.mkdir(exist_ok=True)
    frame.to_csv(out/'per_seed_results.csv', index=False, encoding='utf-8-sig')
    reference = frame[frame.arm.eq('default')]
    paired = frame.merge(reference[['dataset','seed','test_accuracy','target_brier']],
                         on=['dataset','seed'], suffixes=('', '_reference'), validate='many_to_one')
    paired['delta_pp'] = 100*(paired.test_accuracy-paired.test_accuracy_reference)
    paired.to_csv(out/'paired_results.csv', index=False, encoding='utf-8-sig')
    summary = paired.groupby(['dataset','arm']).agg(
        seeds=('seed','nunique'), accuracy_mean=('test_accuracy','mean'),
        accuracy_sd=('test_accuracy','std'), delta_pp=('delta_pp','mean'),
        paired_sd_pp=('delta_pp','std'),
        positive=('delta_pp',lambda x: int((x>1e-6).sum())),
        negative=('delta_pp',lambda x: int((x < -1e-6).sum())),
        balls_mean=('num_balls','mean'), balls_min=('num_balls','min'), balls_max=('num_balls','max'),
        coverage=('alpha_coverage','mean'), brier=('target_brier','mean')).reset_index()
    summary.to_csv(out/'summary_verified.csv', index=False, encoding='utf-8-sig')
    reference.groupby('dataset').agg(
        noisy_brier=('noisy_brier','mean'), target_brier=('target_brier','mean'),
        corrupted_gain=('corrupted_gain','mean'), clean_cost=('clean_cost','mean'),
        net_gain=('brier_gain','mean'), margin_fraction=('positive_clean_margin_fraction','mean'),
        balls=('num_balls','mean'), coverage=('alpha_coverage','mean')).to_csv(
            out/'theory_diagnostics.csv', encoding='utf-8-sig')
    summary['cell'] = summary.apply(
        lambda r: f'{100*r.accuracy_mean:.2f} +/- {100*r.accuracy_sd:.2f} ({r.delta_pp:+.2f})', axis=1)
    wide = summary.pivot(index='arm', columns='dataset', values='cell')
    wide = wide.reindex([a['name'] for a in runner.arms(cfg)])[datasets]
    lines = ['# Results', '', 'Mean +/- sample SD (%); paired mean difference from fresh default (pp).',
             '', '| Arm | '+' | '.join(datasets)+' |', '|---|'+'---:|'*len(datasets)]
    lines += ['| '+name+' | '+' | '.join(values)+' |' for name,values in wide.iterrows()]
    lines += ['', f'Logical evaluations: {len(frame)}. Actual training runs: {frame.reused_from.isna().sum()}.',
              f'Exact-target reuse: {frame.reused_from.notna().sum()}. All training uses cuda:0.',
              f'Maximum default archive Brier difference: {reference.default_archive_brier_delta.abs().max():.3g}.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))


def main():
    check_sources()
    new, cfg = verify(HERE)
    old, oldcfg = verify(OLD)
    for field in ['seeds','noise_type','noise_rate','epochs','threads','grouping_threads','grids','code_root']:
        assert cfg[field] == oldcfg[field], field
    export(new, HERE/'verified', cfg['datasets'], cfg)
    combined = pd.concat([old,new],ignore_index=True)
    export(combined,HERE/'combined_five_datasets',oldcfg['datasets']+cfg['datasets'],cfg)
    archive = pd.read_csv(Path(cfg['code_root'])/'results/main_tables/table2_per_seed.csv')
    archive['dataset'] = archive.dataset.replace({'amazonpho': 'amazonphoto'})
    archive = archive[(archive.noise_type=='uniform') & (archive.noise_rate==.4)]
    check = new[new.arm.isin(['default','mlp_ce'])].copy()
    check['method'] = check.arm.map({'default':'GB-LCSS','mlp_ce':'MLP+CE'})
    check = check.merge(archive[['dataset','seed','method','test_accuracy']],
                       on=['dataset','seed','method'],suffixes=('', '_archived'),validate='one_to_one')
    assert len(check) == 20
    check['archive_delta_pp'] = 100*(check.test_accuracy-check.test_accuracy_archived)
    check.to_csv(HERE/'verified/archive_comparison.csv',index=False)
    archive_brier = new[new.arm.eq('default')].copy()
    archive_report = {
        'gate_enforced': bool(cfg.get('enforce_archive_brier', True)),
        'max_absolute_brier_difference': float(archive_brier.default_archive_brier_delta.abs().max()),
        'by_dataset': archive_brier.groupby('dataset').default_archive_brier_delta.agg(
            max_abs=lambda x: float(x.abs().max()), mean=lambda x: float(x.mean()),
            mismatched=lambda x: int((x.abs() >= 2e-5).sum())).to_dict('index'),
        'interpretation': 'Current KMeans environment is retained; archive differences are reported, not used as a pass gate.'
    }
    (HERE/'verified/archive_brier_report.json').write_text(
        json.dumps(archive_report, indent=2), encoding='utf8')
    print(check[['dataset','seed','method','archive_delta_pp']].to_string(index=False))


if __name__ == '__main__':
    main()
