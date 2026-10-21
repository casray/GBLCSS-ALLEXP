"""Versioned CSV parsing adapter; preserve all frozen experimental code and artifacts."""
from pathlib import Path
import json
import shutil
import time
import numpy as np
import pandas as pd
import audit_report as audit
import data_tools as dt

ORIGINAL_COMPARE=audit.compare_metric


def compare_csv_metric(actual,expected,label,tolerance=2e-12):
    # keep_default_na=False preserves blanks, making partly missing columns strings.
    def number(value):
        if value is None or (isinstance(value,str) and not value.strip()):return np.nan
        return float(value)
    return ORIGINAL_COMPARE(number(actual),number(expected),label,tolerance)


def main(plots_only=False):
    import run
    import torch
    torch.set_num_threads(dt.CONFIG['threads'])
    root=dt.ROOT
    initial=json.loads((root/'status.json').read_text(encoding='utf-8'))
    if initial['status']=='running':raise ValueError('Wait until the fixed training queue exits')
    frame=pd.read_csv(root/'student_results.csv')
    expected=sum(len(dt.arms_for(dt.load_case(n,0,'clean')[0].num_nodes))*len(dt.CONFIG['protocol_seeds'])*len(dt.CONFIG['conditions'])
                 for n in dt.CONFIG['datasets'])
    if len(frame)!=expected:raise ValueError('Incomplete student matrix')
    run.verify_sources()
    archive=root/'initial_postprocess';archive.mkdir(exist_ok=True)
    for name in ['status.json','audit.json','实验结论.md']:
        p=root/name;target=archive/name
        if p.exists() and not target.exists():shutil.copyfile(p,target)
    old_manifest=root/'postprocess_adapter_manifest.json'
    old_copy=archive/'postprocess_adapter_manifest_before_plot_fix.json'
    if old_manifest.exists() and not old_copy.exists():shutil.copyfile(old_manifest,old_copy)
    provenance=dict(reason='CSV columns containing blanks are object strings with keep_default_na=False; convert numeric scalars before independent metric checks',
        original_experiment_sources_unchanged=True,adapter_sha256=dt.d.sha(__file__),
        original_audit_sha256=dt.d.sha(root/'audit_report.py'),regression_test_sha256=dt.d.sha(root/'test_postprocess.py'),
        reviewed_plot_sha256=dt.d.sha(root/'plot_curves_reviewed.py'),
        no_model_or_target_changed=True,initial_status=initial,plots_only=plots_only,
        audit_sha256=dt.d.sha(root/'audit.json'),plot_regression_test_sha256=dt.d.sha(root/'test_plot_layout.py'))
    dt.d.write(root/'postprocess_adapter_manifest.json',provenance)
    audit.compare_metric=compare_csv_metric
    result=json.loads((root/'audit.json').read_text(encoding='utf-8')) if plots_only else audit.audit_and_report(make_plots=False)
    if result['status']!='passed':raise ValueError('Numerical audit must pass before plotting')
    from plot_curves_reviewed import make_plots
    make_plots()
    dt.d.write(root/'status.json',dict(status='completed',stage='all',postprocess_adapter='finish_diagnostic.py',
        updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),target_rows=result['target_artifacts'],
        unique_student_models=result['unique_student_models'],initial_status_preserved='initial_postprocess/status.json'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plots-only',action='store_true')
    main(parser.parse_args().plots_only)
