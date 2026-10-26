"""Read-only scientific audit and descriptive tables; no fit or model selection."""
import csv
from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch

from fit_pipeline import ROOT, EXP, CONFIG, read, sha, save_json, save_csv


def mean(rows,metric='accuracy'):
    return float(np.mean([float(r[metric]) for r in rows]))


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
        '| '+' | '.join(str(x) for x in row)+' |' for row in rows]


def main():
    assert read(ROOT/'analysis_complete.json')['fit_rows']==336
    assert read(ROOT/'fit_status.json')['status']=='complete'
    assert not (ROOT/'fit.lock').exists()
    verified={}
    for manifest in ('fit_sources.json','fit_inputs.json'):
        counts=0
        for filename,digest in read(ROOT/manifest).items():
            path=Path(filename)
            actual=ROOT/'analyze_fits.before_fix.txt' if path==ROOT/'analyze_fits.py' else path
            if sha(actual)!=digest:
                raise RuntimeError('Changed frozen source/input: '+str(path))
            counts+=1
        verified[manifest]=counts
    fits=[]
    for stage in ('E1','E2','E4'):
        manifest=read(ROOT/f'{stage}_fits_complete.json')
        for filename,digest in manifest['results'].items():
            path=Path(filename)
            if sha(path)!=digest:
                raise RuntimeError('Completed result changed')
            r=read(path)
            for name,field in [('student.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
                if sha(path.parent/name)!=r[field]:
                    raise RuntimeError('Model artifact changed')
            epochs=[json.loads(line) for line in (path.parent/'epochs.jsonl').read_text().splitlines()]
            best=max(e['validation_correct'] for e in epochs)
            first=next(e['epoch'] for e in epochs if e['validation_correct']==best)
            if first!=r['best_epoch'] or best!=r['valid_correct'] or len(epochs)!=r['epochs']:
                raise RuntimeError('Checkpoint selection or epoch budget mismatch')
            fits.append((stage,r))
    count=Counter(stage for stage,r in fits)
    assert dict(count)==dict(E1=144,E2=72,E4=120)
    pairs=list((ROOT/'fits/E4').glob('**/pair_audit.json'))
    assert len(pairs)==20
    assert all(read(p)['initial_R2_exact'] and read(p)['A200_A300_prefix_exact'] for p in pairs)
    assert all(read(p)['exact'] for p in (ROOT/'alignment').glob('**/bridge.json'))
    for filename,digest in read(ROOT/'test_release.json')['choices'].items():
        assert sha(Path(filename))==digest
    results=list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8-sig')))
    rows=lambda stage,n,a,clean=False:[r for r in results if r['stage']==stage and r['dataset']==n and r['arm']==a
                                    and ((r['condition']=='clean') if clean else (r['condition']!='clean'))]
    lines=['# Locked posthoc tables','',
        'All classification runs use seed0. Accuracy and macro-F1 are percentages. No sample standard deviation is estimable from one seed; summary.csv uses null, not zero.',
        'Noisy means are equal-weight averages over uniform0.4, uniform0.6 and pair0.4. Selection and continuation used validation only.','',
        '## E1: R0 noisy accuracy','']
    arms=['CE','GCE','OLD64','LEGACY32','FIXED32','SC','VAL','KM','KNN']
    lines+=table(['Dataset']+arms,[[n]+[f'{100*mean(rows("E1",n,a)):.3f}' for a in arms] for n in CONFIG['pilot']['E1_datasets']])
    lines+=['','## E1: clean accuracy / macro-F1','']
    lines+=table(['Dataset','CE','SC','FIXED32','KM','KNN'],[[n]+[
        f'{100*mean(rows("E1",n,a,True)):.3f} / {100*mean(rows("E1",n,a,True),"macro_f1"):.3f}'
        for a in ['CE','SC','FIXED32','KM','KNN']] for n in CONFIG['pilot']['E1_datasets']])
    lines+=['','## E1: selected supports','']
    choices=read(ROOT/'E1_pilot_decision.json')['selections']
    lines+=table(['Dataset','Condition','SC','VAL','SC-VAL validation pp'],[[r['dataset'],r['condition'],r['SC'],r['VAL'],f'{r["SC_minus_VAL_pp"]:.3f}'] for r in choices])
    lines+=['','## E2: mean test accuracy, all four conditions and both heads','']
    e2=[]
    for n in CONFIG['pilot']['E2_datasets']:
        row=[n]
        for p in (100,300,600):
            selected=[r for r in results if r['stage']=='E2' and r['dataset']==n and r['arm'].startswith(f'P{p}_')]
            row.append(f'{100*mean(selected):.3f}')
        e2.append(row)
    lines+=table(['Dataset','100','300','600'],e2)
    lines+=['','## E4: noisy accuracy / macro-F1','']
    arms=['A200','A300','B','D','C_full','C_head']
    lines+=table(['Dataset']+arms,[[n]+[
        f'{100*mean(rows("E4",n,a)):.3f} / {100*mean(rows("E4",n,a),"macro_f1"):.3f}' for a in arms] for n in CONFIG['datasets']])
    lines+=['','## E4: clean accuracy','']
    lines+=table(['Dataset']+arms,[[n]+[f'{100*mean(rows("E4",n,a,True)):.3f}' for a in arms] for n in CONFIG['datasets']])
    selected_comparisons=[]
    for choice in read(ROOT/'E4_decision.json')['selections']:
        n,c,b=choice['dataset'],choice['condition'],choice['comparator']
        indexed={r['arm']:r for r in results if r['stage']=='E4' and r['dataset']==n and r['condition']==c}
        selected_comparisons.append(dict(dataset=n,condition=c,comparator=b,
            B_accuracy=float(indexed['B']['accuracy']),comparator_accuracy=float(indexed[b]['accuracy']),
            delta_pp=100*(float(indexed['B']['accuracy'])-float(indexed[b]['accuracy'])),
            macro_f1_delta_pp=100*(float(indexed['B']['macro_f1'])-float(indexed[b]['macro_f1']))))
    save_csv(ROOT/'E4_locked_comparator_test.csv',selected_comparisons)
    lines+=['','## E4: B versus validation-selected comparator (test deltas)','']
    lines+=table(['Dataset','Noisy accuracy delta pp','Noisy macro-F1 delta pp','Clean accuracy delta pp'],[
        [n,f'{mean([r for r in selected_comparisons if r["dataset"]==n and r["condition"]!="clean"],"delta_pp"):.3f}',
         f'{mean([r for r in selected_comparisons if r["dataset"]==n and r["condition"]!="clean"],"macro_f1_delta_pp"):.3f}',
         f'{mean([r for r in selected_comparisons if r["dataset"]==n and r["condition"]=="clean"],"delta_pp"):.3f}'] for n in CONFIG['datasets']])
    costs=[]
    for stage,r in fits:
        alignment_updates=100 if stage=='E4' and r['arm'] in ('B','D','C_full','C_head') else 0
        costs.append(dict(stage=stage,dataset=r['dataset'],condition=r['condition'],arm=r['arm'],
            measured_fit_seconds=r['seconds'],fit_updates=r['epochs'],alignment_updates=alignment_updates,
            total_student_updates=r['epochs']+alignment_updates,
            excludes='geometry, cached teacher/encoder pretraining and I/O; counts do not imply equal FLOPs'))
    save_csv(ROOT/'update_and_fit_costs.csv',costs)
    lines+=['','## Scope and audit','',
        'E1 144 fits; E2 72 heads plus 3 alignment trajectories; E4 120 student fits plus 5 new clean GCN teachers. Four 3-epoch smoke fits are excluded from all result tables.',
        'No seed1/2 fit expansion passed. E3/E5 were not launched. Cached noisy teachers are ordinary SUPPORT-LCSS GCNs, not ERASE; KD is all-node soft cross-entropy, not the DIST paper loss.',
        'Two analysis-only implementation errors were repaired after the first test release; original logs/source and locked selections were preserved. See ANALYSIS_AMENDMENT.md.']
    (ROOT/'RESULT_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(passed=True,verified_manifests=verified,fit_counts=dict(count),paired_E4_cells=len(pairs),
        first_best_validation_checkpoint_checks=len(fits),all_three_historical100_encoders_exact=True,
        locked_choices_unchanged=True,analysis_source_sha256=sha(ROOT/'analyze_fits.py'),
        analysis_backup_sha256=sha(ROOT/'analyze_fits.before_fix.txt'),repair_document=sha(ROOT/'ANALYSIS_AMENDMENT.md'),
        test_release_sha256=sha(ROOT/'test_release.json'),summary_sha256=sha(ROOT/'summary.csv'))
    save_json(ROOT/'final_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
