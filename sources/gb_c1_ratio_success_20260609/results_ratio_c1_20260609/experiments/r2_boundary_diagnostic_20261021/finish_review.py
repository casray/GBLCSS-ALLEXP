"""Recovery entry: validate optional historical arms explicitly, no training reruns."""
from __future__ import annotations

import gzip
import json
import time

import numpy as np

import diagnose as d


def check_optional_arm(reported,values):
    if reported=='':
        assert not values,'Blank table field unexpectedly has raw runs'
        return False
    assert len(values)==3,'Expected all three seeds, not a partial result'
    assert abs(float(reported)-np.mean(values))<1e-9
    return True


def review():
    with gzip.open(d.BUDGET/'raw_results.jsonl.gz','rt',encoding='utf-8') as f:
        raw=[json.loads(line) for line in f]
    assert len(raw)==1080 and all(r['status']=='completed' for r in raw)
    def key(j):
        return j['dataset'],j['noise'],j['seed'],j['backbone'],float(j['ratio']),j['arm'],j['rule']
    indexed={key(r['job']):r for r in raw}
    assert len(indexed)==1080
    pair_difference=0.
    pairs=d.csv_read(d.BUDGET/'paired_accuracy.csv')
    e2=d.csv_read(d.BUDGET/'E2_diagnostic.csv')
    for row in pairs+e2:
        prefix=row['dataset'],row['noise'],row['backbone'],float(row['ratio'])
        dataset,noise,backbone,ratio=prefix
        values=[]
        for seed in (0,1,2):
            a=indexed[dataset,noise,seed,backbone,ratio,row['arm'],row['rule']]['test_accuracy']
            b=indexed[dataset,noise,seed,backbone,ratio,row['reference'],row['reference_rule']]['test_accuracy']
            values.append(100*(a-b))
        error=abs(np.mean(values)-float(row['delta_pp_mean']))
        pair_difference=max(pair_difference,error)
        assert error<1e-9
        assert abs(np.std(values,ddof=1)-float(row['delta_pp_std']))<1e-9
        np.testing.assert_allclose(values,json.loads(row['seed_deltas_pp']),atol=1e-9,rtol=0)
    available,absent=0,0
    for row in d.csv_read(d.BUDGET/'main_accuracy_and_f1.csv'):
        for arm in ('A','B','Bprime','LCSS','C','D'):
            records=[indexed[k] for seed in (0,1,2)
                     if (k:=(row['dataset'],row['noise'],seed,row['backbone'],.1,arm,'R0')) in indexed]
            values=[r['test_accuracy']*100 for r in records]
            if check_optional_arm(row[arm+'_test_accuracy_mean_percent'],values):
                available+=1
                assert abs(np.std(values,ddof=1)-float(row[arm+'_test_accuracy_std_pp']))<1e-9
            else:
                absent+=1
    summaries=[]
    for backbone in ('cached_mlp','gcn'):
        for reference in ('A','B','LCSS'):
            selected=[r for r in pairs if r['dataset'] in ('cora','citeseer','pubmed','amazonpho','corafull')
                      and r['backbone']==backbone and r['noise']!='clean' and float(r['ratio'])==.1
                      and r['arm']=='C' and r['rule']=='R0' and r['reference']==reference and r['reference_rule']=='R0']
            assert len(selected)==15
            values=[float(r['delta_pp_mean']) for r in selected]
            summaries.append(dict(backbone=backbone,reference=reference,cell_means=15,
                mean_delta_pp=float(np.mean(values)),positive_cell_means=sum(v>0 for v in values),
                all_three_seed_positive_cells=sum(int(r['positive_seeds'])==3 for r in selected)))
    old=d.read(d.BUDGET/'aggregates.json')
    for r in summaries:
        match=next(x for x in old['main'] if x['backbone']==r['backbone'] and x['reference']==r['reference'])
        assert abs(match['mean_delta_pp']-r['mean_delta_pp'])<1e-10
    costs=d.csv_read(d.BUDGET/'costs.csv')
    for row in costs:
        for arm in ('A','C'):
            # A is the shared uncompressed baseline, not independently fitted for each compression ratio.
            ratio=.1 if arm=='A' else float(row['ratio'])
            records=[indexed[row['dataset'],row['noise'],seed,row['backbone'],ratio,arm,'R0'] for seed in (0,1,2)]
            for field in ('cold_total_s','gpu_peak_allocated_gib'):
                assert abs(np.mean([r[field] for r in records])-float(row[arm+'_'+field]))<1e-8
    clean=[r for r in costs if r['noise']=='clean']
    e2_summary={rule:float(np.mean([float(r['delta_pp_mean']) for r in e2 if r['rule']==rule])) for rule in ('R3','O2')}
    result=dict(raw_records_rechecked=1080,paired_rows_checked=len(pairs)+len(e2),
                paired_table_max_mean_difference_pp=float(pair_difference),
                paired_mean_and_sample_std_verified=True,available_main_arm_summaries_checked=available,
                absent_optional_arms_verified_not_run=absent,main_accuracy_table_verified=True,cost_means_verified=True,
                aggregate_comparisons=summaries,E2=e2_summary,clean_cost_cells=len(clean),
                original_efficiency_gate_passes=sum(r['G0_observed_pass'].lower()=='true' for r in clean),
                clean_cold_cost_ratio_min=min(float(r['C_over_A_cold_total_s']) for r in clean),
                clean_cold_cost_ratio_max=max(float(r['C_over_A_cold_total_s']) for r in clean),
                source_report=str(d.BUDGET/'REPORT.md'),source_audit=d.read(d.BUDGET/'audit.json'),
                verification_scope='recomputed raw stored metric/table arithmetic; no model retraining or second full prediction-tensor audit')
    d.csv_write(d.ROOT/'budget_comparison_summary.csv',summaries)
    d.write(d.ROOT/'budget_review.json',result)
    return result


def check_saved_boundary():
    rows=d.csv_read(d.ROOT/'node_metrics.csv')
    protocols=d.csv_read(d.ROOT/'protocol_summary.csv')
    assert len(rows)==20656 and len(protocols)==12
    maximum=0.
    for record in protocols:
        selected=[r for r in rows if r['dataset']==record['dataset'] and r['condition']==record['condition']]
        assert len(selected)==int(record['training_nodes'])
        delta=np.array([float(r['gb_brier'])-float(r['knn_brier']) for r in selected])
        stored=np.array([float(r['brier_delta']) for r in selected])
        np.testing.assert_allclose(delta,stored,atol=1e-12,rtol=0)
        error=abs(delta.mean()-float(record['brier_delta_mean']))
        maximum=max(maximum,error)
        assert error<1e-12
    bins=d.csv_read(d.ROOT/'binned_metrics.csv')
    correlations=d.csv_read(d.ROOT/'correlations.csv')
    for row in correlations:
        selected=[r for r in rows if r['dataset']==row['dataset'] and r['condition']==row['condition'] and r['eligible']=='True']
        if row['subset']!='all':
            keep=row['subset']=='originally_clean'
            selected=[r for r in selected if (r['originally_clean']=='True')==keep]
        assert len(selected)==int(row['nodes'])
        rho=d.spearman([float(r[row['feature']]) for r in selected],[float(r['brier_delta']) for r in selected])
        if rho is None:
            assert row['spearman_brier_delta']==''
        else:
            assert abs(rho-float(row['spearman_brier_delta']))<1e-12
        grouped=[r for r in bins if all(r[k]==row[k] for k in ('dataset','condition','feature','subset'))]
        assert sum(int(r['nodes']) for r in grouped)==len(selected)
    return dict(node_rows=len(rows),protocols=12,correlation_rows=len(correlations),binned_rows=len(bins),
                max_saved_mean_recompute_difference=maximum)


def main():
    started=time.time()
    frozen=d.read(d.ROOT/'frozen_inputs.json')['sha256']
    for path,expected in frozen.items():
        assert d.sha(path)==expected,'Frozen input changed: '+path
    checks=check_saved_boundary()
    budget=review()
    for path,expected in frozen.items():
        assert d.sha(path)==expected
    d.write(d.ROOT/'audit.json',dict(status='passed',**checks,frozen_sources=len(frozen),model_fits=0,
            budget_records_rechecked=1080,paired_budget_mean_max_difference_pp=budget['paired_table_max_mean_difference_pp'],
            recovery_source_sha256=d.sha(__file__),recovery_test_sha256=d.sha(d.ROOT/'test_review.py'),
            recovery_from='historical optional B/Bprime/LCSS arms absent on cost-only Coauthor-CS; failure.json retained',
            seconds=time.time()-started,scope='posthoc descriptive, no causal proof or independent model retraining'))
    print(json.dumps(dict(status='complete',checks=checks,budget=budget['aggregate_comparisons'],
                         decision=d.read(d.ROOT/'decision.json')),ensure_ascii=False))


if __name__=='__main__':
    main()
