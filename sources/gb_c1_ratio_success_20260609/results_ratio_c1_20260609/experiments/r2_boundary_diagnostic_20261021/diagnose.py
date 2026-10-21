"""Read-only analysis of existing target/model caches; no model training."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from diagnostic_utils import (geometry,crossing_fraction,quantile_bins,target_metrics,
                              spearman,within_group_rank_correlation)

ROOT=Path(__file__).resolve().parent
R2=ROOT.parent/'r2_mechanism_controls_20261021'
BASE=ROOT.parent/'representation_supervision_20261021'
BUDGET=ROOT.parent/'budget_gb_coarsening_20261019/analysis_20261020/results'
DATASETS=('cora','citeseer','corafull')
CONDITIONS=('clean','uniform0.4','uniform0.6','pair0.4')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(p):
    value=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def csv_read(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def csv_write(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    if (ROOT/'frozen_inputs.json').exists():
        raise RuntimeError('Analysis already started; do not overwrite frozen outputs')
    sources=dict(read(R2/'frozen_inputs.json')['sha256'])
    for path,expected in sources.items():
        if sha(path)!=expected:
            raise RuntimeError('R2 source changed: '+path)
    paths=[ROOT/'PLAN.md',*ROOT.glob('*.py'),R2/'results.csv',R2/'audit.json',R2/'independent_audit.json']
    for name in DATASETS:
        for c in CONDITIONS:
            for rule in ('GB','KNN'):
                for leaf in ('result.json','probabilities.pt'):
                    paths.append(R2/'runs'/name/c/('REAL_'+rule)/leaf)
    paths.extend(p for p in BUDGET.iterdir() if p.is_file())
    for p in paths:
        sources[str(p)]=sha(p)
    write(ROOT/'frozen_inputs.json',dict(sha256=sources,started=time.time(),model_training=False))
    return sources


def vote_distributions(z,assignment,neighbors,support,classes):
    _,inv=np.unique(assignment,return_inverse=True)
    counts=np.zeros((int(inv.max())+1,classes))
    np.add.at(counts,(inv,z),1)
    local=counts[inv]-np.eye(classes)[z]
    gb=local/np.maximum(support,1)[:,None]
    knn=np.zeros_like(gb)
    for n in range(len(z)):
        if support[n]:
            knn[n]=np.bincount(z[neighbors[n,:support[n]]],minlength=classes)/support[n]
    return gb,knn


def mean(a,mask):
    return float(np.asarray(a)[mask].mean()) if np.any(mask) else None


def boundary():
    all_nodes,binned,correlations,protocols=[],[],[],[]
    direct_brier_diff=0.
    for name in DATASETS:
        assets=torch.load(BASE/'inputs'/name/'assets.pt',weights_only=True,map_location='cpu')
        groups=torch.load(R2/'inputs'/name/'groups.pt',weights_only=True,map_location='cpu')
        train=assets['train'].numpy()
        assignment=assets['assignment'].numpy()
        neighbors=groups['knn'].numpy()
        support=groups['knn_support'].numpy()
        if not np.array_equal(assignment,groups['assignment']['GB'].numpy()):
            raise AssertionError('GB assignment differs')
        h=assets['h'].numpy()
        distance,radius,relative,size=geometry(h,assignment)
        crossing=crossing_fraction(assignment[train],neighbors,support)
        eligible=support>=2
        for n in range(len(train)):
            assert n not in neighbors[n,:support[n]]
        geometry_features=dict(radial_distance=relative[train],cross_ball_fraction=crossing)
        geometry_bins={k:quantile_bins(v,eligible) for k,v in geometry_features.items()}
        protocol=read(BASE/'inputs'/name/'clean.json')
        from torch_geometric.data import Data
        raw=torch.load(protocol['raw_source'],weights_only=False,map_location='cpu')[0]
        clean=(raw['y'] if isinstance(raw,dict) else raw.y).numpy()[train]
        classes=assets['classes']
        for condition in CONDITIONS:
            source=torch.load(BASE/'inputs'/name/(condition+'.pt'),weights_only=True,map_location='cpu')
            targets=torch.load(R2/'inputs'/name/(condition+'.pt'),weights_only=True,map_location='cpu')
            z=source['observed'].numpy()
            gb,knn=targets['GB'].numpy(),targets['KNN'].numpy()
            assert np.array_equal(gb,source['target'].numpy())
            gm,km=target_metrics(gb,clean,z),target_metrics(knn,clean,z)
            for t,metrics in ((gb,gm),(knn,km)):
                direct=np.square(t.astype(float)-np.eye(classes)[clean]).sum(1)
                error=float(np.max(np.abs(direct-metrics['brier'])))
                direct_brier_diff=max(direct_brier_diff,error)
                assert error<1e-12
            delta=gm['brier']-km['brier']
            gv,kv=vote_distributions(z,assignment[train],neighbors,support,classes)
            tv=.5*np.abs(gv-kv).sum(1)
            features=dict(**geometry_features,vote_tv=tv)
            bins=dict(geometry_bins,vote_tv=quantile_bins(tv,eligible))
            predicted={rule:torch.load(R2/'runs'/name/condition/('REAL_'+rule)/'probabilities.pt',
                                      weights_only=True,map_location='cpu').numpy()[train].argmax(1)
                       for rule in ('GB','KNN')}
            model_delta=(predicted['GB']!=clean).astype(float)-(predicted['KNN']!=clean).astype(float)
            subsets=dict(all=eligible,originally_clean=eligible&(z==clean),originally_corrupted=eligible&(z!=clean))
            protocols.append(dict(dataset=name,condition=condition,seed=0,training_nodes=len(train),
                                  eligible_nodes=int(eligible.sum()),observed_clean_nodes=int((z==clean).sum()),
                                  gb_brier_mean=float(gm['brier'].mean()),knn_brier_mean=float(km['brier'].mean()),
                                  brier_delta_mean=float(delta.mean()),
                                  eligible_brier_delta_mean=float(delta[eligible].mean())))
            for n,node in enumerate(train):
                all_nodes.append(dict(dataset=name,condition=condition,seed=0,node_id=int(node),ball=int(assignment[node]),
                    clean_label=int(clean[n]),observed_label=int(z[n]),originally_clean=bool(z[n]==clean[n]),
                    eligible=bool(eligible[n]),support=int(support[n]),ball_size=int(size[node]),
                    center_distance=float(distance[node]),mean_radius=float(radius[node]),
                    radial_distance=float(relative[node]),cross_ball_fraction=None if not np.isfinite(crossing[n]) else float(crossing[n]),
                    vote_tv=float(tv[n]),gb_brier=float(gm['brier'][n]),knn_brier=float(km['brier'][n]),
                    brier_delta=float(delta[n]),gb_strict_wrong=bool(gm['strict_wrong'][n]),knn_strict_wrong=bool(km['strict_wrong'][n]),
                    gb_truth_tied=bool(gm['truth_tied'][n]),knn_truth_tied=bool(km['truth_tied'][n]),
                    gb_softened=bool(gm['softened'][n]),knn_softened=bool(km['softened'][n]),
                    gb_off_mass=float(gm['off_observed_mass'][n]),knn_off_mass=float(km['off_observed_mass'][n]),
                    train_model_error_delta=float(model_delta[n])))
            for feature,x in features.items():
                bin_id,edges=bins[feature]
                count=max(1,len(edges)-1)
                for subset,mask in subsets.items():
                    if not mask.any():
                        continue
                    qs=[]
                    for q in range(count):
                        chosen=mask&(bin_id==q)
                        value=mean(delta,chosen)
                        qs.append(value)
                        binned.append(dict(dataset=name,condition=condition,feature=feature,subset=subset,
                            bin=q+1,bins=count,cutpoints=json.dumps(edges.tolist()),nodes=int(chosen.sum()),
                            feature_mean=mean(x,chosen),brier_delta_mean=value,
                            gb_brier_mean=mean(gm['brier'],chosen),knn_brier_mean=mean(km['brier'],chosen),
                            gb_strict_wrong_rate=mean(gm['strict_wrong'],chosen),knn_strict_wrong_rate=mean(km['strict_wrong'],chosen),
                            gb_truth_tie_rate=mean(gm['truth_tied'],chosen),knn_truth_tie_rate=mean(km['truth_tied'],chosen),
                            gb_softened_rate=mean(gm['softened'],chosen),knn_softened_rate=mean(km['softened'],chosen),
                            gb_true_probability=mean(gm['true_probability'],chosen),knn_true_probability=mean(km['true_probability'],chosen),
                            train_model_error_delta_mean=mean(model_delta,chosen)))
                    gap=qs[-1]-qs[0] if len(qs)>=2 and qs[-1] is not None and qs[0] is not None else None
                    monotonic=bool(all(b>=a for a,b in zip(qs,qs[1:]))) if len(qs)==4 and None not in qs else None
                    correlations.append(dict(dataset=name,condition=condition,feature=feature,subset=subset,
                        nodes=int(mask.sum()),bins=count,spearman_brier_delta=spearman(x[mask],delta[mask]),
                        within_ball_rank_correlation=within_group_rank_correlation(x[mask],delta[mask],assignment[train][mask]),
                        within_class_rank_correlation=within_group_rank_correlation(x[mask],delta[mask],clean[mask]),
                        end_bin_gap=gap,four_bin_nondecreasing=monotonic,
                        train_error_spearman=spearman(x[mask],model_delta[mask])))
            print(f'BOUNDARY {name}/{condition}: target delta={delta.mean():+.6f}, eligible={eligible.sum()}/{len(train)}',flush=True)
    assert len(all_nodes)==20656
    csv_write(ROOT/'node_metrics.csv',all_nodes)
    csv_write(ROOT/'binned_metrics.csv',binned)
    csv_write(ROOT/'correlations.csv',correlations)
    csv_write(ROOT/'protocol_summary.csv',protocols)
    decision={}
    for feature in ('radial_distance','cross_ball_fraction','vote_tv'):
        selected=[r for r in correlations if r['feature']==feature and r['subset']=='all']
        def positive(r):
            return bool(r['spearman_brier_delta'] is not None and r['spearman_brier_delta']>0
                        and r['end_bin_gap'] is not None and r['end_bin_gap']>0)
        clean_rows=[r for r in selected if r['condition']=='clean']
        noisy=[r for r in selected if r['condition']!='clean']
        decision[feature]=dict(clean_positive_cells=sum(positive(r) for r in clean_rows),clean_cells=len(clean_rows),
            noisy_positive_cells=sum(positive(r) for r in noisy),noisy_cells=len(noisy),
            monotonic_cells=sum(r['four_bin_nondecreasing'] is True for r in selected),
            broad_descriptive_support=all(positive(r) for r in clean_rows) and sum(positive(r) for r in noisy)>=6
                if feature!='vote_tv' else None,
            causal_claim=False)
    write(ROOT/'decision.json',dict(features=decision,posthoc_diagnostic=True,previous_gb_continuation_remains_false=True,
                                    new_training_authorized=False,seed_count=1,seed_std=None))
    return dict(node_rows=len(all_nodes),binned_rows=len(binned),correlation_rows=len(correlations),protocols=len(protocols),
                max_independent_brier_difference=direct_brier_diff),decision


def budget_review():
    with gzip.open(BUDGET/'raw_results.jsonl.gz','rt',encoding='utf-8') as f:
        raw=[json.loads(line) for line in f]
    assert len(raw)==1080 and all(r['status']=='completed' for r in raw)
    def key(j):
        return j['dataset'],j['noise'],j['seed'],j['backbone'],float(j['ratio']),j['arm'],j['rule']
    indexed={key(r['job']):r for r in raw}
    assert len(indexed)==len(raw)
    max_pair_difference=0.
    for leaf in ('paired_accuracy.csv','E2_diagnostic.csv'):
        for row in csv_read(BUDGET/leaf):
            ds,noise,backbone,ratio=row['dataset'],row['noise'],row['backbone'],float(row['ratio'])
            delta=[]
            for seed in (0,1,2):
                a=indexed[ds,noise,seed,backbone,ratio,row['arm'],row['rule']]['test_accuracy']
                b=indexed[ds,noise,seed,backbone,ratio,row['reference'],row['reference_rule']]['test_accuracy']
                delta.append(100*(a-b))
            max_pair_difference=max(max_pair_difference,abs(np.mean(delta)-float(row['delta_pp_mean'])))
            assert abs(np.mean(delta)-float(row['delta_pp_mean']))<1e-9
            assert abs(np.std(delta,ddof=1)-float(row['delta_pp_std']))<1e-9
            assert np.allclose(delta,json.loads(row['seed_deltas_pp']),atol=1e-9,rtol=0)
    main=csv_read(BUDGET/'main_accuracy_and_f1.csv')
    for row in main:
        for arm in ('A','B','Bprime','LCSS','C','D'):
            values=[100*indexed[row['dataset'],row['noise'],seed,row['backbone'],.1,arm,'R0']['test_accuracy'] for seed in (0,1,2)]
            assert abs(np.mean(values)-float(row[arm+'_test_accuracy_mean_percent']))<1e-9
            assert abs(np.std(values,ddof=1)-float(row[arm+'_test_accuracy_std_pp']))<1e-9
    pairs=csv_read(BUDGET/'paired_accuracy.csv')
    useful=[]
    for backbone in ('cached_mlp','gcn'):
        for reference in ('A','B','LCSS'):
            selected=[r for r in pairs if r['backbone']==backbone and r['noise']!='clean' and float(r['ratio'])==.1
                      and r['arm']=='C' and r['rule']=='R0' and r['reference']==reference and r['reference_rule']=='R0']
            assert len(selected)==15
            values=[float(r['delta_pp_mean']) for r in selected]
            useful.append(dict(backbone=backbone,reference=reference,cell_means=15,mean_delta_pp=float(np.mean(values)),
                               positive_cell_means=sum(v>0 for v in values),
                               all_three_seed_positive_cells=sum(int(r['positive_seeds'])==3 for r in selected)))
    e2=csv_read(BUDGET/'E2_diagnostic.csv')
    e2_summary={rule:float(np.mean([float(r['delta_pp_mean']) for r in e2 if r['rule']==rule])) for rule in ('R3','O2')}
    costrows=csv_read(BUDGET/'costs.csv')
    for row in costrows:
        ds,noise,backbone,ratio=row['dataset'],row['noise'],row['backbone'],float(row['ratio'])
        for arm in ('A','C'):
            rs=[indexed[ds,noise,seed,backbone,ratio,arm,'R0'] for seed in (0,1,2)]
            for field in ('cold_total_s','gpu_peak_allocated_gib'):
                assert abs(np.mean([r[field] for r in rs])-float(row[arm+'_'+field]))<1e-8
    clean_cost=[r for r in costrows if r['noise']=='clean']
    original=read(BUDGET/'aggregates.json')
    for r in useful:
        match=next(x for x in original['main'] if x['backbone']==r['backbone'] and x['reference']==r['reference'])
        assert abs(match['mean_delta_pp']-r['mean_delta_pp'])<1e-10
    report=dict(raw_records_rechecked=1080,paired_table_max_mean_difference_pp=float(max_pair_difference),
                paired_mean_and_sample_std_verified=True,main_accuracy_table_verified=True,cost_means_verified=True,
                aggregate_comparisons=useful,E2=e2_summary,clean_cost_cells=len(clean_cost),
                original_efficiency_gate_passes=sum(r['G0_observed_pass']=='True' for r in clean_cost),
                clean_cold_cost_ratio_min=min(float(r['C_over_A_cold_total_s']) for r in clean_cost),
                clean_cold_cost_ratio_max=max(float(r['C_over_A_cold_total_s']) for r in clean_cost),
                source_report=str(BUDGET/'REPORT.md'),source_audit=read(BUDGET/'audit.json'),
                verification_scope='raw stored metrics and table arithmetic; no independent retraining or rereading all prediction tensors',
                deployment='coarsening GCN still uses original graph; not X-only deployment')
    csv_write(ROOT/'budget_comparison_summary.csv',useful)
    write(ROOT/'budget_review.json',report)
    return report


def main():
    torch.set_num_threads(4)
    sources=freeze()
    started=time.time()
    with threadpool_limits(limits=4):
        checks,decision=boundary()
        budget=budget_review()
    for path,expected in sources.items():
        assert sha(path)==expected,'Input changed: '+path
    write(ROOT/'audit.json',dict(status='passed',**checks,frozen_sources=len(sources),budget_records_rechecked=1080,
          paired_budget_mean_max_difference_pp=budget['paired_table_max_mean_difference_pp'],
          model_fits=0,seconds=time.time()-started,scope='descriptive/posthoc, no causal proof, no paper changes'))
    print(json.dumps(dict(status='complete',checks=checks,decision=decision,budget=budget['aggregate_comparisons']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    class Journal:
        def __init__(self,console,file):
            self.console,self.file=console,file

        def write(self,value):
            self.console.write(value)
            self.file.write(value)

        def flush(self):
            self.console.flush()
            self.file.flush()

    with (ROOT/'raw_log.txt').open('a',encoding='utf-8') as f:
        sys.stdout=Journal(sys.stdout,f)
        try:
            main()
        except Exception:
            write(ROOT/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
            raise
        finally:
            sys.stdout.flush()
            sys.stdout=sys.stdout.console
