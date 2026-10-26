"""Independent raw-log, target, selection chronology and checkpoint checks."""
from collections import defaultdict
import json

import torch

from protocol import (ROOT, EXP, CONFIG, INPUTS, read, save, sha, now, canonical, csv_save,
                      graph_inputs, cell_inputs, state_hash, load, disk_guard)
from run import cells, folder_for, decision_path, candidates, prepare
from selection import rank_key, probe_survivors, score
from targets import shrink, groups, global_mass, match_mass
from fit import Student, HIDDEN
from analyze import require_global_freeze


def first_best(logs):
    return max(logs,key=lambda r:r['validation_correct'])


def audit():
    errors,checks = [],defaultdict(int)
    def check(condition,message):
        if not condition:
            errors.append(message)
    for manifest in ('sources.json','inputs.json'):
        for path,digest in read(ROOT/manifest).items():
            check(sha(path)==digest,'Hash changed '+path)
            checks[manifest] += 1
    require_global_freeze()
    registry = read(ROOT/'fit_registry.json')
    check(len(registry)==3600,'Expected exactly 3600 full scientific fits')
    check(sum(r['stage']=='D1' for r in registry)==1440,'D1 count')
    check(sum(r['stage']=='C1' for r in registry)==2160,'C1 count')
    initial,inputs,allresults,logs_by_folder = defaultdict(set),defaultdict(set),{},{}
    for item in registry:
        folder = ROOT/item['folder']
        r = read(folder/'result.json')
        allresults[item['folder']] = r
        logs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        logs_by_folder[item['folder']] = logs
        best = first_best(logs)
        check(len(logs)==200 and [p['epoch'] for p in logs]==list(range(1,201)),'Epoch log '+item['folder'])
        check(r['epoch']==200 and r['status']=='complete','Incomplete result')
        check(r['best_epoch']==best['epoch'] and r['valid_correct']==best['validation_correct'],'First strict best '+item['folder'])
        check(r['valid_nll']==best['validation_nll'],'Best NLL mismatch')
        for filename,key in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
            check(sha(folder/filename)==r[key],'Artifact hash '+str(folder/filename))
        target = torch.load(ROOT/r['target_ref'],weights_only=True)
        check(state_hash({'target':target})==r['target_hash'],'Target hash')
        check(not (folder/'resume.pt').exists(),'Terminal resume not cleaned')
        initial[(item['dataset'],item['seed'],item['backbone'])].add((r['initial_hash'],r['initial_rng_hash']))
        inputs[(item['dataset'],item['seed'],item['backbone'],item['condition'])].add(r['input_hash'])
        if item['backbone']=='R2':
            check(r['encoder_frozen_unchanged'],'R2 freeze')
            checks['frozen_R2_fits'] += 1
        checks['full_logs_artifacts_targets'] += 1
    for key,values in initial.items():
        check(len(values)==1,'Unpaired initialization '+str(key))
        checks['initialization_groups'] += 1
    for key,values in inputs.items():
        check(len(values)==1,'Cell input mismatch '+str(key))
    for name in CONFIG['datasets']:
        for b in CONFIG['backbones']:
            check(len({next(iter(initial[(name,s,b)]))[0] for s in CONFIG['seeds']})==3,'Seeds share initialization')
    reloads,bridges,massrows = [],[],[]
    for name in CONFIG['datasets']:
        for seed in CONFIG['seeds']:
            graph = graph_inputs(name,seed)
            for backbone in CONFIG['backbones']:
                for cond in CONFIG['conditions']:
                    values,es,_ = prepare(name,seed,cond,graph)
                    e = es['LEGACY']
                    d1 = read(decision_path('D1',name,seed,backbone,cond))
                    def result(stage,arm):
                        return allresults[canonical(folder_for(stage,name,seed,backbone,cond,arm))]
                    global_arms = {0.:'CE',.25:'SUPPORT_L0.25',.5:'SUPPORT_L0.5',1.:'SUPPORT_L1.0'}
                    bestglobal = min(global_arms,key=lambda l:(-result('D1',global_arms[l])['validation'],l))
                    check(bestglobal==d1['best_cell_global_lambda'],'D1 global strength')
                    recomputed = [score(r['arm'],result('D1',r['arm'])) for r in d1['scores']]
                    check(min(recomputed,key=rank_key)['arm']==d1['selected'],'D1 group rank')
                    wpack = torch.load(ROOT/'evidence'/name/f'seed{seed}'/cond/f'd1_{backbone}.pt',weights_only=True)
                    w = wpack['weights']
                    total = float((e['m']*w).sum())
                    delta1 = abs(float((e['m']*global_mass(e['m'],w)).sum())-total)
                    delta2 = abs(float((e['m']*wpack['shuffled']).sum())-total)
                    check(max(delta1,delta2)<1e-9,'D1 exact mass')
                    massrows.append(dict(dataset=name,seed=seed,backbone=backbone,condition=cond,
                                         total_mass=total,global_residual=delta1,shuffle_residual=delta2))
                    alltargets = {'CE':e['own'],'GCE_Q0.7':e['own'],'KM_L1.0':shrink(es['KM'],1),
                                 'EXACT_MASS_GLOBAL':shrink(e,global_mass(e['m'],w)),
                                 'MASS_MATCHED_SHUFFLE':shrink(e,wpack['shuffled'])}
                    alltargets.update({f'SUPPORT_L{l}':shrink(e,l) for l in CONFIG['strengths']})
                    group = groups(e)
                    for a,b in CONFIG['group_pairs']:
                        alltargets[f'GROUP_A{a}_C{b}'] = shrink(e,torch.where(group==0,a,torch.where(group==1,b,bestglobal)).double())
                    for arm,target in alltargets.items():
                        check(state_hash({'target':target})==result('D1',arm)['target_hash'],'D1 reconstructed target '+arm)
                    probe = read(decision_path('C1',name,seed,backbone,cond))
                    full = read(decision_path('C1',name,seed,backbone,cond,'_full'))
                    prior = list(candidates())
                    for rd,keep in zip(probe['rounds'],[6,2]):
                        expected_scores = []
                        for arm in prior:
                            r = result('C1',arm)
                            logs = logs_by_folder[canonical(folder_for('C1',name,seed,backbone,cond,arm))]
                            top = first_best(logs[:rd['prefix']])
                            s,l = candidates()[arm]
                            expected_scores.append(dict(arm=arm,support=s,strength=l,valid_correct=top['validation_correct'],
                                valid_total=r['valid_total'],valid_nll=top['validation_nll'],best_epoch=top['epoch'],epoch=rd['prefix']))
                        check(rd['scores']==expected_scores,'C1 prefix rawlog reconstruction')
                        check(rd['survivors']==probe_survivors(expected_scores,keep,rd['prefix']),'C1 survivors')
                        check('S32_L1.0' in rd['survivors'],'Forced survivor')
                        prior = rd['survivors']
                    final_scores = [score(a,result('C1',a),*candidates()[a]) for a in prior]+[score('CE',result('D1','CE'))]
                    check(final_scores==probe['final_scores'],'C1 final race scores')
                    check(min(final_scores,key=rank_key)['arm']==probe['PROBE'],'PROBE rank')
                    grid = [score('CE',result('D1','CE'))]+[score(a,result('C1',a),s,l) for a,(s,l) in candidates().items()]
                    check(grid==full['scores'] and min(grid,key=rank_key)['arm']==full['FULL_VAL'],'FULL_VAL rank')
                    paid_before_freeze = 0
                    for arm,(s,l) in candidates().items():
                        r = result('C1',arm)
                        check(state_hash({'target':shrink(es[s],l)})==r['target_hash'],'C1 reconstructed target')
                        segments = [json.loads(line) for line in (folder_for('C1',name,seed,backbone,cond,arm)/'segments.jsonl').read_text().splitlines()]
                        allowed = 200 if arm in prior else 75 if arm in probe['rounds'][0]['survivors'] else 25
                        early = [sg for sg in segments if sg['finished']<=probe['frozen']]
                        check(max(sg['end_epoch'] for sg in early)==allowed,'Causal prefix limit')
                        paid_before_freeze += sum(sg['end_epoch']-sg['start_epoch'] for sg in early)
                        for sg in segments:
                            if sg['end_epoch']>allowed:
                                check(sg['finished']>probe['frozen'],'Full training before PROBE')
                    check(paid_before_freeze==1000,'Race paid candidate updates')
                    check(probe['frozen']<=full['frozen'],'Freeze chronology')
                    checks['causal_C1_cells'] += 1
                    checks['exact_mass_D1_cells'] += 1
                    if cond=='uniform0.4':
                        for stage,arm in [('D1',d1['selected']),('C1','S32_L1.0')]:
                            folder = folder_for(stage,name,seed,backbone,cond,arm)
                            ck = torch.load(folder/'checkpoint.pt',weights_only=True)
                            student = Student(backbone,values,graph['encoder'],seed)
                            student.set_state(ck['model'])
                            student.model.eval()
                            with torch.no_grad():
                                error = float((student.forward().softmax(1).cpu()-torch.load(folder/'probabilities.pt',weights_only=True)).abs().max())
                            student.check_frozen()
                            check(error<3e-7,'Independent checkpoint reload')
                            reloads.append(dict(folder=canonical(folder),max_probability_error=error))
                            del student
                    if seed==0 and name in ('cora','corafull','amazonpho'):
                        for arm in ['CE','SUPPORT_L1.0','GCE_Q0.7','KM_L1.0']:
                            oldarm = 'CE' if arm=='CE' else arm
                            old = EXP/'lcss_strength_support_20261024/fits/B1'/name/backbone/cond/oldarm/'probabilities.pt'
                            if old.exists():
                                new = folder_for('D1',name,seed,backbone,cond,arm)/'probabilities.pt'
                                error = float((torch.load(old,weights_only=True)-torch.load(new,weights_only=True)).abs().max())
                                bridges.append(dict(dataset=name,backbone=backbone,condition=cond,arm=arm,max_probability_error=error))
            del graph,values,es
            HIDDEN.clear()
            torch.cuda.empty_cache()
    csv_save(ROOT/'checkpoint_reload_audit.csv',reloads)
    csv_save(ROOT/'historical_seed0_bridges.csv',bridges)
    csv_save(ROOT/'exact_mass_audit.csv',massrows)
    checks['independent_checkpoint_reload_samples'] = len(reloads)
    testing = read(ROOT/'testing_started.json')
    check(testing['started']>=read(ROOT/'selections_frozen.json')['frozen'],'Test timing')
    save(ROOT/'audit.json',dict(status='passed' if not errors else 'failed',finished=now(),checks=dict(checks),
         errors=errors,unique_fits=len(registry),reload_max_error=max(r['max_probability_error'] for r in reloads),
         historical_seed0_max_error=max((r['max_probability_error'] for r in bridges),default=None),
         free_bytes=disk_guard(),historical_aliases=0))
    if errors:
        raise RuntimeError('Audit failed: '+str(errors[:10]))
