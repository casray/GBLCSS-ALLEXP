from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from core import state_hash
from protocol import ROOT, WORKSPACE, load_dataset, raw_data, read, write, sha
from gblcss.models import MLP
from run_experiment import CONFIG, verify_inputs, require_gpu


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def audit_and_score(folder, result, phase):
    name, condition = result['dataset'], result['condition']
    assets = torch.load(ROOT/'inputs'/name/'assets.pt', weights_only=True)
    probabilities = torch.load(folder/'probabilities.pt', weights_only=True)
    if probabilities.shape != (len(raw_data(name).x), assets['classes']):
        raise RuntimeError('Prediction shape mismatch')
    torch.testing.assert_close(probabilities.sum(1), torch.ones(len(probabilities)), atol=1e-5, rtol=0)
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError('Invalid probabilities')
    trajectory = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
    counts = [row['valid_correct'] for row in trajectory]
    selected_epoch = counts.index(max(counts))+1
    if selected_epoch != result['best_epoch']:
        raise RuntimeError('Validation checkpoint was not the first strict maximum')
    clean = raw_data(name).y.long()
    val = assets['valid']
    val_correct = int((probabilities[val].argmax(1)==clean[val]).sum())
    if abs(val_correct/len(val)-result['validation']) > 1e-12:
        raise RuntimeError('Saved validation predictions mismatch')
    data = load_dataset(name)
    if phase == 'probes':
        layers = 1 if result['kind']=='linear' else 2
        model = MLP(data['h'].shape[1], 64, data['classes'], layers, .5).cuda().eval()
        model.load_state_dict(torch.load(folder/'model.pt', weights_only=True))
        x = data['h']
    else:
        if sha(folder/'student.pt') != result['student_sha256'] or sha(folder/'probabilities.pt') != result['probabilities_sha256']:
            raise RuntimeError('Saved model/prediction hash mismatch')
        model = MLP(data['x'].shape[1],64,data['classes'],2,.5).cuda().eval()
        state = torch.load(folder/'student.pt', weights_only=True)
        model.load_state_dict(state)
        x = data['x']
        if result['route']=='R2':
            reference = read(Path(result['pretrained_source'])/'result.json')
            encoder = {k:v for k,v in state.items() if not k.startswith('layers.1.')}
            if state_hash(encoder) != reference['encoder_hash']:
                raise RuntimeError('R2 encoder differs from frozen stage 1')
    with torch.no_grad():
        reconstructed = model(x).softmax(1).cpu()
    maximum_difference = float((reconstructed-probabilities).abs().max())
    if maximum_difference != 0:
        raise RuntimeError('Checkpoint reload failed')
    test = assets['test']
    pred = probabilities[test].argmax(1)
    accuracy = float((pred==clean[test]).double().mean())
    nll = float(-probabilities[test,clean[test]].clamp_min(1e-30).log().double().mean())
    one_hot = torch.nn.functional.one_hot(clean[test],assets['classes']).float()
    brier = float((probabilities[test]-one_hot).square().sum(1).double().mean())
    row = dict(phase=phase,dataset=name,condition=condition,route=result.get('route',result.get('kind')),
               rule=result.get('rule','T0'),weight=result.get('weight'),seed=0,n_seeds=1,seed_std=None,
               test_percent=100*accuracy,validation_percent=100*result['validation'],best_epoch=result['best_epoch'],
               test_nll=nll,test_brier=brier,seconds=result['seconds'],reload_max_difference=maximum_difference,
               legacy=result.get('legacy',False),folder=str(folder))
    return row


def main():
    require_gpu(); verify_inputs()
    if read(ROOT/'progress.json')['status'] != 'complete':
        raise RuntimeError('Full registered grid must finish before test evaluation')
    rows = []
    for phase, expected in (('main',144),('legacy',24),('probes',24)):
        files = sorted((ROOT/phase).glob('*/*/*/result.json'))
        if len(files) != expected:
            raise RuntimeError(f'{phase}: {len(files)} of {expected}')
        for i,path in enumerate(files,1):
            result = read(path)
            if result['status'] != 'ok':
                raise RuntimeError('Failed cell: '+str(path))
            rows.append(audit_and_score(path.parent,result,phase))
        print(f'AUDITED {phase}: {len(files)}',flush=True)
    lookup = {(r['phase'],r['dataset'],r['condition'],r['route'],r['rule']):r for r in rows}
    deltas=[]
    comparisons = [('T1-T0','T1','T0'),('T3-T2','T3','T2')]
    for dataset in CONFIG['datasets']:
        for condition in CONFIG['conditions']:
            for route in CONFIG['representation_routes']:
                for comparison,a,b in comparisons:
                    value=lookup['main',dataset,condition,route,a]['test_percent']-lookup['main',dataset,condition,route,b]['test_percent']
                    deltas.append(dict(dataset=dataset,condition=condition,route=route,comparison=comparison,delta_pp=value))
                scores={t:lookup['main',dataset,condition,route,t]['test_percent'] for t in CONFIG['supervision_rules']}
                deltas.append(dict(dataset=dataset,condition=condition,route=route,comparison='T3-max(T1,T2)',
                                   delta_pp=scores['T3']-max(scores['T1'],scores['T2'])))
                if route!='R0':
                    for rule in CONFIG['supervision_rules']:
                        deltas.append(dict(dataset=dataset,condition=condition,route=route,comparison=f'{rule}-R0{rule}',
                            delta_pp=scores[rule]-lookup['main',dataset,condition,'R0',rule]['test_percent']))
    summary=[]
    for route in CONFIG['representation_routes']:
        for rule in CONFIG['supervision_rules']:
            for subset in ('noisy','clean'):
                selected=[r for r in rows if r['phase']=='main' and r['route']==route and r['rule']==rule
                          and ((r['condition']=='clean')==(subset=='clean'))]
                summary.append(dict(route=route,rule=rule,subset=subset,n_cells=len(selected),n_seeds=1,seed_std=None,
                     test_macro_percent=float(np.mean([r['test_percent'] for r in selected])),
                     validation_macro_percent=float(np.mean([r['validation_percent'] for r in selected]))))
    gates={}
    for route in ('R1','R2'):
        part=[r for r in deltas if r['route']==route and r['comparison']=='T1-R0T1' and r['condition']!='clean']
        by_dataset={d:float(np.mean([r['delta_pp'] for r in part if r['dataset']==d])) for d in CONFIG['datasets']}
        gates[route+'_transfer']=dict(per_dataset_delta_pp=by_dataset,pass_pilot=sum(v>=1 for v in by_dataset.values())>=2)
    for route in CONFIG['representation_routes']:
        for comparison in ('T1-T0','T3-T2'):
            part=[r for r in deltas if r['route']==route and r['comparison']==comparison and r['condition']!='clean']
            clean=[r['delta_pp'] for r in deltas if r['route']==route and r['comparison']==comparison and r['condition']=='clean']
            by_dataset={d:float(np.mean([r['delta_pp'] for r in part if r['dataset']==d])) for d in CONFIG['datasets']}
            mean=float(np.mean([r['delta_pp'] for r in part]))
            positive=sum(r['delta_pp']>0 for r in part)
            gates[route+'_'+comparison]=dict(mean_delta_pp=mean,positive_cells=positive,total_cells=len(part),
                per_dataset_delta_pp=by_dataset,minimum_clean_delta_pp=min(clean),
                pass_pilot=mean>=.5 and positive>=6 and sum(v>0 for v in by_dataset.values())>=2 and min(clean)>=-1)
    candidates=[r for r in summary if r['subset']=='noisy']
    winner=max(candidates,key=lambda r:r['validation_macro_percent'])
    sim=[]
    with (WORKSPACE/'experiments/baseline_main5seed_20261009/per_seed_results.csv').open(encoding='utf-8-sig') as handle:
        for old in csv.DictReader(handle):
            if old['method']=='simmlp' and old['seed']=='0' and old['dataset'] in CONFIG['datasets'] and old['noise_rate']=='0.4':
                d=old['dataset']
                sim.append(dict(dataset=d,condition='uniform0.4',simmlp_seed0_percent=float(old['test_percent']),
                    selected_route=winner['route'],selected_rule=winner['rule'],
                    selected_test_percent=lookup['main',d,'uniform0.4',winner['route'],winner['rule']]['test_percent']))
    if len(sim)!=3:
        raise RuntimeError('Missing same-seed SimMLP comparison')
    csv_write(ROOT/'results.csv',rows);csv_write(ROOT/'paired_deltas.csv',deltas)
    csv_write(ROOT/'summary.csv',summary);csv_write(ROOT/'simmlp_reference.csv',sim)
    write(ROOT/'decision.json',dict(pilot_gates=gates,validation_selected_global_candidate=winner,
          seed_count=1,seed_std=None,significance_tested=False,automatic_seed_expansion=False))
    write(ROOT/'audit.json',dict(status='passed',checkpoints_reloaded=len(rows),maximum_reload_difference=max(r['reload_max_difference'] for r in rows),
          first_validation_maximum_verified=True,frozen_inputs_unchanged=True,r2_encoder_and_bn_frozen=True,
          all_deployed_students_use_only_x=True,main_count=144,legacy_count=24,probe_count=24))
    lines=['# 表示迁移与鲁棒监督：首轮实验结论','',
           '这是三数据集、四条件、单协议种子0的探索结果。下表是九个带噪条件的宏平均；不是多种子均值，不能计算种子标准差或声称显著。',
           '', '| 表示方式 | 含噪CE | LCSS | small-loss | LCSS+small-loss |','|---|---:|---:|---:|---:|']
    for route in CONFIG['representation_routes']:
        values=[next(r['test_macro_percent'] for r in summary if r['route']==route and r['rule']==t and r['subset']=='noisy')
                for t in CONFIG['supervision_rules']]
        lines.append('| '+route+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','R0不对齐；R1联合对齐；R2先100轮表示对齐，再冻结编码器及BN训练100轮分类头。所有部署模型均为原始X的两层64维MLP。',
            '', '## 冻结门槛结果','']
    for key,value in gates.items():
        lines.append(f'- {key}: '+json.dumps(value,ensure_ascii=False))
    lines+=['','## 解释边界','',
            '- 同一监督下的R1/R2-R0回答表示迁移增量；同一表示下T1-T0和T3-T2回答LCSS剩余作用；T3-max(T1,T2)回答组合增量。',
            '- 旧目标桥接结果单列在results.csv的legacy部分；不能把含自身投票的历史组合当作当前LOO组合的重复。',
            '- H探针接收图预计算表示，不是严格X-only部署对照，也不是数学上界。',
            '- 所有检查点仅按干净验证集选择。lambda仅按Cora/CiteSeer的T0验证结果选择。测试标签在完整网格结束后首次用于本轮评分。',
            '- 三个图共享各自固定的SSL预训练表示，不能称独立SSL复现。当前没有自动扩种子或运行新分支。',
            '', '## 核查','',f'- 已重载并核查{len(rows)}个模型和预测；192个逻辑单元完整，冻结输入未变。',
            '- 原始逐轮日志、配置、检查点、概率和来源指纹均保留；完整配对差值见paired_deltas.csv。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,gates=gates,winner=winner),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
