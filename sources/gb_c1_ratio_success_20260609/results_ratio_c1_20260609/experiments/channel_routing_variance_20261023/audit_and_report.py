"""Independent checkpoint replay and readable tables from frozen results."""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from run_stage0 import ROOT,OLD,CONFIG,f
from run_routing_l1 import cells


def read_csv(name):
    return list(csv.DictReader((ROOT/name).open(encoding='utf-8-sig')))


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])


def main():
    torch.set_num_threads(4)
    if (ROOT/'fit.lock').exists():
        raise RuntimeError('Fitting still active')
    audit=[]
    for name in CONFIG['datasets']:
        d=f.data(name)
        for condition in CONFIG['conditions']:
            for arm in CONFIG['arms']:
                folder=ROOT/'fits'/name/'protocol0/student1'/condition/arm
                meta=f.read(folder/'result.json')
                model=f.MLP(d['x'].shape[1],64,d['classes'],2,.5).cuda().eval()
                model.load_state_dict(f.load(folder/'student.pt'))
                expected=f.load(folder/'probabilities.pt')
                with torch.no_grad():actual=model(d['x']).softmax(1).cpu()
                err=float((actual-expected).abs().max())
                if err>1e-7 or not torch.equal(actual.argmax(1),expected.argmax(1)):
                    raise RuntimeError('Checkpoint replay differs')
                if arm=='C_head':
                    encoder=f.load(Path(meta['encoder']))
                    state=model.state_dict()
                    for k,v in encoder.items():
                        if k.startswith(('backbone.layers.0.','backbone.batch_norms.')):
                            if not torch.equal(v.cpu(),state[k.removeprefix('backbone.')].cpu()):
                                raise RuntimeError('Frozen encoder differs')
                audit.append(dict(dataset=name,condition=condition,arm=arm,max_probability_error=err,encoder_preserved=arm=='C_head'))
    fusion=read_csv('L1_results.csv')
    choices={(r['dataset'],r['condition']):r for r in f.read(ROOT/'L1_selections.json')}
    index={(r['dataset'],r['condition'],r['method']):r for r in fusion}
    replay=[]
    for cell in cells():
        key=cell['dataset'],cell['condition'];sel=choices[key]
        a,c=cell['a'],cell['c'];vi=cell['geo']['valid'].numpy();ti=cell['geo']['test'].numpy();y=cell['y']
        w=max(np.linspace(0,1,21),key=lambda x:(int(np.sum((x*a[vi]+(1-x)*c[vi]).argmax(1)==y[vi])),x))
        if w!=sel['weight']:raise RuntimeError('Global weight was not validation selected')
        threshold=np.inf if sel['confidence_threshold'] is None else sel['confidence_threshold']
        score=1-cell['teacher'].max(1)
        def tq(t):
            switched=(score[vi]>=t)&(a[vi].argmax(1)!=c[vi].argmax(1))
            return int(np.sum(np.where(switched,c[vi].argmax(1),a[vi].argmax(1))==y[vi])),-int(switched.sum()),t
        if threshold!=max([*np.linspace(0,1,21),np.inf],key=tq):raise RuntimeError('Confidence threshold mismatch')
        predictions=dict(A0=a.argmax(1),C0=c.argmax(1),MEAN=((a+c)/2).argmax(1),GLOBAL=(w*a+(1-w)*c).argmax(1),
            TCONF=np.where(score>=threshold,c.argmax(1),a.argmax(1)))
        out=ROOT/'fusion_predictions'/cell['dataset']/cell['condition']
        out.mkdir(parents=True,exist_ok=True)
        for method,p in predictions.items():
            for split,idx in [('valid',vi),('test',ti)]:
                val=float(np.mean(p[idx]==y[idx]))
                if abs(val-float(index[*key,method][split+'_accuracy']))>1e-12:raise RuntimeError('Fusion score mismatch')
        np.savez_compressed(out/'predictions.npz',**predictions)
        replay.append(dict(dataset=key[0],condition=key[1],selection_reproduced=True,all_predictions_reproduced=True))
    manifests=['fit_sources.json','fit_inputs.json','L1_sources.json']
    count={}
    for filename in manifests:
        vals=f.read(ROOT/filename)
        for p,digest in vals.items():
            if f.sha(Path(p))!=digest:raise RuntimeError('Changed source/input '+p)
        count[filename]=len(vals)
    gate=f.read(ROOT/'stage0_decision.json');gaps=read_csv('stage0_gaps.csv')
    for name in CONFIG['datasets']:
        rs=[r for r in gaps if r['dataset']==name and r['split']=='test' and r['condition']!='clean']
        value=np.mean([100*(float(r['cross_union'])-max(float(r['same_A_union']),float(r['same_C_union']))) for r in rs])
        if abs(value-gate['gains_pp'][name])>1e-10:raise RuntimeError('Stage0 gate calculation mismatch')
    if sum(gate['gains_pp'][n]>=2 for n in ['cora','citeseer','corafull'])<2:raise RuntimeError('Stage0 inconsistent')
    l1=f.read(ROOT/'L1_decision.json')
    if l1['passed']:raise RuntimeError('This closeout expects the observed L1 stop; handle later branches separately')
    f.save_json(ROOT/'final_audit.json',dict(passed=True,time=datetime.now().astimezone().isoformat(),
        checkpoint_replays=audit,fusion_replays=replay,manifests=count,stage0_gate_reproduced=True,
        total_new_formal_fits=40,smoke_fits=2,new_router_fits=0))

    pairs=read_csv('pair_results.csv');models=read_csv('model_results.csv')
    title={'cora':'Cora','citeseer':'CiteSeer','corafull':'CoraFull','coauthorcs':'CS','amazonpho':'Photo'}
    def avg(rows,key):return 100*float(np.mean([float(r[key]) for r in rows]))
    def psubset(n,p):return [r for r in pairs if r['dataset']==n and r['pair']==p and r['split']=='test' and r['condition']!='clean']
    def fsubset(n,m,clean=False):return [r for r in fusion if r['dataset']==n and r['method']==m and (r['condition']=='clean')==clean]
    union_rows=[];fusion_rows=[];f1_rows=[];clean_rows=[];weights=[];model_rows=[];disagreement=[]
    for name in CONFIG['datasets']:
        union_rows.append([title[name],*[f'{avg(psubset(name,p),"union"):.3f}' for p in ['A0_C0','A0_A1','C0_C1']],
                           f'{gate["gains_pp"][name]:+.3f}',f'{gate["mean_cross_pair_sensitivity"][name]:+.3f}'])
        fusion_rows.append([title[name],*[f'{avg(fsubset(name,m),"test_accuracy"):.3f}' for m in ['A0','C0','MEAN','GLOBAL','TCONF']],
                            f'{l1["noisy_gain_pp"][name]:+.3f}'])
        f1_rows.append([title[name],*[f'{avg(fsubset(name,m),"test_macro_f1"):.3f}' for m in ['A0','C0','MEAN','GLOBAL','TCONF']]])
        clean_rows.append([title[name],*[f'{avg(fsubset(name,m,True),"test_accuracy"):.3f}' for m in ['A0','C0','MEAN','GLOBAL','TCONF']]])
        weights.append([title[name],*[choices[name,c]['weight'] for c in CONFIG['conditions']]])
        for arm in CONFIG['arms']:
            per_seed=[avg([r for r in models if r['dataset']==name and r['arm']==arm and r['student_seed']==str(seed) and r['split']=='test' and r['condition']!='clean'],'accuracy') for seed in (0,1)]
            model_rows.append([title[name],arm,f'{per_seed[0]:.3f}',f'{per_seed[1]:.3f}',f'{np.mean(per_seed):.3f}',f'{np.std(per_seed,ddof=1):.3f}'])
        disagreement.append([title[name],*[f'{avg(psubset(name,"A0_C0"),k):.3f}' for k in ['disagreement','a_only','c_only','both_wrong']]])
    text='# 完整结果表\n\n日期：2026-10-23。所有准确率/F1/并集为百分数，差值为百分点。带噪均值等权平均U0.4/U0.6/Pair0.4。\n\n'
    text+='## 第0层：并集\n\n'+md_table(['数据集','跨通道A0∪C0','同通道A0∪A1','同通道C0∪C1','主门槛差值pp','四跨配对平均差pp'],union_rows)+'\n\n'
    text+='主差值先逐条件减去两个同通道并集的较大值，再平均。四跨配对平均为敏感性描述。\n\n'
    text+='## 第1层：带噪准确率\n\n'+md_table(['数据集','A0','C0','直接平均','验证选全局权重','教师置信度路由','全局混合验证增量pp'],fusion_rows)+'\n\n'
    text+='## 带噪macro-F1\n\n'+md_table(['数据集','A0','C0','直接平均','全局权重','教师置信度路由'],f1_rows)+'\n\n'
    text+='## Clean准确率\n\n'+md_table(['数据集','A0','C0','直接平均','全局权重','教师置信度路由'],clean_rows)+'\n\n'
    text+='## 两个学生初始化的均值与样本标准差\n\n先在每个学生seed内平均三种噪声，再对两个学生seed计算SD。数据划分、标签噪声、教师和R2编码器均固定。逐条件均值/SD见model_summary.csv。\n\n'+md_table(['数据集','模型','student0','student1','均值','SD'],model_rows)+'\n\n'
    text+='## 全局混合权重w\n\n目标为w*A0+(1-w)*C0；由验证准确率选择。\n\n'+md_table(['数据集',*CONFIG['conditions']],weights)+'\n\n'
    text+='## A0/C0错误组成\n\n以下各列均占全部测试节点的比例；两者都错包括预测一致和预测分歧的节点。\n\n'+md_table(['数据集','预测分歧','仅A0正确','仅C0正确','两者都错'],disagreement)+'\n\n'
    text+='硬路由的可实现比例见L1_results.csv中TCONF的attainable_fraction。概率混合可能把两个错误argmax修正为第三个类别，因此其该字段留空；它的整体准确率单独报告。\n'
    (ROOT/'RESULT_TABLES.md').write_text(text,encoding='utf-8')
    f.save_json(ROOT/'status.json',dict(status='complete',formal_fits=40,smoke_fits=2,stage0='passed',L1='failed',
        later_layers='not_run',new_router_fits=0,final_audit=f.sha(ROOT/'final_audit.json')))
    print(json.dumps(dict(passed=True,checkpoint_replays=len(audit),fusion_cells=len(replay),manifests=count),indent=2))


if __name__=='__main__':main()
