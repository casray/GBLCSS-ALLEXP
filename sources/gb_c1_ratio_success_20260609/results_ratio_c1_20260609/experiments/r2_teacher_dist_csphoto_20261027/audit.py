"""Independent CPU model/metric and five-graph table audit after all fits freeze."""
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HIP_VISIBLE_DEVICES']='-1'
os.environ['ROCR_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parent
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8'))
ROOT=Path(CFG['runtime'])
WORK=CODE.parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def table(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    torch.set_num_threads(1)
    sys.path.insert(0,str(WORK/'GB-LCSS_code/src'))
    from gblcss.models import MLP
    cp=WORK/'experiments/new_graph_mlp_seed0_20261009/common.py'
    spec=importlib.util.spec_from_file_location('csphoto_audit_common',cp)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    frozen=read(ROOT/'selections_frozen.json');fd=sha(ROOT/'selections_frozen.json')
    assert read(ROOT/'testing_started.json')['freeze_sha256']==fd
    assert sha(ROOT/'teachers_frozen.json')==frozen['teacher_freeze_sha256']
    assert sha(ROOT/'student_refs.json')==frozen['student_refs_sha256']
    assert sha(ROOT/'teacher_refs.json')==read(ROOT/'teachers_frozen.json')['refs_sha256']
    for name,digest in frozen['choices'].items():assert sha(ROOT/name)==digest
    for manifest in ('sources','inputs'):
        for name,digest in read(ROOT/(manifest+'.json')).items():assert sha(name)==digest,name
    students=read(ROOT/'student_refs.json');teachers=read(ROOT/'teacher_refs.json')
    assert len(students)==60 and len(teachers)==60
    assert {(r['dataset'],r['seed'],r['condition'],r['route']) for r in students}=={
        (n,s,c,a) for n in CFG['datasets'] for s in CFG['seeds'] for c in CFG['conditions'] for a in CFG['teacher_routes']}
    for r in students:
        result=read(Path(r['folder'])/'result.json');t=teachers[f"{r['dataset']}/seed{r['seed']}/{r['condition']}/{r['route']}"]
        assert result['source_sha256']==sha(ROOT/'sources.json')
        assert result['teacher_probability_sha256']==t['probabilities_sha256']
        for file,field in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256'),('result.json','result_sha256')]:
            assert sha(Path(r['folder'])/file)==r[field]
    previous=Path(CFG['previous_dist_runtime']);pf=read(previous/'selections_frozen.json')
    assert sha(previous/'student_refs.json')==pf['student_refs_sha256']
    oldrefs=read(previous/'teacher_refs.json')
    assert sha(previous/'teacher_refs.json')==read(previous/'teachers_frozen.json')['refs_sha256']
    rows=table(ROOT/'tables_five_graphs/results.csv');assert len(rows)==300
    measured=[];reloads=[];inits={};splitcache={}
    for n in ['cora','citeseer','corafull','coauthorcs','amazonpho']:
        raw=torch.load(common.DATA_FILES[n],map_location='cpu',weights_only=False)[0]
        x=(raw['x'] if isinstance(raw,dict) else raw.x).float();y=(raw['y'] if isinstance(raw,dict) else raw.y).long()
        classes=int(y.max())+1
        for r in (a for a in rows if a['dataset']==n):
            s=int(r['seed']);c=r['condition'];route=r['route'];kind=r['kind'];folder=Path(r['folder'])
            result=read(folder/'result.json');prob=torch.load(folder/'probabilities.pt',map_location='cpu',weights_only=True)
            assert sha(folder/'probabilities.pt')==r['probabilities_sha256']==result['probabilities_sha256']
            if (n,s) not in splitcache:
                path=(WORK/'experiments/sc_support_channel_pilot_20261023/diagnostics/e1'/n/f'seed{s}/geometry.pt') if s<3 else Path(CFG['new_main_runtime'])/'inputs'/n/f'seed{s}/geometry.pt'
                geo=torch.load(path,map_location='cpu',weights_only=False)
                for key,ids in zip(('train','valid','test'),common.percent_split_indices(len(x),s)):
                    assert torch.equal(geo[key],torch.as_tensor(ids).long())
                splitcache[n,s]={k:geo[k] for k in ('train','valid','test')};del geo
            te=splitcache[n,s]['test'];pred=prob[te].argmax(1)
            accuracy=100*float((pred==y[te]).double().mean())
            macro=100*f1_score(y[te].numpy(),pred.numpy(),labels=list(range(classes)),average='macro',zero_division=0)
            assert abs(float(r['accuracy'])-accuracy)<1e-9 and abs(float(r['macro_f1'])-macro)<1e-8
            measured.append(dict(dataset=n,seed=s,condition=c,route=route,kind=kind,accuracy=accuracy,macro_f1=macro,validation=float(r['validation'])))
            if n in CFG['datasets']:
                logs=[json.loads(v) for v in (folder/'epochs.jsonl').read_text().splitlines()]
                assert [a['epoch'] for a in logs]==list(range(1,201))
                best=max(logs,key=lambda a:a['validation_correct'])
                assert best['epoch']==result['best_epoch'] and best['validation_correct']==result['valid_correct']
                ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert sha(folder/'checkpoint.pt')==result['checkpoint_sha256']
                model=MLP(x.shape[1],64,classes,2,.5).eval()
                if kind=='teacher':
                    ep=Path(ck['encoder_path']);assert sha(ep)==ck['encoder_file_sha256']
                    enc=torch.load(ep,map_location='cpu',weights_only=True)
                    state={k:enc['backbone.'+k] for k in model.state_dict() if not k.startswith('layers.1.')}
                    state.update({'layers.1.'+k:v for k,v in ck['model'].items()})
                else:
                    state=ck['model'];inits.setdefault((n,s,c),set()).add((result['initial_hash'],result['initial_rng_hash']))
                    assert (result['backbone'],result['seed'],result['lr'],result['weight_decay'])==('DIST',s,.01,.0005)
                model.load_state_dict(state)
                with torch.inference_mode():actual=model(x).softmax(1)
                diff=float((actual-prob).abs().max())
                assert diff<3e-6 and torch.equal(actual.argmax(1),prob.argmax(1))
                reloads.append(dict(dataset=n,seed=s,condition=c,route=route,kind=kind,max_error=diff))
                del actual,model
        del raw,x,y
        print('CPU_AUDIT '+n,flush=True)
    assert len(reloads)==120 and len(inits)==30 and all(len(v)==1 for v in inits.values())
    verified=0
    for r in table(ROOT/'tables_five_graphs/summary.csv'):
        for m in ('accuracy','macro_f1','validation'):
            values=[]
            for s in CFG['seeds']:
                chosen=[a for a in measured if a['seed']==s and a['route']==r['route'] and a['kind']==r['kind']
                    and (r['dataset']=='ALL' or r['dataset']==a['dataset'])
                    and (a['condition']!='clean' if r['condition']=='NOISY_MACRO' else a['condition']==r['condition'])]
                assert chosen;values.append(np.mean([a[m] for a in chosen]))
            assert abs(np.mean(values)-float(r[m+'_mean']))<1e-9
            assert abs(np.std(values,ddof=1)-float(r[m+'_std_ddof1']))<1e-9
            verified+=1
    assert fd==sha(ROOT/'selections_frozen.json') and not torch.cuda.is_initialized()
    result=dict(status='PASSED',new_students=60,new_updates=12000,new_teacher_student_reloads=120,
        max_probability_error=max(r['max_error'] for r in reloads),merged_metric_rows=300,mean_sd_checks=verified,
        paired_initializations=len(inits),reloads=reloads)
    write(ROOT/'audit.json',result);shutil.copy2(ROOT/'audit.json',CODE/'audit.json')
    report=ROOT/'REPORT_zh.md';text=report.read_text(encoding='utf-8').replace('最终结论待独立CPU审计。','独立CPU审计已通过。')
    text+='\n独立审计：120新教师/学生模型重载、300合并指标、种子均值/样本SD、输入源码hash和冻结链通过。全部任务完成，无自动更多实验或上传。\n'
    report.write_text(text,encoding='utf-8');shutil.copy2(report,CODE/'REPORT_zh.md')
    complete=dict(status='COMPLETE_AUDITED',time=datetime.now(timezone.utc).isoformat(),new_students=60,
        five_graph_student_endpoints=150,new_updates=12000,audit_sha256=sha(ROOT/'audit.json'))
    write(ROOT/'complete.json',complete);shutil.copy2(ROOT/'complete.json',CODE/'complete.json')
    for name in ('sources','inputs','teacher_refs','student_refs','selections_frozen','teachers_frozen'):
        shutil.copy2(ROOT/(name+'.json'),CODE/(name+'.json'))
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
