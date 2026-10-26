"""Frozen student-route screening, all selections saved before test scoring."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from router import CONFIG, FEATURES, fit_router, apply_route, predict_model, student_features
from metrics import accuracy_f1, pair_metrics, route_metrics
from routing_kernels import choose_threshold

OLD = ROOT.parent/CONFIG['source']
sys.path.insert(0, str(OLD))
import fit_pipeline as legacy

INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load(path):
    INPUTS[str(path)] = sha(path)
    return torch.load(path, map_location='cpu', weights_only=True)


def table(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)


def log(event, **values):
    record = dict(time=datetime.now(timezone.utc).isoformat(), event=event, **values)
    with (ROOT/'raw.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(json.dumps(record), flush=True)


def probability(path):
    p = load(path).numpy()
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities: '+str(path))
    return p


def select_all():
    records = []
    for name in CONFIG['datasets']:
        geo = load(OLD/'diagnostics/e1'/name/'seed0/geometry.pt')
        vi = geo['valid'].numpy()
        for condition in CONFIG['conditions']:
            folder = ROOT/'cells'/name/condition; folder.mkdir(parents=True, exist_ok=True)
            base = OLD/'fits/E4'/name/'seed0'/condition
            a = probability(base/'A300/probabilities.pt')
            c = probability(base/'C_head/probabilities.pt')
            teacher = probability(base/'A300/target.pt')
            state = load(OLD/'diagnostics/e1'/name/'seed0'/f'{condition}.pt')
            yv = state['valid_y'].numpy()
            r = fit_router(a, c, vi, yv)
            tconf = choose_threshold(yv, a[vi].argmax(1), c[vi].argmax(1), 1-teacher[vi].max(1))
            record = dict(dataset=name, condition=condition, threshold=r['threshold'],
                          global_weight=r['global_weight'], teacher_threshold=None if np.isinf(tconf) else tconf,
                          model=r['model'], fold_models=r['fold_models'], full_oof_models=r['full_oof_models'])
            save(folder/'selection.json', record)
            arrays = {k:r[k] for k in ['score','nested_pred','nested_score','nested05','global_nested','fold_ids','full_oof']}
            np.savez_compressed(folder/'routing.npz', **arrays, valid=vi, valid_y=yv)
            records.append(record)
            log('selected', dataset=name, condition=condition, threshold=r['threshold'],
                nested_gain_pp=100*float(np.mean(r['nested_pred']==yv)-np.mean(r['global_nested']==yv)))
    save(ROOT/'selections.json', records)
    save(ROOT/'selection_complete.json', dict(cells=len(records), time=datetime.now(timezone.utc).isoformat(),
                                              selections_sha256=sha(ROOT/'selections.json')))
    return records


def score_all(records):
    rows, auc_rows, audits = [], [], []
    for name in CONFIG['datasets']:
        raw_path = Path(legacy.r2.DATA_FILES[name]); INPUTS[str(raw_path)] = sha(raw_path)
        raw = legacy.r2.raw_data(name)
        y = raw.y.cpu().numpy()
        geo = load(OLD/'diagnostics/e1'/name/'seed0/geometry.pt')
        vi, ti = geo['valid'].numpy(), geo['test'].numpy()
        for selection in [r for r in records if r['dataset']==name]:
            cond = selection['condition']; folder = ROOT/'cells'/name/cond
            base = OLD/'fits/E4'/name/'seed0'/cond
            a = probability(base/'A300/probabilities.pt'); c = probability(base/'C_head/probabilities.pt')
            teacher = probability(base/'A300/target.pt')
            cached = np.load(folder/'routing.npz')
            assert np.array_equal(y[vi], cached['valid_y'])
            score = cached['score']; ap, cp = a.argmax(1), c.argmax(1)
            replay = predict_model(selection['model'], student_features(a,c))
            assert np.array_equal(replay, score)
            w = selection['global_weight']
            pred = dict(A=ap, C=cp, MEAN=((a+c)/2).argmax(1), GLOBAL=(w*a+(1-w)*c).argmax(1),
                        ROUTER=apply_route(ap,cp,score,selection['threshold']),
                        ROUTER05=apply_route(ap,cp,score,.5),
                        TCONF=apply_route(ap,cp,1-teacher.max(1),selection['teacher_threshold']))
            np.savez_compressed(folder/'predictions.npz', **pred, test=ti)
            pair = pair_metrics(y[ti],ap[ti],cp[ti])
            for method, p in pred.items():
                valid_p = {'GLOBAL':cached['global_nested'], 'ROUTER':cached['nested_pred'],
                           'ROUTER05':cached['nested05']}.get(method, p[vi])
                vm = accuracy_f1(y[vi], valid_p, a.shape[1]); tm = accuracy_f1(y[ti], p[ti], a.shape[1])
                row = dict(dataset=name,condition=cond,method=method,valid_accuracy=vm['accuracy'],
                           valid_macro_f1=vm['macro_f1'],test_accuracy=tm['accuracy'],test_macro_f1=tm['macro_f1'],
                           fullfit_valid_accuracy=float(np.mean(p[vi]==y[vi])), oracle=pair['union'],
                           best_single=pair['best_single'], gain_best_pp=100*(tm['accuracy']-pair['best_single']),
                           disagreement=pair['disagreement'], both_wrong=pair['both_wrong'],
                           validation_mode='nested' if method in ['GLOBAL','ROUTER','ROUTER05'] else 'fixed_or_full_validation')
                if method in ['ROUTER','ROUTER05','TCONF']:
                    rm = route_metrics(y[ti],ap[ti],cp[ti],p[ti]!=ap[ti])
                    assert abs(tm['accuracy']-np.mean(ap[ti]==y[ti])-rm['gain']) < 1e-12
                    row.update({k:rm[k] for k in ['rescued','harmed','switched','coverage']})
                    row['attainable_fraction']=(tm['accuracy']-pair['best_single'])/pair['headroom'] if pair['headroom']>0 else None
                rows.append(row)
            features = student_features(a,c)
            for split, idx in [('valid',vi),('test',ti)]:
                usable = (ap[idx]==y[idx]) ^ (cp[idx]==y[idx])
                route_y = (cp[idx][usable]==y[idx][usable]).astype(int)
                values = {'ROUTER':cached['nested_score'] if split=='valid' else score[idx]}
                values.update({feature:features[idx,j] for j,feature in enumerate(FEATURES)})
                for feature, value in values.items():
                    auc = float(roc_auc_score(route_y,value[usable])) if len(np.unique(route_y))==2 else None
                    auc_rows.append(dict(dataset=name,condition=cond,split=split,feature=feature,auc=auc,
                                         n=int(usable.sum()), positive=int(route_y.sum()), orientation='raw_increasing'))
            audits.append(dict(dataset=name,condition=cond,model_replay_max_error=float(np.max(abs(replay-score))),
                               net_correction_identity=True, validation_labels_exact=True))
    table(ROOT/'results.csv',rows); table(ROOT/'feature_auc.csv',auc_rows)
    save(ROOT/'prediction_audit.json', audits)
    return rows


def report(rows):
    index = {(r['dataset'],r['condition'],r['method']):r for r in rows}
    noisy, clean = {}, {}
    summary = []
    for name in CONFIG['datasets']:
        noise = [c for c in CONFIG['conditions'] if c!='clean']
        noisy[name] = float(np.mean([100*(index[name,c,'ROUTER']['valid_accuracy']-index[name,c,'GLOBAL']['valid_accuracy']) for c in noise]))
        clean[name] = 100*(index[name,'clean','ROUTER']['valid_accuracy']-index[name,'clean','GLOBAL']['valid_accuracy'])
        for method in ['A','C','MEAN','GLOBAL','ROUTER','ROUTER05','TCONF']:
            subset = [index[name,c,method] for c in noise]
            summary.append(dict(dataset=name,method=method,
                noisy_test_mean_pct=100*float(np.mean([r['test_accuracy'] for r in subset])),
                noisy_condition_std_pp=100*float(np.std([r['test_accuracy'] for r in subset],ddof=1)),
                noisy_macro_f1_pct=100*float(np.mean([r['test_macro_f1'] for r in subset])),
                noisy_valid_mean_pct=100*float(np.mean([r['valid_accuracy'] for r in subset])),
                clean_test_pct=100*index[name,'clean',method]['test_accuracy'],
                gain_best_pp=float(np.mean([r['gain_best_pp'] for r in subset])), seed_std=None, seeds=1))
    g = CONFIG['gate']
    passed = sum(noisy[n]>=g['gain_pp'] for n in g['primary'])>=g['required'] and min(clean.values())>=g['clean_floor_pp']
    decision = dict(passed=bool(passed), split='nested_validation',noisy_gain_global_pp=noisy,
                    clean_gain_global_pp=clean, next_step='design local feature controls' if passed else 'stop student-signal router expansion',
                    neural_fits=0, lr_fit_slots=20*(5*(4+1)+5+1), seed_confirmation=False)
    save(ROOT/'decision.json',decision); table(ROOT/'summary.csv',summary)
    lines = ['# 节点路由与全局融合：结果', '',
             '协议：5 图 × 4 条件 × seed0。复用概率，新增神经网络拟合 0 次；每单元 31 个路由拟合槽位。',
             '外层验证评估隔离新增路由训练与阈值选择；原学生选模已使用这些验证标签，本轮为探索。', '',
             '## 预注册验证门槛', '',
             '| 数据集 | 带噪 ROUTER−GLOBAL (pp) | clean 差值 (pp) |', '|---|---:|---:|']
    for name in CONFIG['datasets']: lines.append(f'| {name} | {noisy[name]:+.3f} | {clean[name]:+.3f} |')
    lines += ['', '**决定：'+('通过，后续设计局部统计增量对照。' if passed else '未通过，停止扩展当前学生信号路由。')+'**', '',
              '## 测试集：三种噪声条件平均准确率 (%)', '',
              '| 数据集 | A | C | 平均 | GLOBAL | ROUTER | 固定0.5路由 |', '|---|---:|---:|---:|---:|---:|---:|']
    si = {(r['dataset'],r['method']):r for r in summary}
    for name in CONFIG['datasets']:
        lines.append('| '+name+' | '+' | '.join(f"{si[name,m]['noisy_test_mean_pct']:.3f}" for m in ['A','C','MEAN','GLOBAL','ROUTER','ROUTER05'])+' |')
    lines += ['', '## 路由净纠错（测试集，每图合并三种噪声）', '',
              '| 数据集 | 救回 | 误伤 | 净数量 | ROUTER−GLOBAL pp |', '|---|---:|---:|---:|---:|']
    for name in CONFIG['datasets']:
        rr=[r for r in rows if r['dataset']==name and r['condition']!='clean' and r['method']=='ROUTER']
        rescue=sum(r['rescued'] for r in rr); harm=sum(r['harmed'] for r in rr)
        gain=si[name,'ROUTER']['noisy_test_mean_pct']-si[name,'GLOBAL']['noisy_test_mean_pct']
        lines.append(f'| {name} | {rescue} | {harm} | {rescue-harm:+d} | {gain:+.3f} |')
    lines += ['', '## 解释与可追溯范围', '',
        '- 主指标为全部节点净纠错；两者都错的节点保留在分母与改判覆盖率中。',
        '- AUROC 仅描述恰有一个学生正确时的可区分性。feature_auc.csv 保留特征原始方向，未按测试结果翻转。',
        '- 全局融合可预测第三个类别，硬选择的并集与兑现比例仅用于硬路由。',
        '- 单 seed 的种子标准差留空；summary.csv 的标准差是三个噪声条件间波动。',
        '- 测试结果在全部规则落盘后评分；旧模型和旧 L1 停止标记保持原样。',
        '- 当前假设是学生自身概率具有可学习的纠错线索；局部图/粒球信息的额外价值留待独立实验。',
        '', '文件：PLAN.md/config.json、sources.json/inputs.json、raw.jsonl、cells 下折模型及概率、results.csv、summary.csv、feature_auc.csv、prediction_audit.json。']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('complete',**decision)


def main():
    if (ROOT/'complete.json').exists():
        print('Already complete; no duplicate execution'); return
    with (ROOT/'run.lock').open('x') as stream: stream.write(str(os.getpid()))
    try:
        sources = [*ROOT.glob('*.py'),ROOT/'PLAN.md',ROOT/'config.json',
                   ROOT.parent/'channel_routing_variance_20261023/routing_kernels.py',
                   ROOT.parent/'channel_routing_variance_20261023/metrics.py',Path(legacy.__file__),Path(legacy.r2.__file__)]
        save(ROOT/'sources.json',{str(p):sha(p) for p in sources})
        save(ROOT/'environment.json',dict(python=sys.version,torch=torch.__version__,sklearn=sklearn.__version__,
                                          pid=os.getpid(), device='CPU cached-prediction routing'))
        with threadpool_limits(limits=2):
            records=select_all()
            rows=score_all(records)
            report(rows)
        save(ROOT/'inputs.json',INPUTS)
        for path,digest in {**INPUTS,**json.loads((ROOT/'sources.json').read_text(encoding='utf-8'))}.items():
            if sha(path)!=digest: raise RuntimeError('Source or input changed: '+path)
        save(ROOT/'complete.json',dict(cells=20,neural_fits=0,input_hashes_verified=len(INPUTS),
                                       completed=datetime.now(timezone.utc).isoformat()))
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        (ROOT/'run.lock').unlink(missing_ok=True)


if __name__ == '__main__': main()
