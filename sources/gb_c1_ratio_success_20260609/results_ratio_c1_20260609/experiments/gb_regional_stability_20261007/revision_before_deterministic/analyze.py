from __future__ import annotations

import csv
import json
import statistics

import torch
from torch.nn import functional as F

import run_stability as run

ROOT = run.ROOT


def save_csv(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else None,
                wins=sum(v>1e-10 for v in values), losses=sum(v < -1e-10 for v in values),
                ties=sum(abs(v)<=1e-10 for v in values), worst=min(values), best=max(values))


def postfit_metrics(probabilities, teacher, data, region):
    test = data.test_index
    predicted, teacher_predicted = probabilities[test].argmax(1), teacher[test].argmax(1)
    labels = data.y[test]
    index = data.train_index
    truth = F.one_hot(data.y[index], data.num_classes).float()
    return dict(test_accuracy=100*int((predicted==labels).sum())/test.numel(),
                teacher_wrong_student_right=int(((teacher_predicted!=labels)&(predicted==labels)).sum()),
                teacher_right_student_wrong=int(((teacher_predicted==labels)&(predicted!=labels)).sum()),
                clean_train_brier=float((probabilities[index]-truth).square().sum(1).mean()),
                pointwise_kl=float(F.kl_div(probabilities.clamp_min(1e-30).log(),
                                          teacher, reduction='batchmean')),
                real_region_kl=float(region(probabilities)),
                **{f'residual_{k}':v for k,v in
                   run.residual_components(probabilities, teacher, region.groups).items()})


def analyze():
    run.old.require_gpu()
    run.verify()
    pairs, cells, metrics, fixed, trajectory, audit = [], [], [], [], [], []
    for dataset in run.DATASETS:
        for rate in (.4, .6):
            data, protocol = run.previous.load_data(dataset, rate)
            config = run.read(run.old.cache_folder(dataset, rate)/'config.json')
            assert protocol==config['protocol']
            cache = torch.load(run.old.cache_folder(dataset, rate)/'regional.pt',
                               map_location='cpu', weights_only=True)
            teacher = cache['teacher']
            region = run.RegionalKL(teacher, cache['assignments']['R1'])
            x = data.x.cuda()
            pair_lookup = {}
            for seed in (0,1,2):
                pair = dict(dataset=dataset, noise_rate=rate, student_seed=seed,
                            teacher_accuracy=100*int((teacher[data.test_index].argmax(1)
                                                     ==data.y[data.test_index]).sum())/data.test_index.numel())
                initial_hashes, rng_hashes = set(), set()
                for arm in run.ARMS:
                    folder = run.old.run_folder(dataset, rate, arm) if seed==0 else run.folder_for(dataset,rate,seed,arm)
                    result = run.read(folder/'result.json')
                    assert result['status']=='ok'
                    initial_hashes.add(result['initial_state_hash'])
                    if seed:
                        rng_hashes.add(result['initial_rng_hash'])
                        recorded = run.read(folder/'config.json')
                        assert recorded['protocol']==protocol and recorded['student_seed']==seed
                        logs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
                        assert [v['epoch'] for v in logs]==list(range(1,201))
                        best_val = max(v['valid'] for v in logs)
                        assert next(v['epoch'] for v in logs if v['valid']==best_val)==result['best_epoch']
                        traces = [json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
                        assert [v['epoch'] for v in traces]==[1]+list(range(10,201,10))
                        for record in traces:
                            assert record['residual_decomposition_error']<1e-10
                            assert abs(record['gradient_cosine'])<=1+1e-9
                            trajectory.append(dict(dataset=dataset,noise_rate=rate,student_seed=seed,arm=arm,**record))
                    checkpoints = ['student'] if seed==0 else ['student']+[f'epoch{e}' for e in run.SNAPSHOTS]
                    for checkpoint in checkpoints:
                        probabilities = torch.load(folder/f'{checkpoint}_probabilities.pt',
                                                   map_location='cpu',weights_only=True)
                        assert probabilities.shape==teacher.shape and torch.isfinite(probabilities).all()
                        assert (probabilities.sum(1)-1).abs().max()<1e-5
                        error=0.
                        if seed:
                            model=run.previous.MLP(data.x.shape[1],64,data.num_classes,2,.5).cuda()
                            model.load_state_dict(torch.load(folder/f'{checkpoint}.pt',
                                                            map_location='cuda',weights_only=True))
                            model.eval()
                            with torch.no_grad():
                                restored=model(x).softmax(1).cpu()
                            error=float((restored-probabilities).abs().max())
                            assert error<=1e-6 and torch.equal(restored.argmax(1), probabilities.argmax(1))
                            audit.append(dict(dataset=dataset,noise_rate=rate,student_seed=seed,
                                              arm=arm,checkpoint=checkpoint,max_abs_difference=error))
                            del model, restored
                        values=postfit_metrics(probabilities,teacher,data,region)
                        assert values['residual_decomposition_error']<1e-10
                        if checkpoint=='student':
                            exact=run.old.exact_metrics(probabilities,data)
                            for key,value in exact.items():
                                assert abs(result[key]-value)<1e-12,(dataset,rate,seed,arm,key)
                            pair[arm]=values['test_accuracy']
                            pair[f'{arm}_best_epoch']=result['best_epoch']
                            pair[f'{arm}_valid_correct']=result['valid_correct']
                            pair[f'{arm}_test_correct']=result['test_correct']
                            metrics.append(dict(dataset=dataset,noise_rate=rate,student_seed=seed,
                                                arm=arm,best_epoch=result['best_epoch'],**values))
                        else:
                            epoch=int(checkpoint.removeprefix('epoch'))
                            valid=int((probabilities[data.val_index].argmax(1)==data.y[data.val_index]).sum())
                            assert valid==next(v['validation_correct'] for v in traces if v['epoch']==epoch)
                            fixed.append(dict(dataset=dataset,noise_rate=rate,student_seed=seed,arm=arm,
                                              epoch=epoch,validation_correct=valid,**values))
                assert len(initial_hashes)==1
                if seed:
                    assert len(rng_hashes)==1
                pair['delta']=pair['R1']-pair['R0']
                pairs.append(pair)
                pair_lookup[seed]=pair
            new=[pair_lookup[s]['delta'] for s in (1,2)]
            all_seeds=[pair_lookup[s]['delta'] for s in (0,1,2)]
            cells.append(dict(dataset=dataset,noise_rate=rate,
                              seed0_delta=all_seeds[0],seed1_delta=new[0],seed2_delta=new[1],
                              new_delta_mean=statistics.mean(new),new_delta_std=statistics.stdev(new),
                              R0_mean=statistics.mean(pair_lookup[s]['R0'] for s in (0,1,2)),
                              R0_std=statistics.stdev(pair_lookup[s]['R0'] for s in (0,1,2)),
                              R1_mean=statistics.mean(pair_lookup[s]['R1'] for s in (0,1,2)),
                              R1_std=statistics.stdev(pair_lookup[s]['R1'] for s in (0,1,2)),
                              delta_mean=statistics.mean(all_seeds),delta_std=statistics.stdev(all_seeds)))
            del x
            torch.cuda.empty_cache()
    assert len(audit)==200 and len(pairs)==30 and len(cells)==10 and len(trajectory)==840
    per_seed={str(seed):stats([v['delta'] for v in pairs if v['student_seed']==seed]) for seed in (0,1,2)}
    new_pairs=[v for v in pairs if v['student_seed'] in (1,2)]
    positive_cells=sum(v['new_delta_mean']>1e-10 for v in cells)
    gates=dict(seed1_mean_positive=per_seed['1']['mean']>0,
               seed2_mean_positive=per_seed['2']['mean']>0,
               at_least_six_positive_cells=positive_cells>=6)
    lookup={(v['dataset'],v['noise_rate'],v['student_seed'],v['arm']):v for v in metrics}
    mechanism={}
    for metric in ['pointwise_kl','real_region_kl','residual_between','residual_within','clean_train_brier']:
        changes=[lookup[(v['dataset'],v['noise_rate'],v['student_seed'],'R1')][metric]-
                 lookup[(v['dataset'],v['noise_rate'],v['student_seed'],'R0')][metric] for v in new_pairs]
        mechanism[metric]=dict(mean_change=statistics.mean(changes),lower=sum(v<0 for v in changes))
    fixed_lookup={(v['dataset'],v['noise_rate'],v['student_seed'],v['arm'],v['epoch']):v for v in fixed}
    fixed_summary={}
    for epoch in run.SNAPSHOTS:
        differences=[fixed_lookup[(v['dataset'],v['noise_rate'],v['student_seed'],'R1',epoch)]['test_accuracy']-
                     fixed_lookup[(v['dataset'],v['noise_rate'],v['student_seed'],'R0',epoch)]['test_accuracy']
                     for v in new_pairs]
        fixed_summary[str(epoch)]=stats(differences)
    summary=dict(status='complete',new_training_runs=40,engineering_replays=2,
                 teacher_split_noise_seed=0,student_seeds=[1,2],per_seed=per_seed,
                 new_pairs=stats([v['delta'] for v in new_pairs]),positive_cells=positive_cells,
                 all_three_seed_mean=statistics.mean(v['delta'] for v in pairs),
                 gates=gates,passed=all(gates.values()),mechanism=mechanism,
                 fixed_epoch_diagnostics=fixed_summary,
                 scope='student optimization randomness only; shared split, noise, teacher and partition',
                 mean_std_note='per-cell mean/std across student seeds; seed 0 is development evidence',
                 fixed_epoch_note='post-fit diagnostics; not used to change the validation selection rule')
    for name,rows in [('paired_results.csv',pairs),('cell_summary.csv',cells),
                      ('selected_diagnostics.csv',metrics),('fixed_epoch_results.csv',fixed),
                      ('trajectory.csv',trajectory)]:
        save_csv(name,rows)
    run.write(ROOT/'summary.json',summary)
    run.write(ROOT/'output_audit.json',dict(status='passed',new_runs=40,restored_checkpoints=200,
              checks=audit,paired_initialization_and_rng=True,selection_checks=40,input_hashes_unchanged=True))
    run.verify()
    report(summary,cells)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


def report(summary,cells):
    lines=['# 固定教师区域蒸馏：学生 seed 1/2 重复结果','',
           '日期：2026-10-07。40 次新训练全部完成；另有两次 seed 0 工程回放。','',
           '## 协议与研究问题','',
           '检验当前 lambda=1 的真实粒球区域损失能否在改变学生初始化及 dropout 后保留增益。',
           '复用原 ERASE 教师、五数据集、uniform 0.4/0.6、划分、噪声和 GRACE 分区。',
           'R0 为原软交叉熵，R1 加球大小加权区域 KL；两层属性 MLP、hidden 64、BN、dropout 0.5，',
           'Adam lr 0.01、weight decay 0.0005、200 轮，全节点蒸馏，按干净验证精度选首个最优 epoch。',
           '新增 seed 1、2 只控制学生初始化和 dropout。这不是三个完整数据/教师种子的独立验证。','',
           '## 每个种子的配对差值','',
           '| 学生 seed | 十格平均差值 | 胜/平/负 | 最差 |','|---|---:|---:|---:|']
    for seed,values in summary['per_seed'].items():
        lines.append(f'| {seed} | {values["mean"]:+.4f} | {values["wins"]}/{values["ties"]}/{values["losses"]} | {values["worst"]:+.4f} |')
    lines+=['','差值单位为百分点。seed 0 为上一轮开发结果，新增两种子为本轮主分析。','',
            '| 数据集 | 噪声 | seed 0 差值 | seed 1 差值 | seed 2 差值 | 新两种子均值 | 三种子差值 mean ± std |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for v in cells:
        lines.append(f'| {v["dataset"]} | {v["noise_rate"]} | {v["seed0_delta"]:+.3f} | {v["seed1_delta"]:+.3f} | {v["seed2_delta"]:+.3f} | {v["new_delta_mean"]:+.3f} | {v["delta_mean"]:+.3f} ± {v["delta_std"]:.3f} |')
    lines+=['','std 使用样本标准差 ddof=1；每格只有三个学生种子，不是完整协议的不确定性。','',
            '## 预设门槛与决策','']
    for key,passed in summary['gates'].items():
        lines.append(f'- {key}: {"通过" if passed else "未通过"}')
    lines+=['',f'新增二十个配对差值的等权均值为 {summary["new_pairs"]["mean"]:+.4f} 点；十格中 {summary["positive_cells"]} 格的新两种子均值为正。','']
    if summary['passed']:
        lines+=['**决策：通过本轮优化稳定性筛查，可继续机制验证。**',
                '下一阶段应检验同 GRACE 空间的 k-means 及更强蒸馏对照。本轮没有自动启动后续实验。']
    else:
        lines+=['**决策：未通过本轮优化稳定性筛查，暂缓扩大当前候选。**',
                '应先定位退化与训练路径波动；本轮没有按测试结果调参或追加损失。']
    lines+=['','上一轮 +0.5 点幅度门槛未通过的记录不变，本轮考察的条件已在运行前单独写入 PLAN.md。','',
            '## 机制与训练过程','',
            '| 最佳检查点 R1-R0 指标 | 新二十对平均变化 | 下降次数 |','|---|---:|---:|']
    for key,v in summary['mechanism'].items():
        lines.append(f'| {key} | {v["mean_change"]:+.6f} | {v["lower"]}/20 |')
    lines+=['','| 固定轮次（诊断） | 平均精度差 | 胜/平/负 |','|---|---:|---:|']
    for epoch,v in summary['fixed_epoch_diagnostics'].items():
        lines.append(f'| {epoch} | {v["mean"]:+.4f} | {v["wins"]}/{v["ties"]}/{v["losses"]} |')
    lines+=['','固定轮次结果是在训练结束后计算的诊断，不用于挑选替代最佳轮次。',
            '梯度统计来自更新前 train 前向；概率统计来自更新后 eval 前向。二者的状态差异已在轨迹记录中标注。',
            '区域偏差缩小只说明该方向更接近教师，不能单独推导分类准确率或因果机制。','',
            '## 核验与文件','',
            '40 组初始化与初始随机状态配对一致；200 轮日志和首个验证最佳 epoch 已核验；',
            '40 个最佳和 160 个固定轮次检查点全部在 GPU 恢复并核对预测。原输入 SHA-256 未变。',
            '观察函数的随机数/参数更新不变性单测通过，Cora 0.4 R0/R1 seed 0 回放通过。',
            '数值和记录见 paired_results.csv、cell_summary.csv、selected_diagnostics.csv、trajectory.csv、',
            'fixed_epoch_results.csv、summary.json 和 output_audit.json。runs/ 保存配置、原始日志和检查点。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    analyze()
