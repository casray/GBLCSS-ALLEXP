import json
import shutil
from pathlib import Path
from replay_gate import HERE

paper = HERE.parents[1]/'GB-LCSS_Elsevier_LaTeX_final_integrated_v2_20260921'
summary = json.loads((HERE/'matrix/summary.json').read_text())
records = json.loads((HERE/'matrix/final_per_seed.json').read_text())
index = {(r['dataset'],r['arm']):r for r in summary}
datasets = [('cora','Cora'),('citeseer','CiteSeer'),('corafull','CoraFull'),('coauthorcs','Coauthor-CS'),('amazonpho','Amazon-Photo')]
backup = HERE/'tables_before_replay'
backup.mkdir(exist_ok=True)


def fmt(dataset, arm):
    r=index[dataset,arm]
    return f"{r['mean']:.2f}\\ensuremath{{\\pm}}{r['std']:.2f}"


def count(dataset, arm):
    values=[int(r['num_balls']) for r in records if r['dataset']==dataset and r['arm']==arm]
    return str(min(values)) if min(values)==max(values) else f'{min(values)}--{max(values)}'


def replace_rows(name, mapping):
    path=paper/'tables'/name
    if not (backup/name).exists():
        shutil.copy2(path,backup/name)
    lines=path.read_text(encoding='utf-8').splitlines()
    seen=set()
    for i,line in enumerate(lines):
        label=line.split(' & ',1)[0]
        if label in mapping:
            suffix=r' \\*' if line.endswith(r'\\*') else r' \\'
            lines[i]=label+' & '+' & '.join(mapping[label])+suffix
            seen.add(label)
    assert seen==set(mapping),(name,seen,set(mapping))
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


replace_rows('table_6.tex',{label:[r'\textbf{'+fmt(d,'default')+'}',fmt(d,'raw_X'),fmt(d,'A2X'),count(d,'default'),count(d,'raw_X'),count(d,'A2X')] for d,label in datasets})
arms=[('Default','default'),(r'\(\beta=0\)','beta_0'),(r'\(\beta=0.25\)','beta_0.25'),(r'\(\beta=0.75\)','beta_0.75'),(r'\(\beta=1\)','beta_1'),(r'\(B_{\max}=32\)','Bmax_32'),(r'\(B_{\max}=128\)','Bmax_128'),(r'\(s_{\min}=5\)','smin_5'),(r'\(s_{\min}=10\)','smin_10'),(r'\(n_{\min}=1\)','nmin_1'),(r'\(n_{\min}=16\)','nmin_16'),(r'\(n_{\min}=64\)','nmin_64'),(r'\(\delta=0.01\)','delta_0.01'),(r'\(\delta=0.10\)','delta_0.1'),('MLP+CE','mlp_ce'),('Random grouping','shuffled'),(r'Raw \(X\)','raw_X'),(r'\(A^2X\)','A2X')]
mapping={}
for label,arm in arms:
    cells=[]
    for d,_ in datasets:
        value=fmt(d,arm)
        highest=max(index[d,a]['mean'] for _,a in arms)
        if abs(index[d,arm]['mean']-highest)<1e-9:
            value=r'\textbf{'+value+'}'
        cells.append(value)
    mapping[label]=cells
replace_rows('table_I_1.tex',mapping)
replace_rows('table_I_2.tex',{label:[f"{index[d,'default'][key]:.4f}" for key in ['observed_brier','target_brier','corrected_error_gain','clean_error_cost','net_brier_gain']] for d,label in datasets})
print('Updated three tables from 450 verified result records.')
