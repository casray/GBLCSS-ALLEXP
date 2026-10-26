"""Independent table/decision check from saved probabilities and plain CSV."""
import csv
from pathlib import Path
from statistics import mean
import torch
from sklearn.metrics import f1_score
from run import ROOT, CONFIG, raw_data, read, write, sha, verify


def table(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    verify()
    rows = table('results.csv')
    scores = {}
    for row in rows:
        key = tuple(row[k] for k in ('dataset','condition','route','arm'))
        assert key not in scores
        geo = torch.load(ROOT/'inputs'/row['dataset']/'geometry.pt', weights_only=True)
        y = raw_data(row['dataset']).y[geo['test']].numpy()
        probs = torch.load(Path(row['source'])/'probabilities.pt', weights_only=True)[geo['test']]
        pred = probs.argmax(1).numpy()
        value = 100*sum(int(a==b) for a,b in zip(y,pred))/len(y)
        f1 = 100*f1_score(y,pred,average='macro',labels=list(range(probs.shape[1])),zero_division=0)
        assert abs(value-float(row['accuracy']))<1e-10
        assert abs(f1-float(row['macro_f1']))<1e-10
        scores[key] = value
    expected = {(d,c,r,a) for d in CONFIG['datasets'] for c in CONFIG['conditions'] for r in CONFIG['routes'] for a in CONFIG['arms']}
    assert set(scores)==expected and len(rows)==192
    for row in table('summary.csv'):
        values = [v for (d,c,r,a),v in scores.items() if r==row['route'] and a==row['arm'] and (c=='clean')==(row['domain']=='clean')]
        assert abs(mean(values)-float(row['accuracy']))<1e-10 and row['seed_std']==''
    for row in table('paired_differences.csv'):
        prefix = tuple(row[k] for k in ('dataset','condition','route'))
        assert abs(scores[prefix+('NP',)]-scores[prefix+(row['reference'],)]-float(row['np_delta_pp']))<1e-10
    gates = read(ROOT/'gates.json')
    for route in CONFIG['routes']:
        clean = [scores[d,'clean',route,'NP']-scores[d,'clean',route,'CE'] for d in CONFIG['datasets']]
        noisy = [scores[d,c,route,'NP']-scores[d,c,route,'OLD64'] for d in CONFIG['datasets'] for c in CONFIG['conditions'] if c!='clean']
        safe = min(clean)>=-1-1e-10
        repair = safe and scores['corafull','clean',route,'NP']-scores['corafull','clean',route,'OLD64']>=1-1e-10 and mean(noisy)>=-1e-10 and sum(v>=-1e-10 for v in noisy)>=6 and min(noisy)>=-1-1e-10
        incremental = safe
        for reference in ['TREE64','FINE','KM']:
            delta = [scores[d,c,route,'NP']-scores[d,c,route,reference] for d in CONFIG['datasets'] for c in CONFIG['conditions'] if c!='clean']
            incremental = incremental and mean(delta)>=.5-1e-10 and sum(v>1e-10 for v in delta)>=6
        assert repair==gates['routes'][route]['repair_pass']
        assert incremental==gates['routes'][route]['incremental_pass']
    for name in ['cora','citeseer']:
        for condition in CONFIG['conditions']:
            for route in CONFIG['routes']:
                base = ROOT/'runs'/name/condition
                assert torch.equal(torch.load(base/(route+'_TREE64')/'probabilities.pt',weights_only=True),
                                   torch.load(base/(route+'_FINE')/'probabilities.pt',weights_only=True))
    write(ROOT/'table_audit.json',dict(passed=True,cells=192,summary_rows=32,paired_rows=168,
          identical_tree64_fine_pairs=16,gates_checked=4,source_sha256=sha(__file__)))
    lines = ['# Full seed0 result tables','', 'Accuracy (%); no independent-seed standard deviation.','']
    for route in CONFIG['routes']:
        lines += ['## '+route,'','| Dataset / condition | '+' | '.join(CONFIG['arms'])+' |',
                  '|---|'+'---:|'*len(CONFIG['arms'])]
        for name in CONFIG['datasets']:
            for condition in CONFIG['conditions']:
                lines.append('| '+name+' / '+condition+' | '+' | '.join(f'{scores[name,condition,route,arm]:.3f}' for arm in CONFIG['arms'])+' |')
        lines.append('')
    (ROOT/'RESULT_TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    print('INDEPENDENT CHECK PASSED: 192 cells, 32 summaries, 168 paired differences, four gates, 16 identical-arm pairs',flush=True)


if __name__=='__main__':
    main()
