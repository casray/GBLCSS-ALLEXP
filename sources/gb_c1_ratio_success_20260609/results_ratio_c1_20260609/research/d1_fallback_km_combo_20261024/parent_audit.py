"""Independent selector/statistic/CPU prediction audit; old outputs are read-only."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
previous = HERE.parent / 'd1_c1_confirmation_20261024/parent_audit.py'
spec = importlib.util.spec_from_file_location('prior_readonly_audit_helpers', previous)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.EXP = HERE.parents[1] / 'experiments/d1_fallback_km_combo_20261024'
ROOT = helper.EXP
read, rows, load, sha = helper.read, helper.rows, helper.load, helper.sha
torch, F, np = helper.torch, helper.F, helper.np


def main():
    complete = read(ROOT / 'complete.json')
    assert complete['audit'] == 'passed' and not (ROOT / 'run.lock').exists()
    for p, digest in read(ROOT / 'sources.json').items():
        assert sha(p) == digest
    freeze = read(ROOT / 'selections_frozen.json')
    assert sha(ROOT / 'reuse_manifest.json') == freeze['reuse_manifest_sha256']
    refs = read(ROOT / 'reuse_manifest.json')
    selector_checks = 0
    for path, digest in freeze['choices'].items():
        assert sha(ROOT / path) == digest
        cell = read(ROOT / path)
        for selector, methods in cell['choices'].items():
            for name, selected in methods.items():
                def rank(r):
                    primary = r['validation'] if selector == 'accuracy' else (r['validation'] + r['valid_macro_f1']) / 2
                    return (-primary, -r['validation'], r['valid_nll'], r['best_epoch'])
                assert min(cell['pools'][name], key=rank)['ref'] == selected
                selector_checks += 1
        if cell['backbone'] == 'R0':
            for method in ('LEGACY_EXPANDED', 'KM_EXPANDED', 'FIXED_EXPANDED'):
                assert len(cell['pools'][method]) == 8
            support = cell['meta']['adaptive_support']
            if support is None:
                assert cell['pools']['ADAPTIVE_EXPANDED'] == cell['pools']['CE']
            else:
                assert all(refs[r['ref']]['partition'] == f'S{support}' for r in cell['pools']['ADAPTIVE_EXPANDED'])
    assert read(ROOT / 'testing_started.json')['started'] >= freeze['frozen']
    selected, paired = rows('selected_results.csv'), rows('paired_differences.csv')
    keys = ('dataset', 'seed', 'backbone', 'condition', 'selector', 'method')
    lookup = {tuple(r[k] for k in keys): r for r in selected}
    assert len(lookup) == len(selected)
    metrics = ['test_accuracy', 'test_macro_f1', 'validation', 'valid_macro_f1']
    for r in paired:
        key = tuple(r[k] for k in keys[:-1])
        a, b = lookup[key + (r['method'],)], lookup[key + (r['reference'],)]
        for m in metrics:
            assert abs(100*(float(a[m])-float(b[m]))-float(r[m+'_diff_pp'])) < 1e-10
    checks = 0
    for prefix, seeds in [('', (0,1,2)), ('seeds12_', (1,2))]:
        checks += helper.check_summaries(selected, prefix+'summary.csv',
            ['dataset','backbone','condition','selector','method'], metrics, seeds)
        checks += helper.check_summaries(paired, prefix+'paired_summary.csv',
            ['stage','dataset','backbone','condition','selector','method','reference'],
            [m+'_diff_pp' for m in metrics], seeds)
    project = HERE.parents[1].parents[1]
    files = {
        'cora': project/'external/NoisyGL/data/Cora/processed/data.pt',
        'citeseer': project/'external/NoisyGL/data/CiteSeer/processed/data.pt',
        'corafull': project/'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
        'coauthorcs': project/'external/NoisyGL/data/CS/processed/data.pt',
        'amazonpho': project/'external/NoisyGL/data/Photo/processed/data.pt',
    }
    physical = read(ROOT/'physical_fits.json')
    assert len(physical) == complete['scientific_fits'] <= 840
    result_lookup = {r['folder']: r for r in rows('results.csv')}
    reloads = []
    for name, path in files.items():
        raw = torch.load(path, map_location='cpu', weights_only=False)[0]
        x,y = (raw['x'].float(),raw['y'].long()) if isinstance(raw,dict) else (raw.x.float(),raw.y.long())
        for seed in (0,1,2):
            geo = load(helper.HISTORY/name/f'seed{seed}/geometry.pt')
            te = geo['test']
            for kind in ('KM','DEEP'):
                candidates = [r for r in physical.values() if r['dataset']==name and r['seed']==seed and r['condition']=='uniform0.6'
                    and (r['partition']=='KM' if kind=='KM' else r['partition'].startswith('S'))]
                assert candidates, (name,seed,kind)
                folder = ROOT/candidates[0]['folder']
                r = result_lookup[str(folder.resolve())]
                p,ck = load(folder/'probabilities.pt'),load(folder/'checkpoint.pt')
                pred = p[te].argmax(1).numpy()
                acc = helper.accuracy_score(y[te].numpy(),pred)
                f1 = helper.f1_score(y[te].numpy(),pred,labels=np.arange(p.shape[1]),average='macro',zero_division=0)
                assert abs(acc-float(r['test_accuracy'])) < 1e-12
                assert abs(f1-float(r['test_macro_f1'])) < 1e-12
                state = ck['model']
                hidden = F.relu(F.linear(x,state['backbone.layers.0.weight'],state['backbone.layers.0.bias']))
                bn='backbone.batch_norms.0.'
                hidden = F.batch_norm(hidden,state[bn+'running_mean'],state[bn+'running_var'],state[bn+'weight'],state[bn+'bias'],training=False,eps=1e-5)
                prob = F.linear(hidden,state['backbone.layers.1.weight'],state['backbone.layers.1.bias']).softmax(1)
                error = float((prob-p).abs().max())
                assert error < 1e-5 and torch.equal(prob[te].argmax(1),p[te].argmax(1))
                reloads.append(dict(folder=str(folder),cpu_max_difference=error))
    result = dict(status='passed',physical_fits=len(physical),selector_checks=selector_checks,
        paired_rows=len(paired),summary_rows=checks,metric_and_reload_samples=reloads,
        cpu_max_difference=max(r['cpu_max_difference'] for r in reloads),source_sha256=sha(Path(__file__)),helper_sha256=sha(previous))
    (HERE/'PARENT_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='metric_and_reload_samples'},indent=2))


if __name__ == '__main__':
    main()
