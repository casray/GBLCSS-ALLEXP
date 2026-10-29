"""Exact midpoint replay plus provenance-labelled reuse of a full-budget baseline."""
import gc
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run as original
from runtime import *
from schedule import convert_checkpoint


def expected_baseline(d, s):
    return dict(dataset=d, seed=s, arm='ecl', config=dataset_config(d), receipt_sha256=sha(ROOT/'receipt.json'), stage='formal')


def ensure_midpoint(d, s, testing=False):
    prefix = ROOT/'budget_revision/smoke' if testing else ROOT/'formal'
    shared = prefix/'warmup'/d/f'seed{s}'
    work = ROOT/'budget_revision'/('smoke_replay' if testing else 'midpoint_replay')/d/f'seed{s}'
    expected = expected_baseline(d, s)
    cfg = expected['config']; mid = cfg['epochs']//2
    if (shared/'budget_ready.json').exists():
        ready = read(shared/'budget_ready.json')
        assert ready['expected'] == expected
        for name, value in ready['files'].items():
            assert sha(shared/name) == value
        return shared
    data = features(d, s); x = data['x'].cuda()
    edge = ge.with_loops(data['edge_index'].cuda(), len(x)); adj = ge.adjacency(edge, len(x), x.dtype)
    seed_all(s); model = ge.Model(x.shape[1], cfg).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']); seed_all(s)
    resume = shared/'checkpoint.pt' if (shared/'checkpoint.pt').exists() else work/'resume.pt'
    first = 1
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        assert ck['expected'] == expected and ck['epoch'] <= mid
        restore_snapshot(model, optimizer, ck); first = ck['epoch'] + 1
    work.mkdir(parents=True, exist_ok=True)
    logpath = work/'epochs.jsonl'
    if logpath.exists():
        prior = logpath.read_text(encoding='utf-8'); rows = [json.loads(v) for v in prior.splitlines()]
        if any(r['epoch'] >= first for r in rows):
            (work/f'epochs_prior_resume_{time.time_ns()}.jsonl').write_text(prior, encoding='utf-8')
            logpath.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch'] < first), encoding='utf-8')
    started = time.perf_counter()
    with logpath.open('a', encoding='utf-8') as log:
        for epoch in range(first, mid+1):
            model.train(); optimizer.zero_grad(set_to_none=True); g, t = model(x, adj)
            loss = ge.backward_loss(g, t, edge, chunk=CFG['chunk'], **loss_kwargs(cfg))
            if not math.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite midpoint update')
            optimizer.step()
            log.write(json.dumps(dict(epoch=epoch, loss=loss, seconds_current_attempt=time.perf_counter()-started))+'\n'); log.flush()
            if epoch % 25 == 0 or epoch == mid:
                save_tensor(work/'resume.pt', dict(expected=expected, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state()))
                original.status('BUDGET_MIDPOINT', dataset=d, seed=s, epoch=epoch, total=mid)
    save_tensor(shared/'checkpoint.pt', dict(expected=expected, epoch=mid, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state()))
    h = original.export_h(model, x, shared/'H.pt')
    cache = original.build_context(h, data['edge_index'], s, shared/'context', 'midpoint_replay')
    original.grad_diagnostics(model, x, adj, edge, cache, cfg, shared/'gradient_diagnostics.json')
    save(shared/'budget_ready.json', dict(expected=expected, source='exact prefix of original training loop', epochs=mid,
        seconds_current_attempt=time.perf_counter()-started,
        files={name: sha(shared/name) for name in ['checkpoint.pt','H.pt','context/context.pt','context/stats.json','gradient_diagnostics.json']}))
    del model, optimizer, x, edge, adj; gc.collect(); torch.cuda.empty_cache()
    return shared


def import_baseline(d, s, testing=False):
    expected = expected_baseline(d, s); cfg = expected['config']; T = cfg['epochs']
    folder = ROOT/'budget_revision/smoke/imported'/d/f'seed{s}' if testing else ssl_dir(d, s, 'ecl')
    if art.reusable(folder, expected):
        return folder
    old = old_ssl(d, s); manifest = read(old/'complete.json')
    for name, value in manifest['files'].items():
        assert sha(old/name) == value
    old_result = read(old/'result.json')
    for key in ['dataset','seed','config']:
        assert old_result[key] == expected[key], key
    ck = convert_checkpoint(torch.load(old/'resume.pt', map_location='cpu', weights_only=False), expected)
    old_model = load(old/f'model_epoch{T}.pt'); h = load(old/f'H_epoch{T}.pt')['H'].float()
    assert state_hash(ck['model']) == state_hash(old_model)
    data = features(d, s); model = ge.Model(data['x'].shape[1], cfg).cuda(); model.load_state_dict(old_model); model.eval()
    with torch.no_grad():
        replay = torch.nn.functional.normalize(model.embedding(data['x'].cuda()).cpu(), dim=1)
    error = float((replay-h).abs().max()); assert error <= 1e-7
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/'config.json', dict(expected, initial_hash=old_result['initial_hash'], baseline_origin='frozen_full_budget_reference'))
    art.atomic(folder/'epochs.jsonl', lambda p: Path(p).write_bytes((old/'epochs.jsonl').read_bytes()))
    save_tensor(folder/'H.pt', h); save_tensor(folder/'model.pt', old_model); save_tensor(folder/'resume.pt', ck)
    audit = dict(source='frozen_full_budget_reference', legacy_source=str(old), legacy_manifest_sha256=sha(old/'complete.json'),
        initial_hash=old_result['initial_hash'], old_state_equal=True, old_h_max_error=0., reload_error=error,
        seconds_current_attempt=0., cached_training_seconds=old_result['seconds_current_attempt'],
        original_total_epochs=T, new_baseline_updates=0, shared_midpoint_updates=T//2)
    save(folder/'result.json', dict(expected, status='complete', audit=audit))
    art.complete(folder, expected, ['config.json','epochs.jsonl','H.pt','model.pt','resume.pt','result.json'])
    del model, data, ck; gc.collect(); torch.cuda.empty_cache()
    return folder
