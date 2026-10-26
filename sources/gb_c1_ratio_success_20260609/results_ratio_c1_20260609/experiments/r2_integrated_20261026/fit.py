"""Serial paired GPU training with per-candidate optimizer/RNG continuation."""
import json
import time

import torch
import torch.nn.functional as F

from protocol import (ROOT, AlignedMLP, state_hash, seed_all, rng_get, rng_set, training_data,
                      output, save, read, sha, now, canonical, put_target, disk_guard, append)

INPUT_HASHES = {}
HIDDEN = {}


def f1_score(pred, y, classes):
    matrix = torch.bincount(y*classes+pred, minlength=classes**2).reshape(classes, classes).double()
    den = matrix.sum(0)+matrix.sum(1)
    return float(torch.where(den>0, 2*matrix.diag()/den.clamp_min(1), 0).mean())


class Student:
    def __init__(self, backbone, values, encoder, seed, hidden_key=None):
        self.data = training_data(values)
        self.kind = backbone
        seed_all(seed)
        self.model = AlignedMLP(values['x'].shape[1], values['classes'],
                                encoder['projection.weight'].shape[0], 64, .5).cuda()
        self.hidden = None
        self.encoder_hash = None
        if backbone == 'R2':
            state = self.model.state_dict()
            for key in state:
                if key.startswith(('backbone.layers.0.', 'backbone.batch_norms.', 'projection.')):
                    state[key] = encoder[key]
            self.model.load_state_dict(state)
            self.model.freeze_encoder()
            self.encoder_hash = state_hash(self.model.encoder_state())
            cache_key = (hidden_key, self.encoder_hash)
            if hidden_key is not None and cache_key in HIDDEN:
                self.hidden = HIDDEN[cache_key]
            else:
                with torch.no_grad():
                    self.hidden = self.model.encode(values['x']).detach()
                if hidden_key is not None:
                    HIDDEN.clear()
                    HIDDEN[cache_key] = self.hidden
            params = self.model.classifier_parameters()
        elif backbone == 'R0':
            params = self.model.backbone.parameters()
        else:
            raise ValueError(backbone)
        self.optimizer = torch.optim.Adam(params, lr=.01, weight_decay=.0005)
        self.initial_hash = state_hash(self.model.state_dict())
        self.initial_rng = state_hash(dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state()))

    def forward(self, ids=None):
        if self.hidden is not None:
            return self.model.backbone.layers[-1](self.hidden if ids is None else self.hidden[ids])
        value = self.model(self.data['x'])[0]
        return value if ids is None else value[ids]

    def state(self):
        state = self.model.backbone.layers[-1].state_dict() if self.kind=='R2' else self.model.state_dict()
        return {k:v.detach().cpu().clone() for k,v in state.items()}

    def set_state(self, state):
        if self.kind=='R2':
            self.model.backbone.layers[-1].load_state_dict(state)
        else:
            self.model.load_state_dict(state)

    def check_frozen(self):
        if self.kind=='R2' and state_hash(self.model.encoder_state()) != self.encoder_hash:
            raise RuntimeError('Frozen encoder/BN changed')


def cleanup_resume(folder):
    folder = output(folder)
    result = read(folder/'result.json')
    if result['epoch'] != 200 or result['status'] != 'complete':
        raise ValueError('Only complete 200epoch fits may drop terminal resume')
    for filename, key in [('checkpoint.pt','checkpoint_sha256'), ('probabilities.pt','probabilities_sha256')]:
        if sha(folder/filename) != result[key]:
            raise RuntimeError('Artifact verification failed before temporary cleanup')
    resume = output(folder/'resume.pt')
    if resume.exists():
        size = resume.stat().st_size
        digest = sha(resume)
        resume.unlink()
        append(ROOT/'storage_cleanup.jsonl', dict(time=now(), path=canonical(resume), bytes=size,
               sha256=digest, reason='complete_200_verified_best_and_probabilities'))


def fit(folder, backbone, values, target, encoder, *, seed, until=200, q=None, metadata=None):
    folder = output(folder)
    folder.mkdir(parents=True, exist_ok=True)
    values = training_data(values)
    metadata = metadata or {}
    target_ref, target_hash = put_target(target)
    cache_key = (metadata.get('dataset'), seed, metadata.get('condition'), backbone)
    if cache_key not in INPUT_HASHES:
        INPUT_HASHES[cache_key] = state_hash({k:v for k,v in values.items() if isinstance(v,torch.Tensor)})
    meta = dict(metadata, backbone=backbone, seed=seed, q=q, target_hash=target_hash, target_ref=target_ref,
                input_hash=INPUT_HASHES[cache_key], source_sha256=sha(ROOT/'sources.json'),
                optimizer='Adam', lr=.01, weight_decay=.0005, selection='first_strict_max_validation_accuracy')
    if (folder/'result.json').exists():
        previous = read(folder/'result.json')
        for key in meta:
            if previous[key] != meta[key]:
                raise RuntimeError('Resume manifest mismatch '+str(folder)+' '+key)
        if previous['epoch'] >= until:
            for filename,key in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
                if sha(folder/filename) != previous[key]:
                    raise RuntimeError('Completed artifact changed')
            if until==200:
                cleanup_resume(folder)
            return previous
    disk_guard(128*1024**2)
    student = Student(backbone, values, encoder, seed, (metadata.get('dataset'),seed))
    meta.update(initial_hash=student.initial_hash, initial_rng_hash=student.initial_rng,
                encoder_hash=student.encoder_hash, checkpoint_format='head_with_encoder_reference' if backbone=='R2' else 'full')
    current, best, best_epoch, best_nll, best_f1 = 0, -1, 0, 1e30, 0.
    best_state, seconds = None, 0.
    if (folder/'resume.pt').exists():
        ck = torch.load(folder/'resume.pt', map_location='cpu', weights_only=False)
        for key in ('target_hash','input_hash','seed','source_sha256','initial_hash'):
            if ck['meta'][key] != meta[key]:
                raise RuntimeError('Prefix resume passport mismatch '+key)
        student.set_state(ck['model'])
        student.optimizer.load_state_dict(ck['optimizer'])
        current,best,best_epoch,best_nll,best_f1 = [ck[k] for k in ('epoch','best','best_epoch','best_nll','best_f1')]
        best_state,seconds = ck['best_state'], ck['seconds']
        logs = [json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
        if len(logs) != current:
            raise RuntimeError('Prefix log/checkpoint epoch mismatch; preserve failed attempt')
        rng_set(ck['rng'])
    elif (folder/'epochs.jsonl').exists():
        raise RuntimeError('Partial noncheckpointed fit: preserve and repair before retry')
    save(folder/'config.json', meta)
    tr,vi,z,yv = [values[k] for k in ('train','valid','observed','valid_y')]
    t = target.cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with (folder/'epochs.jsonl').open('a',encoding='utf-8') as log:
        for epoch in range(current+1,until+1):
            student.model.train_head() if backbone=='R2' else student.model.train()
            student.optimizer.zero_grad(set_to_none=True)
            lp = student.forward(tr).log_softmax(1)
            value = ((1-torch.exp(q*lp.gather(1,z[:,None])))/q).mean() if q is not None else -(t*lp).sum(1).mean()
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite loss')
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.model.parameters()):
                raise FloatingPointError('Nonfinite gradients')
            student.optimizer.step()
            student.model.eval()
            with torch.no_grad():
                lv = student.forward(vi)
                pred = lv.argmax(1)
                correct = int((pred==yv).sum())
                nll = float(F.cross_entropy(lv,yv))
                vf1 = f1_score(pred,yv,values['classes'])
            if correct>best:
                best,best_epoch,best_nll,best_f1 = correct,epoch,nll,vf1
                best_state = student.state()
            log.write(json.dumps(dict(epoch=epoch,loss=float(value.detach()),validation_correct=correct,
                                     validation_nll=nll,validation_macro_f1=vf1))+'\n')
            log.flush()
    student.check_frozen()
    torch.cuda.synchronize()
    seconds += time.perf_counter()-start
    if until<200:
        torch.save(dict(meta=meta,model=student.state(),optimizer=student.optimizer.state_dict(),rng=rng_get(),
                        epoch=until,best=best,best_epoch=best_epoch,best_nll=best_nll,best_f1=best_f1,
                        best_state=best_state,seconds=seconds),folder/'resume.pt')
    torch.save(dict(model=best_state,kind=backbone,epoch=best_epoch,
                    encoder_path=metadata.get('encoder_path'), encoder_file_sha256=metadata.get('encoder_file_sha256')),
               folder/'checkpoint.pt')
    student.set_state(best_state)
    student.model.eval()
    with torch.no_grad():
        probs = student.forward().softmax(1).cpu()
    torch.save(probs,folder/'probabilities.pt')
    result = dict(meta,epoch=until,status='complete' if until==200 else 'prefix',best_epoch=best_epoch,
                  valid_correct=best,valid_total=len(vi),validation=best/len(vi),valid_nll=best_nll,
                  valid_macro_f1=best_f1,seconds=seconds,finished=now(),
                  encoder_frozen_unchanged=backbone=='R2',
                  checkpoint_sha256=sha(folder/'checkpoint.pt'),probabilities_sha256=sha(folder/'probabilities.pt'))
    save(folder/'result.json',result)
    append(folder/'segments.jsonl',dict(start_epoch=current,end_epoch=until,finished=result['finished'],
           result_sha256=sha(folder/'result.json'),seconds=seconds))
    if until==200:
        cleanup_resume(folder)
    print(f'FIT {canonical(folder)} epoch={until} val={result["validation"]:.5f} sec={seconds:.2f}',flush=True)
    return result
