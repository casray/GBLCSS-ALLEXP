"""Paired students, serial GPU fits, resumable optimizer and per-arm RNG."""
from copy import deepcopy
import json
import time

import torch
import torch.nn.functional as F

from protocol import (ROOT, EXP, AlignedMLP, state_hash, seed_all, load, track, save, read,
                      output, rng_get, rng_set, import_file, training_data, now, encoder_path, sha)
from meta_gate import static_features, gate_features, gate_target, meta_step
from targets import mass_direction, global_mass, match_mass

INPUT_HASHES = {}


def f1_score(pred, y, classes):
    matrix = torch.bincount(y*classes+pred, minlength=classes**2).reshape(classes, classes).double()
    denominator = matrix.sum(0)+matrix.sum(1)
    return float(torch.where(denominator > 0, 2*matrix.diag()/denominator.clamp_min(1), 0).mean())


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class Student:
    def __init__(self, backbone, values, encoder=None, sim=None):
        self.data = training_data(values)
        self.kind = backbone
        x = self.data['x']
        classes = self.data['classes']
        seed_all(0)
        self.hidden = None
        self.encoder_hash = None
        if backbone == 'SIM':
            obj, emb, source = sim
            self.model = obj['LogisticRegression'](emb.shape[1], classes).cuda()
            self.hidden = emb
            self.encoder_hash = source
            self.head = self.model.fc
            params = self.model.parameters()
            self.optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=.0005)
        else:
            dim = encoder['projection.weight'].shape[0]
            self.model = AlignedMLP(x.shape[1], classes, dim, 64, .5).cuda()
            if backbone == 'R2':
                state = self.model.state_dict()
                for key in state:
                    if key.startswith(('backbone.layers.0.', 'backbone.batch_norms.', 'projection.')):
                        state[key] = encoder[key]
                self.model.load_state_dict(state)
                self.model.freeze_encoder()
                self.encoder_hash = state_hash(self.model.encoder_state())
                with torch.no_grad():
                    self.hidden = self.model.encode(x).detach()
                params = self.model.classifier_parameters()
            else:
                params = self.model.backbone.parameters()
            self.head = self.model.backbone.layers[-1]
            self.optimizer = torch.optim.Adam(params, lr=.01, weight_decay=.0005)
        self.initial_hash = state_hash(self.model.state_dict())
        self.initial_rng = state_hash(dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state()))

    def forward(self, ids=None, train=False):
        if self.hidden is not None:
            return self.model(self.hidden if ids is None else self.hidden[ids]) if self.kind == 'SIM' else self.head(self.hidden if ids is None else self.hidden[ids])
        value = self.model(self.data['x'])[0]
        return value if ids is None else value[ids]

    def train(self):
        if self.kind == 'R2':
            self.model.train_head()
        else:
            self.model.train()

    def check_frozen(self):
        if self.kind == 'R2' and state_hash(self.model.encoder_state()) != self.encoder_hash:
            raise RuntimeError('Frozen encoder/BN changed')


def sim_embedding(name, x):
    sim = import_file('lcss_sim_readonly', EXP/'simmlp_tined_audit_seed0_20261009/simmlp_adapter.py')
    path = track(EXP/'simmlp_tined_audit_seed0_20261009/runs'/name/'simmlp/checkpoint.pt')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    from torch_geometric.data import Data
    model, obj = sim.build_model(Data(x=x), ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()
    before = state_hash(model.state_dict())
    with torch.no_grad():
        emb = model.encode(x).detach()
    assert before == state_hash(model.state_dict())
    del model
    return obj, emb, sha(path)


def loss_value(logits, target, observed, q=None):
    lp = logits.log_softmax(1)
    if q is not None:
        return ((1-torch.exp(q*lp.gather(1, observed[:, None])))/q).mean()
    return -(target*lp).sum(1).mean()


def fit(folder, backbone, values, target, encoder, *, until=200, sim=None, q=None,
        mode='STANDARD', evidence=None, replay=None, metadata=None):
    folder = output(folder)
    folder.mkdir(parents=True, exist_ok=True)
    values = training_data(values)
    digest = state_hash({'target': target})
    cache_key = (backbone, (metadata or {}).get('dataset'), (metadata or {}).get('condition'), 'meta' in values,
                 tuple((k, tuple(v.shape)) for k, v in values.items() if isinstance(v, torch.Tensor)))
    if metadata is None or cache_key not in INPUT_HASHES:
        value_hash = state_hash({k: v for k, v in values.items() if isinstance(v, torch.Tensor)})
        if metadata is not None:
            INPUT_HASHES[cache_key] = value_hash
    else:
        value_hash = INPUT_HASHES[cache_key]
    meta = dict(backbone=backbone, target_hash=digest, q=q, mode=mode, seed=0,
                optimizer='AdamW' if backbone == 'SIM' else 'Adam', lr=.01, weight_decay=.0005,
                selection='first_strict_max_validation_accuracy', source_sha256=sha(ROOT/'sources.json'),
                input_hash=value_hash,
                **(metadata or {}))
    if (folder/'result.json').exists():
        previous = read(folder/'result.json')
        for k in ('backbone', 'target_hash', 'q', 'mode', 'input_hash', 'source_sha256'):
            if previous[k] != meta[k]:
                raise RuntimeError('Resume/alias input mismatch '+str(folder)+' '+k)
        if previous['epoch'] >= until:
            return previous
    student = Student(backbone, values, encoder, sim)
    meta.update(initial_hash=student.initial_hash, initial_rng_hash=student.initial_rng,
                encoder_hash=student.encoder_hash)
    gate = None
    gate_optimizer = None
    weights_history = []
    if mode.startswith('GATE_'):
        gate = torch.nn.Linear(9, 1).cuda()
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.zeros_(gate.bias)
        gate_optimizer = torch.optim.Adam(gate.parameters(), lr=.001)
        static = static_features(evidence).float().cuda()
    current, best, best_epoch, best_nll, best_f1 = 0, -1, 0, 1e30, 0.
    best_state, best_gate = None, None
    seconds, virtual_seconds, grad_norms = 0., 0., []
    if (folder/'resume.pt').exists():
        ck = torch.load(folder/'resume.pt', map_location='cpu', weights_only=False)
        student.model.load_state_dict(ck['model'])
        student.optimizer.load_state_dict(ck['optimizer'])
        current, best, best_epoch = ck['epoch'], ck['best'], ck['best_epoch']
        best_nll, best_f1 = ck['best_nll'], ck['best_f1']
        best_state, best_gate = ck['best_state'], ck['best_gate']
        seconds, virtual_seconds = ck['seconds'], ck['virtual_seconds']
        weights_history, grad_norms = ck['weights_history'], ck['grad_norms']
        if gate is not None:
            gate.load_state_dict(ck['gate'])
            gate_optimizer.load_state_dict(ck['gate_optimizer'])
        rng_set(ck['rng'])
    elif (folder/'epochs.jsonl').exists():
        raise RuntimeError('Partial noncheckpointed fit requires explicit repair receipt')
    tr, vi, z, yv = [values[k] for k in ('train', 'valid', 'observed', 'valid_y')]
    t = target.cuda()
    torch.save(target.cpu(), folder/'target.pt')
    save(folder/'config.json', meta)
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(current+1, until+1):
            student.train()
            student.optimizer.zero_grad(set_to_none=True)
            outer, gn = None, None
            if gate is not None:
                tx, mx = student.hidden[tr], student.hidden[values['meta']]
                with torch.no_grad():
                    ft = gate_features(evidence, student.head(tx).softmax(1), static)
                if epoch <= 5:
                    weights = tx.new_full((len(tr),), .5)
                else:
                    vstart = time.perf_counter()
                    outer, gn = meta_step(student.head, gate, gate_optimizer, tx, mx,
                                           values['meta_y'], evidence, ft,
                                           mass_direction(evidence) if mode == 'GATE_MASS' else None)
                    torch.cuda.synchronize()
                    virtual_seconds += time.perf_counter()-vstart
                    grad_norms.append(gn)
                    with torch.no_grad():
                        weights = gate(ft).sigmoid().squeeze(1)
                weights_history.append(weights.detach().cpu())
                t = gate_target(evidence, weights).detach() if mode != 'GATE_MASS' else gate_target(evidence, weights, mass_direction(evidence)).detach()
                assert not t.requires_grad
            elif mode in ('MEAN_LCSS', 'SHUFFLE_LCSS'):
                w = replay[epoch-1].double()
                if mode == 'MEAN_LCSS':
                    weights = torch.full_like(w, global_mass(evidence['m'], w))
                else:
                    perm = torch.randperm(len(w), generator=torch.Generator().manual_seed(20261024+epoch))
                    weights = match_mass(evidence['m'], w, w[perm])
                from targets import shrink
                t = shrink(evidence, weights).cuda().detach()
                weights_history.append(weights.cpu())
            logits = student.forward(tr, train=True)
            value = loss_value(logits, t, z, q)
            if mode in ('CLEAN_META_CE', 'CLEAN_META_GCE'):
                train_loss = loss_value(logits, t, z, .7 if mode == 'CLEAN_META_GCE' else None)
                value = .5*train_loss+.5*F.cross_entropy(student.forward(values['meta']), values['meta_y'])
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite loss '+str(folder))
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.model.parameters()):
                raise FloatingPointError('Nonfinite gradients '+str(folder))
            student.optimizer.step()
            student.model.eval()
            with torch.no_grad():
                lv = student.forward(vi)
                pred = lv.argmax(1)
                correct = int((pred == yv).sum())
                nll = float(F.cross_entropy(lv, yv))
                vf1 = f1_score(pred, yv, values['classes'])
            if correct > best:
                best, best_epoch, best_nll, best_f1 = correct, epoch, nll, vf1
                best_state = clone_state(student.model)
                best_gate = clone_state(gate) if gate is not None else None
            log.write(json.dumps(dict(epoch=epoch, loss=float(value.detach()), validation_correct=correct,
                                      validation_nll=nll, validation_macro_f1=vf1,
                                      meta_loss=outer, meta_gradient_norm=gn))+'\n')
            log.flush()
    student.check_frozen()
    torch.cuda.synchronize()
    seconds += time.perf_counter()-started
    last = dict(model=clone_state(student.model), optimizer=student.optimizer.state_dict(), epoch=until,
                rng=rng_get(), best=best, best_epoch=best_epoch, best_nll=best_nll, best_f1=best_f1,
                best_state=best_state, best_gate=best_gate, seconds=seconds, virtual_seconds=virtual_seconds,
                weights_history=weights_history, grad_norms=grad_norms,
                gate=clone_state(gate) if gate is not None else None,
                gate_optimizer=gate_optimizer.state_dict() if gate is not None else None)
    torch.save(last, folder/'resume.pt')
    torch.save(dict(model=best_state, gate=best_gate, epoch=best_epoch, kind=backbone), folder/'checkpoint.pt')
    student.model.load_state_dict(best_state)
    student.model.eval()
    with torch.no_grad():
        probs = student.forward().softmax(1).cpu()
    torch.save(probs, folder/'probabilities.pt')
    if weights_history:
        torch.save(torch.stack(weights_history), folder/'weights_history.pt')
    out = dict(meta, epoch=until, status='complete' if until == 200 else 'prefix', best_epoch=best_epoch,
               valid_correct=best, valid_total=len(vi), validation=best/len(vi), valid_nll=best_nll,
               valid_macro_f1=best_f1, seconds=seconds, virtual_seconds=virtual_seconds,
               mean_meta_gradient_norm=sum(grad_norms)/len(grad_norms) if grad_norms else None,
               checkpoint_sha256=sha(folder/'checkpoint.pt'), probabilities_sha256=sha(folder/'probabilities.pt'),
               finished=now(), encoder_frozen_unchanged=backbone != 'R0')
    save(folder/'result.json', out)
    print(f'FIT {folder.relative_to(ROOT)} epoch={until} val={best/len(vi):.5f} sec={seconds:.2f}', flush=True)
    return out
