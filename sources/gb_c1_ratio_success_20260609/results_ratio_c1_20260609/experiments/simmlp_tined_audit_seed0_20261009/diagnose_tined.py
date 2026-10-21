"""Short validation-only single-factor probes; never replaces formal results."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'new_graph_mlp_seed0_20261009'
sys.path.insert(0, str(OLD))
import tined_adapter as adapter
from common import load_data, write_json


def validation(model, data, size):
    model.eval()
    with torch.no_grad():
        logits = adapter._student_logits(model, data.x[data.val_index], size)
        pred = logits.argmax(1)
        hist = pred.bincount(minlength=data.num_classes)
        return dict(valid=float((pred == data.val_y).double().mean()),
                    predicted_classes=int((hist > 0).sum()),
                    majority_fraction=float(hist.max() / hist.sum()))


def gradient_diagnostic(model, graph, data, teacher_logp, cfg):
    model.eval()
    params = list(model.layers.parameters())
    logits = adapter._student_logits(model, data.x, cfg['batch_size'])
    kl = F.kl_div(logits.log_softmax(1), teacher_logp, reduction='batchmean', log_target=True)
    ratios = torch.stack(model.get_appoximated_DE(data.x, graph))
    de = adapter.de_loss(ratios, model.DE_targets, cfg)
    kd_grad = torch.autograd.grad(kl, params, retain_graph=True)
    de_grad = torch.autograd.grad(de, params)
    dot = sum((a*b).sum() for a,b in zip(kd_grad, de_grad))
    a = sum(t.square().sum() for t in kd_grad).sqrt()
    b = sum(t.square().sum() for t in de_grad).sqrt()
    stages = []
    with torch.no_grad():
        h = data.x
        for i, layer in enumerate(model.layers):
            h = layer(h)
            if i < len(model.layers)-1:
                h = F.relu(model.norms[i](h))
                stages.append(dict(layer=i, zero_fraction=float((h == 0).float().mean()),
                                   dead_channels=int((h.abs().sum(0) == 0).sum()),
                                   width=h.shape[1]))
    return dict(mode='eval, full graph, at saved epoch-4 student; no dropout sampling',
                kl=float(kl.detach()), de=float(de.detach()), kd_grad_norm=float(a), de_grad_norm=float(b),
                gradient_cosine=float(dot / (a*b).clamp_min(1e-30)),
                log_de_ratios=ratios.detach().cpu().tolist(), relu_stages=stages)


def main():
    torch.set_num_threads(4)
    data, audit = load_data('corafull')
    ckpt = torch.load(OLD / 'runs/corafull/tined/checkpoint.pt', map_location='cuda', weights_only=False)
    cfg = copy.deepcopy(ckpt['config'])
    directory = ROOT / 'tined_diagnostics'
    directory.mkdir(exist_ok=True)
    if (directory / 'summary.json').exists():
        raise RuntimeError('Existing diagnostics must be preserved')
    write_json(directory / 'protocol.json', audit)
    write_json(directory / 'original_config.json', cfg)
    graph = adapter.SparseGraph(data.edge_index, data.x.shape[0], data.x.dtype)
    teacher = adapter.SAGETeacher(ckpt['teacher_config']['dims'], cfg['dropout']).cuda()
    teacher.load_state_dict(ckpt['teacher_state'])
    teacher.eval()
    teacher.requires_grad_(False)
    with torch.no_grad():
        logits, targets = teacher.forward_with_targets(graph, data.x)
        teacher_logp = logits.log_softmax(1).detach()
    adapter._seed()
    selected = adapter.TINEDStudent(teacher, cfg)
    selected.load_state_dict(ckpt['student_state'])
    gradient = gradient_diagnostic(selected, graph, data, teacher_logp, cfg)
    write_json(directory / 'saved_checkpoint_gradient.json', gradient)
    del selected
    original_step = adapter._step
    variants = [('baseline', {}), ('de_weight_zero', {'DE_regularization_rate': 0.}),
                ('dropout_zero', {'dropout': 0.})]
    summaries = []
    for name, overrides in variants:
        settings = {**cfg, 'student_epochs': 20, **overrides}
        adapter._seed()
        model = adapter.TINEDStudent(teacher, settings)
        model.set_de_targets(targets)
        phases = []
        steps = math.ceil(len(data.train_index) / cfg['batch_size'])
        kd_steps = math.ceil(len(data.x) / cfg['batch_size'])
        total = steps + kd_steps + 1
        counter = [0]

        def traced_step(optimizer, loss):
            position = counter[0] % total + 1
            epoch = counter[0] // total + 1
            original_step(optimizer, loss)
            counter[0] += 1
            if position in [steps, steps + kd_steps, total]:
                phase = {steps: 'after_ce', steps+kd_steps: 'after_kl', total: 'after_de'}[position]
                metric = validation(model, data, cfg['batch_size'])
                phases.append(dict(epoch=epoch, phase=phase, **metric))
                model.train()

        history = []
        initial = validation(model, data, cfg['batch_size'])
        adapter._step = traced_step
        try:
            _, best = adapter.fit_student(model, graph, data, teacher_logp, settings, history.append)
        finally:
            adapter._step = original_step
        write_json(directory / f'{name}_config.json', settings)
        with (directory / f'{name}_epochs.jsonl').open('w') as f:
            for record in history:
                f.write(json.dumps(record) + '\n')
        write_json(directory / f'{name}_phases.json', phases)
        summary = dict(variant=name, changed=overrides, initial=initial, best=best,
                       final=validation(model, data, cfg['batch_size']),
                       test_labels_read=False, teacher_retrained=False, student_epochs=20)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        del model
    baseline = summaries[0]
    original_best = ckpt['student_best']['val']
    if baseline['best']['epoch'] != 4 or abs(baseline['best']['val'] - original_best) > 1e-8:
        raise RuntimeError('Diagnostic baseline did not reproduce original prefix')
    write_json(directory / 'summary.json', dict(variants=summaries,
               teacher_validation=ckpt['teacher_best']['val'], gradient=gradient,
               baseline_prefix_matches=True, test_labels_read=False))


if __name__ == '__main__':
    main()
