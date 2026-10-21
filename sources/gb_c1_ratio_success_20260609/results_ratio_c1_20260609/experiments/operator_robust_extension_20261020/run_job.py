"""Independent robustness worker; original-P selection and label-safe supervision."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
_operators_path = HERE / 'prior' / 'operators.py'
if not _operators_path.is_file():
    _operators_path = HERE.parent / 'operator_training_view_20261020' / 'operators.py'
_spec = importlib.util.spec_from_file_location('_robust_prior_operators', _operators_path)
_operators = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_operators)
OperatorGCN = _operators.OperatorGCN


class DoseGCN(OperatorGCN):
    """Native biased CE GCN with (1-rho) P + rho Pi P Pi at both layers."""

    def __init__(self, din, hidden, classes, dropout, assignment, rho=0., mask_mode='independent'):
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError('rho must be in [0, 1]')
        super().__init__(din, hidden, classes, dropout, assignment, mode='P', mask_mode=mask_mode)
        self.rho = rho

    def _operator(self, z, p, original):
        if original or self.rho == 0:
            return _operators._apply_operator(z, p, self.assignment, 'P', self.num_groups)
        smooth = _operators._apply_operator(z, p, self.assignment, 'PIPPI', self.num_groups)
        if self.rho == 1:
            return smooth
        native = _operators._apply_operator(z, p, self.assignment, 'P', self.num_groups)
        return (1-self.rho)*native + self.rho*smooth

    def forward(self, x, p, original=False):
        _operators._validate_features(x, self.assignment)
        if x.shape[0] != self.assignment.numel():
            raise ValueError('input node count must match assignment')
        _operators._validate_p(p, x)
        z = F.relu(self._operator(self.l1(x), p, original))
        if self.training and not original:
            if self.mask_mode == 'independent':
                z = F.dropout(z, self.dropout, training=True)
            else:
                mask = F.dropout(z.new_ones((self.num_groups, z.shape[1])), self.dropout, training=True)
                z = z * mask[self.assignment]
        return self._operator(self.l2(z), p, original)


def trace_metrics(prediction, clean_y, noisy_y, train_ids):
    """Clean labels are consulted only here and in validation/test reporting."""
    pred = np.asarray(prediction)[train_ids]
    clean, noisy = np.asarray(clean_y)[train_ids], np.asarray(noisy_y)[train_ids]
    flipped = noisy != clean
    def mean(mask, targets):
        return float(np.mean(pred[mask] == targets[mask])) if mask.any() else None
    return dict(flipped_count=int(flipped.sum()), unflipped_count=int((~flipped).sum()),
        fit_wrong_label=mean(flipped, noisy), recover_true_label=mean(flipped, clean),
        unflipped_accuracy=mean(~flipped, clean), train_noisy_accuracy=float(np.mean(pred == noisy)),
        train_clean_accuracy=float(np.mean(pred == clean)))


def collect_trace(model, x, p, clean_y, noisy_y, train_ids, epoch):
    """Deterministic eval views with no gradient or random-stream feedback."""
    was_training = model.training
    devices = [x.device.index] if x.is_cuda else []
    rows = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for name, original in [('P', True), ('train', False)]:
                pred = model(x, p, original=original).argmax(1).cpu().numpy()
                rows.append(dict(epoch=epoch, view=name,
                    **trace_metrics(pred, clean_y, noisy_y, train_ids)))
    finally:
        model.train(was_training)
    return rows


def _cpu_state(module):
    return {k: value.detach().cpu().clone() for k, value in module.state_dict().items()}


def run(config_path, job_path, root):
    from experiment import VARIANTS, atomic_json, file_hash
    from legacy_api import protocol_labels, sparse_tensor, set_seed, Checkpoints, Stage, array_hash

    start = time.perf_counter()
    root = Path(root)
    config = json.loads(Path(config_path).read_text(encoding='utf8'))
    job = json.loads(Path(job_path).read_text(encoding='utf8'))
    variant = VARIANTS[job['variant']]
    settings = config[variant['family']]
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf8'))
    if file_hash(config_path) != protocol['config_sha256']:
        raise ValueError('frozen configuration changed')
    torch.set_num_threads(config['cpu_threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    device = torch.device('cuda')
    sync = torch.cuda.synchronize
    times = {}
    process = psutil.Process()
    resource = {'rss': process.memory_info().rss}
    stop = threading.Event()
    def sample():
        while not stop.wait(.1):
            resource['rss'] = max(resource['rss'], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with Stage(times, 'load_cache_s'):
            folder = root/'cache'/job['dataset']
            meta = json.loads((folder/'metadata.json').read_text(encoding='utf8'))
            if meta['config_sha256'] != protocol['config_sha256']:
                raise ValueError('cache config mismatch')
            old = Path(meta['old_directory'])
            for name, expected in meta['original_cache_files'].items():
                if (old/name).stat().st_size != expected['bytes']:
                    raise ValueError('original cache size changed')
            y = np.load(old/'y.npy')
            x = np.load(old/'x.npy', mmap_mode='r')
            p = sp.load_npz(old/'p.npz')
            tr, va, te, permitted = protocol_labels(y, job['seed'], job['noise'], meta['C'])
            if np.any(permitted[va]) or np.any(permitted[te]):
                raise ValueError('held-out labels must be hidden from supervision')
        with Stage(times, 'targets_s'):
            part = variant['partition']
            filename = 'kmeans.npy' if part == 'kmeans' else 'granule.npy'
            if file_hash(folder/filename) != meta['new_files'][filename]:
                raise ValueError('partition changed')
            assignment = np.load(folder/filename)
            m = int(assignment.max()) + 1
            support = np.bincount(assignment[tr], minlength=m)
            targets = np.eye(meta['C'], dtype=np.float32)[permitted[tr]]
            edges = None
            if variant['family'] == 'tfr':
                if file_hash(folder/'edge_index.npy') != meta['new_files']['edge_index.npy']:
                    raise ValueError('raw TFR graph changed')
                edges = np.load(folder/'edge_index.npy')
                if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in 'iu':
                    raise ValueError('invalid raw edge indices')
                if edges.size and (edges.min() < 0 or edges.max() >= len(y)):
                    raise ValueError('raw edge index outside node range')

        torch.cuda.reset_peak_memory_stats()
        free_before, total_memory = torch.cuda.mem_get_info()
        with Stage(times, 'materialize_s', sync):
            x_full = torch.from_numpy(np.array(x)).to(device)
            model_assignment = torch.as_tensor(assignment, dtype=torch.long, device=device)
            train_ids = torch.as_tensor(tr, dtype=torch.long, device=device)
            val_ids = torch.as_tensor(va, dtype=torch.long, device=device)
            noisy_y = torch.as_tensor(permitted, dtype=torch.long, device=device)
            decoder = encoder = None
            if variant['family'] == 'ce':
                graph = sparse_tensor(p, device)
                set_seed(job['seed'])
                model = DoseGCN(x.shape[1], settings['hidden'], meta['C'], settings['dropout'],
                    model_assignment, rho=variant['rho'], mask_mode=variant['mask']).to(device)
            else:
                from tfr_core import TFRClassifier, load_reference_classes, tfr_loss
                # Reference GCNLayers normalize this raw adjacency internally. The decoder
                # uses the same coalesced, loop-free edges as the previous local adapter.
                edge_index = torch.as_tensor(edges, dtype=torch.long, device=device)
                graph = torch.sparse_coo_tensor(edge_index,
                    torch.ones(edge_index.shape[1], device=device), (len(y), len(y))).coalesce()
                edge_index = graph.indices()
                _, Decoder = load_reference_classes()
                set_seed(job['seed'])
                model = TFRClassifier(x.shape[1], settings['n_hidden'], meta['C'], settings['dropout'],
                    model_assignment, rho=variant['rho'], mask_mode=variant['mask']).cuda()
                decoder = Decoder(meta['C'], settings['n_hidden'], x.shape[1],
                    settings['decoder_dropout'], settings['att_softmax']).cuda()
                # Upstream constructs and optimizes this unused layer, consuming RNG.
                encoder = torch.nn.Linear(x.shape[1], settings['n_hidden']).cuda()
            initial_parameter_hash = array_hash(*[t.detach().cpu().numpy() for t in model.parameters()])
            initial_decoder_hash = (array_hash(*[t.detach().cpu().numpy() for t in decoder.parameters()])
                                    if decoder is not None else None)
            initial_encoder_hash = (array_hash(*[t.detach().cpu().numpy() for t in encoder.parameters()])
                                    if encoder is not None else None)
            parameters = list(model.parameters())
            if decoder is not None:
                parameters += list(decoder.parameters()) + list(encoder.parameters())
            optimizer = torch.optim.Adam(parameters, lr=settings['lr'], weight_decay=settings['weight_decay'])

        bank = Checkpoints(settings.get('patience_checks', settings['epochs']+1))
        history, traces = [], []
        coefficient = 0.
        selected_decoder = selected_encoder = None
        with Stage(times, 'training_s', sync):
            for epoch in range(1, settings['epochs']+1):
                model.train()
                if decoder is not None:
                    # Preserve the reference's decoder eval mode after first validation.
                    decoder.train(epoch == 1)
                optimizer.zero_grad()
                logits = model(x_full, graph)
                used_coefficient = coefficient
                if decoder is None:
                    loss = F.cross_entropy(logits[train_ids], noisy_y[train_ids])
                else:
                    loss, components = tfr_loss(logits, x_full, noisy_y, train_ids, decoder,
                        edge_index, coefficient, settings)
                    del components
                    coefficient = float((logits[train_ids].argmax(1) == noisy_y[train_ids]).float().mean())
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                if epoch == 1 and any(t.grad is not None and not torch.isfinite(t.grad).all().item()
                                      for t in parameters):
                    raise FloatingPointError('nonfinite initial gradients')
                optimizer.step()
                due = epoch % settings['eval_every'] == 0 or epoch == settings['epochs']
                if due:
                    with Stage(times, 'validation_s', sync):
                        model.eval()
                        if decoder is not None:
                            decoder.eval()
                        with torch.no_grad():
                            validation_logits = model(x_full, graph, original=True)
                            if decoder is not None:
                                # Both reference evaluation calls consume edge augmentation
                                # RNG, while zeroed validation labels keep labels out of loss.
                                for _ in range(2):
                                    tfr_loss(validation_logits, x_full, noisy_y, val_ids,
                                        decoder, edge_index, coefficient, settings)
                            pred = validation_logits.argmax(1).cpu().numpy()
                        val = float(np.mean(pred[va] == y[va]))
                        if val > bank.primary and decoder is not None:
                            selected_decoder, selected_encoder = _cpu_state(decoder), _cpu_state(encoder)
                        bank.observe(epoch, val, None, _cpu_state(model))
                        record = dict(epoch=epoch, primary_val=val, val=val,
                                      train_loss=float(loss.detach().cpu()))
                        if decoder is not None:
                            record.update(coefficient=coefficient, loss_coefficient=used_coefficient)
                        history.append(record)
                        del validation_logits
                final = epoch == settings['epochs'] or (decoder is None and due and bank.stop)
                if epoch == 1 or epoch % config['trace_every'] == 0 or final:
                    with Stage(times, 'training_trace_s', sync):
                        traces.extend(collect_trace(model, x_full, graph, y, permitted, tr, epoch))
                del logits
                if final:
                    break

        training_peak = torch.cuda.max_memory_allocated()/2**30
        del optimizer, loss, parameters
        model.zero_grad(set_to_none=True)
        model.load_state_dict(bank.primary_state)
        model.eval()
        if decoder is not None:
            decoder.zero_grad(set_to_none=True)
            decoder.load_state_dict(selected_decoder)
            decoder.eval()
            encoder.load_state_dict(selected_encoder)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with Stage(times, 'primary_inference_s', sync), torch.no_grad():
            logits = model(x_full, graph, original=True)
            pred = logits.argmax(1).cpu().numpy()
            probabilities = logits[torch.as_tensor(te, device=device)].softmax(1).cpu().numpy()
            del logits
        inference_peak = torch.cuda.max_memory_allocated()/2**30
        with Stage(times, 'train_view_diagnostic_s', sync), torch.no_grad():
            view = model(x_full, graph).argmax(1).cpu().numpy()
        view_peak = torch.cuda.max_memory_allocated()/2**30
        checkpoint = root/'checkpoints'/f"{job['id']}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=bank.primary_state, decoder_state_dict=selected_decoder,
            encoder_state_dict=selected_encoder, job=job, best_epoch=bank.primary_epoch,
            config_sha256=protocol['config_sha256'], source_sha256=protocol['source_sha256']), checkpoint)
        (root/'results').mkdir(parents=True, exist_ok=True)
        predpath = root/'results'/f"{job['id']}_predictions.npz"
        np.savez(predpath, indices=te, prediction=pred[te], probabilities=probabilities,
            train_indices=tr, train_prediction=pred[tr], train_noisy_labels=permitted[tr],
            validation_indices=va, validation_prediction=pred[va], train_view_prediction=view[te],
            assignment=assignment, targets=targets)
        tracepath = root/'traces'/f"{job['id']}.json"
        atomic_json(tracepath, traces)
        original = meta['original_metadata']
        required = original['times']['load_data_s'] + original['times']['adjacency_s']
        if part == 'kmeans':
            required += meta['coordinates_s'] + meta['kmeans_s']
        else:
            required += original['times']['coordinates_s'] + original['times']['tree_s'] + meta['original_cut_cold_s']
        worker_wall = time.perf_counter()-start
        result = dict(status='completed', job=job, variant_spec=variant,
            config_sha256=protocol['config_sha256'], source_sha256=protocol['source_sha256'],
            split_sha256=array_hash(tr, va, te), noise_sha256=array_hash(permitted[tr]),
            assignment_sha256=array_hash(assignment), targets_sha256=array_hash(targets),
            initial_parameter_sha256=initial_parameter_hash, source_data_sha256=original['source_sha256'],
            initial_decoder_parameter_sha256=initial_decoder_hash,
            initial_encoder_parameter_sha256=initial_encoder_hash,
            source_cache_metadata_sha256=meta['original_metadata_sha256'],
            N=len(y), C=meta['C'], L=len(tr), M=m, M_plus=int((support>0).sum()),
            coarse_edges=None, target_diagnostics={'target': 'own_noisy_label'},
            actual_noise_rate=float(np.mean(permitted[tr] != y[tr])),
            test_accuracy=float(np.mean(pred[te] == y[te])),
            macro_f1=float(f1_score(y[te], pred[te], labels=np.arange(meta['C']), average='macro', zero_division=0)),
            per_class_recall=recall_score(y[te], pred[te], labels=np.arange(meta['C']), average=None, zero_division=0).tolist(),
            train_clean_accuracy=float(np.mean(pred[tr] == y[tr])),
            train_noisy_accuracy=float(np.mean(pred[tr] == permitted[tr])),
            same_checkpoint_train_view_accuracy=float(np.mean(view[te] == y[te])),
            best_primary_val=bank.primary, best_primary_epoch=bank.primary_epoch, epochs=epoch,
            validation_history=history, times=times, worker_wall_s=worker_wall,
            required_historical_preparation_s=required, reconstructed_cold_s=worker_wall+required,
            central_preparation_wall_s=meta['preparation_wall_s'],
            cost_scope='Worker process and trace costs; reconstructed historical preparation is not a measured cold repeat; no efficiency claim.',
            gpu_training_peak_gib=training_peak, gpu_primary_inference_peak_gib=inference_peak,
            gpu_complete_peak_gib=max(training_peak, view_peak), gpu_total_gib=total_memory/2**30,
            gpu_free_before_gib=free_before/2**30, gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            host_rss_peak_gib=max(resource['rss'], process.memory_info().rss)/2**30,
            prediction_file=predpath.name, prediction_sha256=file_hash(predpath),
            checkpoint_file=str(checkpoint.relative_to(root)), checkpoint_sha256=file_hash(checkpoint),
            trace_file=tracepath.relative_to(root).as_posix(), trace_sha256=file_hash(tracepath),
            interpretation='Local fixed-feature protocol adaptation; original-P selection and test; own noisy-label supervision; train-view diagnostic at same selected checkpoint.',
            training_settings=settings)
        atomic_json(root/'results'/f"{job['id']}.json", result)
        print(json.dumps(dict(id=job['id'], accuracy=result['test_accuracy'], epochs=epoch,
                              worker_wall_s=worker_wall)), flush=True)
    finally:
        stop.set()
        sampler.join(timeout=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.config, args.job, args.run_root)
