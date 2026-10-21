"""Independent GPU worker: fixed train view, original-P model selection."""
import argparse
import copy
import json
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

from experiment import VARIANTS, atomic_json, file_hash
from legacy_api import protocol_labels, sparse_tensor, set_seed, build_targets, random_size_matched_assignment
from legacy_api import Checkpoints, Stage, array_hash
from operators import OperatorGCN, per_example_loss


def run(config_path, job_path, root):
    start=time.perf_counter();root=Path(root)
    config=json.loads(Path(config_path).read_text(encoding='utf8'))
    job=json.loads(Path(job_path).read_text(encoding='utf8'));v=VARIANTS[job['variant']]
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    if file_hash(config_path)!=protocol['config_sha256']:
        raise ValueError('frozen configuration changed')
    torch.set_num_threads(config['cpu_threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    device=torch.device('cuda');sync=torch.cuda.synchronize
    process=psutil.Process();resource={'rss':process.memory_info().rss};stop=threading.Event()
    def sample():
        while not stop.wait(.1):
            resource['rss']=max(resource['rss'],process.memory_info().rss)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start();times={}
    with Stage(times,'load_cache_s'):
        folder=root/'cache'/job['dataset']
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        if meta['config_sha256']!=protocol['config_sha256']:
            raise ValueError('cache config mismatch')
        old=Path(meta['old_directory'])
        for name,expected in meta['original_cache_files'].items():
            if (old/name).stat().st_size!=expected['bytes']:
                raise ValueError('original cache size changed')
        y=np.load(old/'y.npy');x=np.load(old/'x.npy',mmap_mode='r');p=sp.load_npz(old/'p.npz')
        tr,va,te,permitted=protocol_labels(y,job['seed'],job['noise'],meta['C'])
    with Stage(times,'targets_s'):
        part=v['partition']
        if part=='none':
            assignment=np.arange(len(y),dtype=np.int64)
        else:
            filename='kmeans.npy' if part=='kmeans' else 'granule.npy'
            if file_hash(folder/filename)!=meta['new_files'][filename]:
                raise ValueError('partition changed')
            assignment=np.load(folder/filename)
            if part=='random':
                assignment=random_size_matched_assignment(assignment,900000+job['seed'])
        targets,q,support,diagnostics=build_targets(permitted,assignment,tr,meta['C'],v['target'])
        if not np.isfinite(targets).all() or not np.allclose(targets.sum(1),1):
            raise ValueError('invalid target probabilities')
        m=len(support);selected=np.flatnonzero(support>0)
    torch.cuda.reset_peak_memory_stats();free_before,total_memory=torch.cuda.mem_get_info()
    with Stage(times,'materialize_s',sync):
        p_full=sparse_tensor(p,device);x_full=torch.from_numpy(np.array(x)).to(device)
        pc=None
        if v['coarse']:
            h=sp.csr_matrix((np.ones(len(y),np.float32),(np.arange(len(y)),assignment)),shape=(len(y),m))
            r=(sp.diags(1/np.bincount(assignment,minlength=m))@h.T).tocsr().astype(np.float32)
            pc=(r@p@h).tocsr().astype(np.float32)
            fit_x=torch.from_numpy((r@x).astype(np.float32)).to(device);fit_p=sparse_tensor(pc,device)
            model_assignment=torch.arange(m,device=device)
            fit_ids=torch.as_tensor(selected,device=device)
            fit_targets=torch.from_numpy(q[selected]).to(device)
            weights=torch.from_numpy((support[selected]/support.sum()).astype(np.float32)).to(device)
        else:
            fit_x,fit_p=x_full,p_full
            model_assignment=torch.as_tensor(assignment,device=device)
            fit_ids=torch.as_tensor(tr,device=device)
            fit_targets=torch.from_numpy(targets).to(device)
            weights=torch.full((len(tr),),1/len(tr),device=device)
        def make_model():
            return OperatorGCN(x.shape[1],config['hidden'],meta['C'],config['dropout'],
                model_assignment,mode=v['mode'],mask_mode=v['mask']).to(device)
        set_seed(job['seed']);model=make_model()

    def loss_for(logits, mix_rng=None):
        tt=fit_targets
        kind=v['loss']
        if kind=='postprop_mixup':
            lam=float(mix_rng.beta(config['mixup_alpha'],config['mixup_alpha']))
            order=torch.as_tensor(mix_rng.permutation(len(tt)),device=device)
            logits=lam*logits+(1-lam)*logits[order]
            tt=lam*tt+(1-lam)*tt[order]
            kind='ce'
        return (per_example_loss(logits,tt,kind,q=config['gce_q'],epsilon=config['smoothing_epsilon'])*weights).sum()

    def original_logits():
        # Exactly the old GCN's eval forward. It also supports coarse-trained parameters.
        z=F.relu(torch.sparse.mm(p_full,model.l1(x_full)))
        return torch.sparse.mm(p_full,model.l2(z))

    with Stage(times,'warmup_s',sync):
        model.train();rng=np.random.default_rng(job['seed'])
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            loss_for(model(fit_x,fit_p)[fit_ids],rng).backward()
    del model
    set_seed(job['seed']);model=make_model()
    initial_parameter_hash=array_hash(*[t.detach().cpu().numpy() for t in model.parameters()])
    rng=np.random.default_rng(job['seed'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    bank=Checkpoints(config['patience_checks']);history=[]
    with Stage(times,'training_s',sync):
        for epoch in range(1,config['epochs']+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model(fit_x,fit_p)[fit_ids],rng)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            if epoch==1 and any(not torch.isfinite(t.grad).all().item() for t in model.parameters()):
                raise FloatingPointError('nonfinite initial gradients')
            optimizer.step()
            if epoch%config['eval_every']==0 or epoch==config['epochs']:
                with Stage(times,'validation_s',sync):
                    model.eval()
                    with torch.no_grad():
                        pred=original_logits().argmax(1).cpu().numpy()
                    val=float(np.mean(pred[va]==y[va]))
                    bank.observe(epoch,val,None,{k:t.detach().cpu().clone() for k,t in model.state_dict().items()})
                    history.append(dict(epoch=epoch,primary_val=val,train_loss=float(loss.detach().cpu())))
                if bank.stop:
                    break
    training_peak=torch.cuda.max_memory_allocated()/2**30
    del optimizer,loss
    model.zero_grad(set_to_none=True);model.load_state_dict(bank.primary_state);model.eval()
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with Stage(times,'primary_inference_s',sync):
        with torch.no_grad():
            logits=original_logits();pred=logits.argmax(1).cpu().numpy()
            probabilities=logits[torch.as_tensor(te,device=device)].softmax(1).cpu().numpy()
        del logits
    original_inference_peak=torch.cuda.max_memory_allocated()/2**30
    with Stage(times,'train_view_diagnostic_s',sync):
        with torch.no_grad():
            view=model(fit_x,fit_p).argmax(1).cpu().numpy()
        if v['coarse']:
            view=view[assignment]
    view_peak=torch.cuda.max_memory_allocated()/2**30
    checkpoint=root/'checkpoints'/f"{job['id']}.pt";checkpoint.parent.mkdir(exist_ok=True)
    torch.save(dict(state_dict=bank.primary_state,job=job,best_epoch=bank.primary_epoch,
        config_sha256=protocol['config_sha256'],source_sha256=protocol['source_sha256']),checkpoint)
    predpath=root/'results'/f"{job['id']}_predictions.npz"
    np.savez(predpath,indices=te,prediction=pred[te],probabilities=probabilities,
        train_indices=tr,train_prediction=pred[tr],train_noisy_labels=permitted[tr],
        validation_indices=va,validation_prediction=pred[va],train_view_prediction=view[te],
        assignment=assignment,targets=targets)
    stop.set();sampler.join(timeout=1)
    original=meta['original_metadata'];required=original['times']['load_data_s']+original['times']['adjacency_s']
    if part in ('granule','random'):
        required+=original['times']['coordinates_s']+original['times']['tree_s']+meta['original_cut_cold_s']
    elif part=='kmeans':
        required+=meta['coordinates_s']+meta['kmeans_s']
    worker_wall=time.perf_counter()-start
    result=dict(status='completed',job=job,variant_spec=v,config_sha256=protocol['config_sha256'],
        source_sha256=protocol['source_sha256'],split_sha256=array_hash(tr,va,te),
        noise_sha256=array_hash(permitted[tr]),assignment_sha256=array_hash(assignment),
        targets_sha256=array_hash(targets),initial_parameter_sha256=initial_parameter_hash,
        source_data_sha256=original['source_sha256'],source_cache_metadata_sha256=meta['original_metadata_sha256'],
        N=len(y),C=meta['C'],L=len(tr),M=m,M_plus=int((support>0).sum()),
        coarse_edges=int(pc.nnz) if pc is not None else None,target_diagnostics=diagnostics,
        actual_noise_rate=float(np.mean(permitted[tr]!=y[tr])),
        test_accuracy=float(np.mean(pred[te]==y[te])),
        macro_f1=float(f1_score(y[te],pred[te],labels=np.arange(meta['C']),average='macro',zero_division=0)),
        per_class_recall=recall_score(y[te],pred[te],labels=np.arange(meta['C']),average=None,zero_division=0).tolist(),
        train_clean_accuracy=float(np.mean(pred[tr]==y[tr])),train_noisy_accuracy=float(np.mean(pred[tr]==permitted[tr])),
        same_checkpoint_train_view_accuracy=float(np.mean(view[te]==y[te])),
        best_primary_val=bank.primary,best_primary_epoch=bank.primary_epoch,epochs=epoch,validation_history=history,
        times=times,worker_wall_s=worker_wall,required_historical_preparation_s=required,
        reconstructed_cold_s=worker_wall+required,central_preparation_wall_s=meta['preparation_wall_s'],
        cost_scope='worker process wall added by queue; reconstructed method-required historical preparation separate; central audit/coordinate/KM wall separately reported, not a measured cold repeat',
        gpu_training_peak_gib=training_peak,gpu_primary_inference_peak_gib=original_inference_peak,
        gpu_complete_peak_gib=max(training_peak,view_peak),gpu_total_gib=total_memory/2**30,
        gpu_free_before_gib=free_before/2**30,gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,
        host_rss_peak_gib=resource['rss']/2**30,prediction_file=predpath.name,prediction_sha256=file_hash(predpath),
        checkpoint_file=str(checkpoint.relative_to(root)),checkpoint_sha256=file_hash(checkpoint),
        interpretation='Qualification only; original-P validation; train-view diagnostic uses same selected checkpoint')
    atomic_json(root/'results'/f"{job['id']}.json",result)
    print(json.dumps({'id':job['id'],'accuracy':result['test_accuracy'],'epochs':epoch,'worker_wall_s':worker_wall}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--job',required=True);parser.add_argument('--run-root',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.config,args.job,args.run_root)
