"""One fixed-feature linear kernel for both initializations and feature scales."""
import time,json
import torch
import torch.nn.functional as F
import artifacts
from data_bridge import state_hash

def fit_linear(folder,h,target,data,initial,expected):
    if artifacts.reusable(folder,expected):return artifacts.read(folder/'result.json')
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('Incomplete fit retained: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    model=torch.nn.Linear(h.shape[1],target.shape[1]).cuda();model.load_state_dict(initial)
    assert state_hash(model.state_dict())==expected['initial_hash']
    hd=h.cuda();td=target.cuda();train=data['train'].cuda();valid=data['valid'].cuda();vy=data['valid_y'].cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=expected['lr'],weight_decay=expected['weight_decay'])
    artifacts.save_tensor(folder/'initial.pt',initial);artifacts.save(folder/'config.json',expected)
    best=-1;best_epoch=0;state=None;start=time.perf_counter()
    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,expected['epochs']+1):
            optimizer.zero_grad(set_to_none=True)
            loss=-(td*F.log_softmax(model(hd[train]),dim=1)).sum(1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head loss')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            with torch.no_grad():correct=int((model(hd[valid]).argmax(1)==vy).sum())
            if correct>best:
                best,best_epoch=correct,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            log.write(json.dumps(dict(epoch=epoch,training_loss=float(loss.detach()),validation_correct=correct))+'\n');log.flush()
    model.load_state_dict(state)
    with torch.no_grad():p=model(hd).softmax(1).cpu()
    assert int((p[data['valid']].argmax(1)==data['valid_y']).sum())==best
    artifacts.save_tensor(folder/'student.pt',state);artifacts.save_tensor(folder/'probabilities.pt',p);artifacts.save_tensor(folder/'target.pt',target)
    result=dict(expected,status='complete',best_epoch=best_epoch,valid_correct=best,seconds=time.perf_counter()-start)
    artifacts.save(folder/'result.json',result)
    artifacts.complete(folder,expected,['initial.pt','student.pt','probabilities.pt','target.pt','config.json','epochs.jsonl','result.json'])
    return result
