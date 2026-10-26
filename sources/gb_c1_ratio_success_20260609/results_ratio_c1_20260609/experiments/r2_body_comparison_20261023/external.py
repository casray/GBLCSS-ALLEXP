"""Frozen author SimMLP encoder probes and GSSC author-core controls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

def run_external():
    import run as r
    sys.path.insert(0,str(r.EXP/'simmlp_tined_audit_seed0_20261009'))
    import simmlp_adapter as sim
    sys.path.insert(0,str(r.EXP/'new_graph_mlp_seed0_20261009'))
    import gssc_adapter as gssc
    from torch_geometric.data import Data
    for name in r.CFG['datasets']:
        raw=r.f.r2.raw_data(name);b,geo=r.fixed_data(name,0)
        x=raw.x.float().cuda();classes=int(raw.y.max())+1
        masks={k:torch.zeros(len(x),dtype=torch.bool,device='cuda') for k in ['train','valid','test']}
        for k in masks:masks[k][geo[k].cuda()]=True
        minimal=Data(x=x,edge_index=raw.edge_index.long().cuda(),num_classes=classes)
        checkpoint_path=r.EXP/'simmlp_tined_audit_seed0_20261009/runs'/name/'simmlp/checkpoint.pt'
        ck=r.load(checkpoint_path);cfg=ck['config'];model,obj=sim.build_model(minimal,cfg)
        model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():emb=model.encode(x).detach()
        for cond in r.CFG['conditions']:
            pack=r.load(b/f'{cond}.pt');tr=geo['train'].cuda();vi=geo['valid'].cuda()
            for method in ['SIMMLP','GSSC']:
                folder=r.ROOT/'external'/name/cond/method
                if (folder/'result.json').exists():
                    prior=r.f.read(folder/'result.json');assert prior['status']=='complete';continue
                folder.mkdir(parents=True,exist_ok=True)
                r.save(r.ROOT/'status.json',dict(stage='external',dataset=name,condition=cond,method=method))
                started=time.perf_counter()
                if method=='SIMMLP':
                    r.f.set_seed(0);clf=obj['LogisticRegression'](cfg['hidden_dim'],classes).cuda()
                    initial=r.state_hash(clf.state_dict());opt=torch.optim.AdamW(clf.parameters(),lr=.01,weight_decay=.0005)
                    best=-1.;state=None
                    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as log:
                        for ep in range(1,201):
                            clf.train();opt.zero_grad(set_to_none=True);value=F.cross_entropy(clf(emb[tr]),pack['observed'].cuda());value.backward();opt.step();clf.eval()
                            with torch.no_grad():correct=int((clf(emb[vi]).argmax(1)==pack['valid_y'].cuda()).sum())
                            if correct>=best:best=correct;selected=ep;state=deepcopy(clf.state_dict())
                            log.write(json.dumps(dict(epoch=ep,loss=float(value.detach()),validation_correct=correct))+'\n')
                    clf.load_state_dict(state);clf.eval()
                    with torch.no_grad():probs=clf(emb).softmax(1).cpu()
                    torch.save(dict(classifier=state,encoder_source=str(checkpoint_path)),folder/'checkpoint.pt')
                    r.save(folder/'config.json',dict(cfg,probe_initial_seed=0,pretraining_reused=True))
                    result=dict(best_epoch=selected,best_val=best/len(vi),initial_hash=initial,
                        author_encoder=str(checkpoint_path),selection='last tie',parameters=sum(p.numel() for p in model.encoder.parameters())+sum(p.numel() for p in clf.parameters()))
                else:
                    old=r.EXP/'new_graph_mlp_seed0_20261009/runs'/name/'gssc'
                    if cond=='uniform0.4':
                        r.load(old/'probabilities.pt');oldmeta=r.f.read(old/'result.json')
                        probs=torch.load(old/'probabilities.pt',weights_only=True,map_location='cpu')
                        result=dict(oldmeta,reused_source=str(old),selection='last tie')
                        r.save(folder/'config.json',r.f.read(old/'config.json'))
                    else:
                        labels=torch.full((len(x),),-1,dtype=torch.long,device='cuda');labels[tr]=pack['observed'].cuda()
                        data=Data(x=x,edge_index=raw.edge_index.long().cuda(),y=labels,
                                  train_index=tr,val_index=vi,test_index=geo['test'].cuda(),
                                  train_mask=masks['train'],val_mask=masks['valid'],test_mask=masks['test'],
                                  val_y=pack['valid_y'].cuda(),num_classes=classes,name=name)
                        config=gssc.resolve_config(name,dict(epochs=200))
                        r.f.set_seed(0);result=gssc.run(data,config,folder)
                        probs=torch.load(folder/'probabilities.pt',weights_only=True,map_location='cpu')
                        r.save(folder/'config.json',config)
                torch.save(probs,folder/'probabilities.pt')
                r.save(folder/'result.json',dict(result,status='complete',dataset=name,seed=0,condition=cond,arm=method,
                         probabilities_sha256=r.sha(folder/'probabilities.pt'),seconds=time.perf_counter()-started))
                print('EXTERNAL',name,cond,method,flush=True)
        del model,emb,x;torch.cuda.empty_cache()
    r.save(r.ROOT/'external_complete.json',dict(cells=40))
