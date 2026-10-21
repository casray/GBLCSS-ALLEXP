"""Fixed-endpoint GREET-portable-MC pretraining; no label/probe inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from methods import GreetPortable, weighted_filters, return_features, contrastive_loss

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
sys.path.insert(0,str(WORKSPACE/'experiments/heterophily_mechanism_20261014'))
import diagnostic_v2 as old
from gblcss.data import set_seed


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_knn(x,k,chunk=128):
    if len(x)<3:raise ValueError('At least three nodes required')
    z=F.normalize(x,dim=1); k=min(k,len(x)-2); rows=[]
    with torch.no_grad():
        for start in range(0,len(x),chunk):
            stop=min(start+chunk,len(x)); sims=z[start:stop]@z.T
            sims[torch.arange(stop-start,device=x.device),torch.arange(start,stop,device=x.device)]=-torch.inf
            rows.append(torch.argsort(sims,dim=1,descending=True,stable=True)[:,:k])
    return torch.cat(rows)


def masked_features(x,rate):
    keep=x.new_ones(x.shape[1]); keep[torch.randperm(x.shape[1],device=x.device)[:int(rate*x.shape[1])]]=0
    return x*keep


def batches(n,size,k):
    order=list(range(n)); random.shuffle(order)
    groups=[order[i:i+size] for i in range(0,n,size)]
    if len(groups)>1 and len(groups[-1])<=k+1:groups[-2].extend(groups.pop())
    return groups


def train_representation(x,edges,se,knn,config,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    set_seed(config['seed'])
    model=GreetPortable(x.shape[1],config['half_dim'],config['projection_dim'],config['layers'],
                        se.shape[1],config['highpass_alpha']).to(x.device)
    enc_params=model.encoder_parameters(); disc_params=model.discriminator_parameters()
    optimizer=torch.optim.Adam(enc_params,lr=config['learning_rate'])
    disc_optimizer=torch.optim.Adam(disc_params,lr=config['learning_rate'])
    start=time.perf_counter(); losses=[]
    with (folder/'training.jsonl').open('a',encoding='utf-8') as stream:
        for epoch in range(config['epochs']):
            model.train(); cl_value=0.
            for _ in range(config['cl_rounds']):
                with torch.no_grad():weights=model.sample_weights(x,se,edges)
                indices,low,high=weighted_filters(edges,weights,len(x),config['highpass_alpha'])
                z1=model.project_channel(masked_features(x,config['maskfeat_rates'][0]),indices,
                                         F.dropout(low,p=config['dropedge_rates'][0],training=True),'low')
                z2=model.project_channel(masked_features(x,config['maskfeat_rates'][1]),indices,
                                         F.dropout(high,p=config['dropedge_rates'][1],training=True),'high')
                loss=x.new_zeros(())
                for group in batches(len(x),config['batch_size'],knn.shape[1]):
                    ids=torch.as_tensor(group,device=x.device)
                    loss=loss+len(group)/len(x)*(contrastive_loss(z1[ids],z2[ids],ids,knn)+
                                                contrastive_loss(z2[ids],z1[ids],ids,knn))/2
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite contrastive loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                cl_value+=float(loss.detach())/config['cl_rounds']
                del z1,z2,loss,indices,low,high
            for parameter in enc_params:parameter.requires_grad_(False)
            weights=model.sample_weights(x,se,edges)
            indices,low,high=weighted_filters(edges,weights,len(x),config['highpass_alpha'])
            embedding=model.encode(x,indices,low,high)
            u,v=edges
            rand_u=torch.randint(len(x),(len(u),),device=x.device)
            rand_v=(rand_u+torch.randint(1,len(x),(len(u),),device=x.device))%len(x)
            edge_sim=F.cosine_similarity(embedding[u],embedding[v])
            random_sim=F.cosine_similarity(embedding[rand_u],embedding[rand_v])
            rank=(F.relu(random_sim-edge_sim+config['margin'])*F.relu(weights-.5)).mean()
            rank=rank+(F.relu(edge_sim-random_sim+config['margin'])*F.relu(.5-weights)).mean()
            rank=rank/2
            if not bool(torch.isfinite(rank)):raise FloatingPointError('Nonfinite rank loss')
            disc_optimizer.zero_grad(set_to_none=True); rank.backward(); disc_optimizer.step()
            for parameter in enc_params:parameter.requires_grad_(True)
            row=dict(epoch=epoch+1,cl_loss=cl_value,rank_loss=float(rank.detach()),elapsed_seconds=time.perf_counter()-start)
            losses.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            if epoch==0 or (epoch+1)%10==0:
                logging.info('%s epoch%d/%d CL %.5f rank %.5f elapsed %.1fs',folder.name,epoch+1,
                             config['epochs'],cl_value,row['rank_loss'],row['elapsed_seconds'])
                write(folder/'progress.json',dict(status='training',**row))
            del embedding,weights,indices,low,high,rank,edge_sim,random_sim
    model.eval(); h,weights=model.export(x,se,edges,config['export_seed'])
    torch.save(h.cpu(),folder/'H.pt'); torch.save(weights.cpu(),folder/'edge_weights.pt')
    torch.save(model.state_dict(),folder/'checkpoint.pt')
    summary=dict(status='completed',epochs=config['epochs'],shape=list(h.shape),label_usage='none',
                 elapsed_seconds=time.perf_counter()-start,H_sha256=sha(folder/'H.pt'),
                 weights_sha256=sha(folder/'edge_weights.pt'),checkpoint_sha256=sha(folder/'checkpoint.pt'),
                 export_seed=config['export_seed'],final_loss=losses[-1])
    write(folder/'summary.json',summary)
    return summary


def prepare(name,config,device):
    folder=ROOT/'representations'/name
    identity=dict(config=config,method_sha256=sha(ROOT/'methods.py'),trainer_sha256=sha(__file__),
                  data_sha256=old.sha(old.DATA[name]),label_usage='none; loader metadata includes truth but trainer API receives only X/A')
    if (folder/'summary.json').exists():
        if json.loads((folder/'config.json').read_text(encoding='utf-8'))!=identity:raise ValueError('Representation config changed')
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        for filename,key in [('H.pt','H_sha256'),('edge_weights.pt','weights_sha256')]:
            if sha(folder/filename)!=summary[key]:raise ValueError('Representation hash changed')
        return
    write(folder/'config.json',identity)
    graph=old.load_graph(name,0)
    keep=graph.edge_index[0]!=graph.edge_index[1]; edges=graph.edge_index[:,keep]
    logging.info('PREPARE %s nodes%d features%d directed_edges%d',name,len(graph.x),graph.x.shape[1],edges.shape[1])
    torch.save(edges,folder/'edges.pt')
    se=return_features(edges.numpy(),len(graph.x),hops=config['se_hops'],walks=config['se_walks'],
                       seed=config['seed'],block=config['se_block'])
    np.save(folder/'structural_encoding.npy',se)
    x=graph.x.to(device); edges=edges.to(device); se=torch.from_numpy(se).to(device)
    logging.info('KNN raw features %s',name)
    knn=feature_knn(x,config['k']); torch.save(knn.cpu(),folder/'knn.pt')
    train_representation(x,edges,se,knn,config,folder)
    del x,edges,se,knn,graph
    if torch.cuda.is_available():torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--datasets',nargs='+'); parser.add_argument('--device',default='cuda')
    args=parser.parse_args(); config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'greet_raw.log',encoding='utf-8')])
    torch.set_num_threads(config['threads'])
    with threadpool_limits(limits=config['threads']):
        for name in args.datasets or config['datasets']:prepare(name,config['greet'],args.device)


if __name__=='__main__':main()
