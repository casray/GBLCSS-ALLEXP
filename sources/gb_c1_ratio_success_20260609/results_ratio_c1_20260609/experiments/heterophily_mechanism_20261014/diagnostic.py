"""Frozen-geometry mechanism screen. Label-free relations are built before target evaluation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import platform

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_code/src"))
from gblcss.data import GraphData, percent_split_indices, inject_train_label_noise
from gblcss.granular_ball import build_granular_balls
from gblcss.representation import load_frozen_representation

DATA = {
 "corafull": PROJECT / "gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt",
 "coauthorcs": PROJECT / "external/NoisyGL/data/CS/processed/data.pt",
 "roman": ROOT / "data/roman_empire.npz",
}
CACHED_H = {
 "corafull": PROJECT / "gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/CoraFull/grace/H.pt",
 "coauthorcs": PROJECT / "gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/CoauthorCS/grace/H.pt",
}
CONFIG = dict(budgets=[64,128], min_child=32, split_improvement=.05, min_votes=2,
 beta=.5, ess_kappa=10., js_tau=.1, kernel_tau=.2, noise_conditions={"clean":0.,"uniform":.4,"pair":.4},
 protocol_seeds=list(range(5)), partition_seed="protocol_seed", pretrain_seeds=[0],
 stage="exploratory_frozen_geometry", label_access="only training observed labels for targets",
 representation_scope="one SSL seed per dataset; five protocol seeds are not five SSL runs")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""): h.update(block)
    return h.hexdigest()

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def fingerprint(*values):
    h=hashlib.sha256()
    for value in values:
        a=np.ascontiguousarray(value); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def mask(index,n):
    m=torch.zeros(n,dtype=torch.bool); m[torch.as_tensor(index)]=True; return m

def load_graph(name, seed):
    if name=="roman":
        with np.load(DATA[name],allow_pickle=False) as raw:
            x=torch.from_numpy(raw["node_features"].copy()).float()
            y=torch.from_numpy(raw["node_labels"].copy()).long()
            edges=to_undirected(torch.from_numpy(raw["edges"].copy()).long().T,num_nodes=len(y))
            indices=[np.flatnonzero(raw[k][seed]) for k in ("train_masks","val_masks","test_masks")]
    else:
        raw=torch.load(DATA[name],map_location="cpu",weights_only=False)[0]
        if isinstance(raw,dict): x,edges,y=raw["x"],raw["edge_index"],raw["y"]
        else: x,edges,y=raw.x,raw.edge_index,raw.y
        x,edges,y=x.float(),edges.long(),y.long()
        indices=percent_split_indices(len(y),seed)
    tr,va,te=indices
    if len(set(tr)&set(va)) or len(set(tr)&set(te)) or len(set(va)&set(te)):
        raise ValueError("overlapping split")
    return GraphData(name,x,edges,y,mask(tr,len(y)),mask(va,len(y)),mask(te,len(y)),
                     train_index=torch.as_tensor(tr),val_index=torch.as_tensor(va),test_index=torch.as_tensor(te))

def group_weights(groups, h=None):
    """CSR on ordered training nodes; no self-vote, one entry per voter."""
    groups=np.asarray(groups)
    rows=[]; cols=[]; vals=[]
    for g in np.unique(groups):
        ids=np.flatnonzero(groups==g)
        rr=np.repeat(ids,len(ids)); cc=np.tile(ids,len(ids))
        keep=rr!=cc; rr,cc=rr[keep],cc[keep]
        if h is None: vv=np.ones(len(rr),dtype=np.float64)
        else:
            # Row-constant exp(-1/tau) does not change normalized weights or Kish support.
            vv=np.exp((np.einsum("ij,ij->i",h[rr],h[cc])-1.)/CONFIG["kernel_tau"]).astype(np.float64)
        rows.append(rr); cols.append(cc); vals.append(vv)
    return sparse.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(len(groups),len(groups)))

def knn_weights(h, counts, device="cpu"):
    counts=np.asarray(counts,dtype=np.int64)
    if np.any(counts<0) or np.any(counts>=len(h)): raise ValueError("invalid requested unique support")
    k=int(counts.max(initial=0))
    if not k: return sparse.csr_matrix((len(h),len(h)),dtype=np.float64)
    z=torch.as_tensor(h,dtype=torch.float32,device=device)
    z=torch.nn.functional.normalize(z,dim=1)
    rows=[]; cols=[]
    with torch.no_grad():
        for start in range(0,len(h),256):
            stop=min(start+256,len(h))
            sims=z[start:stop] @ z.T
            sims[torch.arange(stop-start,device=device),torch.arange(start,stop,device=device)]=-torch.inf
            # Stable sort makes exact similarity ties reproducible by training-index order.
            order=torch.argsort(sims,dim=1,descending=True,stable=True)[:,:k].cpu().numpy()
            for offset,node in enumerate(range(start,stop)):
                rows.append(np.full(counts[node],node,dtype=np.int64))
                cols.append(order[offset,:counts[node]])
    del z
    rr,cc=np.concatenate(rows),np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(h),len(h)))

def vote(w, observed, n_classes):
    w=w.tocsr()
    if w.shape!=(len(observed),len(observed)) or np.any(w.data<0) or not np.isfinite(w.data).all():
        raise ValueError("invalid nonnegative voting weights")
    if np.any(w.diagonal()!=0): raise ValueError("self-vote")
    support=np.diff(w.indptr)
    total=np.asarray(w.sum(1)).ravel()
    squares=np.asarray(w.multiply(w).sum(1)).ravel()
    effective=np.divide(total**2,squares,out=np.zeros_like(total),where=squares>0)
    hist=w @ np.eye(n_classes)[observed]
    r=np.divide(hist,total[:,None],out=np.full_like(hist,1/n_classes),where=total[:,None]>0)
    maximum=r.max(1)
    alpha=np.clip((n_classes*maximum-1)/(n_classes-1),0,1)
    alpha[support<CONFIG["min_votes"]]=0
    tied=np.isclose(r,maximum[:,None],atol=1e-12,rtol=0).sum(1)>1
    q=r.copy(); unique=~tied
    q[unique]=(1-CONFIG["beta"])*r[unique]+CONFIG["beta"]*np.eye(n_classes)[r[unique].argmax(1)]
    return r,q,alpha,support,effective

def mix(q,alpha,observed):
    return (1-alpha[:,None])*np.eye(q.shape[1])[observed]+alpha[:,None]*q

def js_divergence(a,b):
    m=.5*(a+b)
    def kl(p):
        return np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300)),0),axis=1)
    return .5*(kl(a)+kl(b))

def alpha_controls(alpha, fine, coarse, effective, fine_count, coarse_count, seed):
    ess=effective/(effective+CONFIG["ess_kappa"])
    both=(fine_count>=2)&(coarse_count>=2)
    factor=np.ones_like(alpha)
    factor[both]=np.exp(-js_divergence(fine[both],coarse[both])/CONFIG["js_tau"])
    full=ess*factor
    scale=float(np.sum(alpha*full)/alpha.sum()) if alpha.sum()>0 else 0.
    shuffled=np.random.RandomState(10000+int(seed)).permutation(full)
    return dict(original=alpha,half=.5*alpha,ess=alpha*ess,js=alpha*factor,
                full=alpha*full,mean_matched=alpha*scale,shuffled=alpha*shuffled)

def target_metrics(t,q,alpha,clean,observed):
    truth=np.eye(t.shape[1])[clean]
    brier=((t-truth)**2).sum(1)
    initial_clean=observed==clean
    pred=t.argmax(1); changed=pred!=observed
    def avg(a,m): return float(np.mean(a[m])) if np.any(m) else None
    return dict(target_brier=float(brier.mean()),q_brier=float(((q-truth)**2).sum(1).mean()),
        q_accuracy=float((q.argmax(1)==clean).mean()),target_accuracy=float((pred==clean).mean()),
        mean_alpha=float(alpha.mean()),q_true_probability=float(q[np.arange(len(clean)),clean].mean()),
        clean_brier=avg(brier,initial_clean),noisy_brier=avg(brier,~initial_clean),
        clean_harm_rate=avg(pred!=clean,initial_clean),noise_recovery_rate=avg(pred==clean,~initial_clean),
        correction_coverage=float(changed.mean()),observed_error=float((~initial_clean).mean()))

def graph_stats(data):
    edges=data.edge_index.numpy()
    u,v=np.minimum(edges[0],edges[1]),np.maximum(edges[0],edges[1])
    pairs=np.unique(np.stack([u[u!=v],v[u!=v]],1),axis=0)
    y=data.y.numpy()
    degree=np.bincount(pairs.ravel(),minlength=len(y))
    byclass=np.bincount(y,weights=degree,minlength=data.num_classes)
    expected=float(np.square(byclass/byclass.sum()).sum())
    homo=float((y[pairs[:,0]]==y[pairs[:,1]]).mean())
    return dict(nodes=len(y),edges=len(pairs),classes=data.num_classes,edge_homophily=homo,
        degree_prior_agreement=expected,adjusted_homophily=(homo-expected)/(1-expected),
        feature_dim=data.x.shape[1])

def csv_write(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def summarize():
    import pandas as pd
    rows=[]
    for p in (ROOT/"runs").glob("*/*/seed*/metrics.json"):
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    csv_write(ROOT/"target_metrics.csv",rows)
    if not rows:return
    frame=pd.DataFrame(rows)
    keys=["dataset","representation","noise","arm"]
    numeric=[c for c in frame.select_dtypes("number").columns if c!="seed"]
    frame.groupby(keys)[numeric].agg(["mean","std","count"]).to_csv(ROOT/"target_summary.csv")
    pairs=[("GB128","GB64"),("GB128","KNN128"),("GB128","KM128"),
           ("GB128_full","GB128_mean_matched"),("GB128_full","GB128_shuffled"),
           ("GB128_weighted","GB128")]
    deltas=[]
    for left,right in pairs:
        a=frame[frame.arm==left];b=frame[frame.arm==right]
        merged=a.merge(b,on=["dataset","representation","noise","seed"],suffixes=("_a","_b"))
        for _,r in merged.iterrows():
            deltas.append(dict(dataset=r.dataset,representation=r.representation,noise=r.noise,seed=int(r.seed),
                comparison=left+" - "+right,brier_delta=float(r.target_brier_a-r.target_brier_b),
                target_accuracy_delta=float(r.target_accuracy_a-r.target_accuracy_b),
                clean_harm_delta=float(r.clean_harm_rate_a-r.clean_harm_rate_b)))
    csv_write(ROOT/"paired_deltas.csv",deltas)
    if deltas:
        df=pd.DataFrame(deltas)
        df.groupby(["dataset","representation","noise","comparison"])[["brier_delta","target_accuracy_delta","clean_harm_delta"]].agg(["mean","std","count"]).to_csv(ROOT/"paired_summary.csv")

def run_dataset(name,representation,hpath,seeds,device):
    logging.info("begin dataset=%s representation=%s",name,representation)
    first=load_graph(name,0)
    h=load_frozen_representation(hpath,len(first.y)).numpy()
    inputs={str(p.resolve()):sha(p) for p in [DATA[name],hpath,ROOT/"PLAN.md",ROOT/"diagnostic.py",
         WORKSPACE/"GB-LCSS_code/src/gblcss/data.py",WORKSPACE/"GB-LCSS_code/src/gblcss/granular_ball.py",
         WORKSPACE/"GB-LCSS_code/src/gblcss/targets.py"]}
    base=ROOT/"runs"/name/representation;base.mkdir(parents=True,exist_ok=True)
    write(base/"source_manifest.json",inputs);write(base/"graph_stats.json",graph_stats(first))
    write(base/"config.json",{**CONFIG,"dataset":name,"representation":representation,"h_path":str(hpath),
        "seeds_requested":seeds,"split":"official" if name=="roman" else "percent20_10_70"})
    for seed in seeds:
        folder=base/f"seed{seed}"
        if (folder/"metrics.json").exists():
            logging.info("skip completed %s",folder);continue
        folder.mkdir(exist_ok=True)
        data=load_graph(name,seed);tr=data.train_index.numpy()
        np.savez_compressed(folder/"split.npz",train=tr,val=data.val_index.numpy(),test=data.test_index.numpy())
        relations={};assignments={};stats={}
        for budget in CONFIG["budgets"]:
            logging.info("%s/%s seed%d constructing GB%d",name,representation,seed,budget)
            result=build_granular_balls(h,max_balls=budget,min_ball_size=32,split_improvement=.05,random_state=seed)
            assignments[f"GB{budget}"]=result.assignment
            km=KMeans(n_clusters=int(result.summary["num_balls"]),n_init=10,random_state=seed).fit_predict(h)
            assignments[f"KM{budget}"]=km
            for method,assignment in [(f"GB{budget}",result.assignment),(f"KM{budget}",km)]:
                relations[method]=group_weights(assignment[tr])
                stats[method]={"actual_groups":int(np.unique(assignment).size)}
            counts=np.diff(relations[f"GB{budget}"].indptr)
            relations[f"KNN{budget}"]=knn_weights(h[tr],counts,device=device)
            stats[f"KNN{budget}"]={"actual_groups":0}
        relations["GB128_weighted"]=group_weights(assignments["GB128"][tr],h[tr])
        stats["GB128_weighted"]=stats["GB128"]
        np.savez_compressed(folder/"assignments.npz",**assignments)
        write(folder/"relations_audit.json",{k:{"nonzeros":w.nnz,"self_votes":int(np.count_nonzero(w.diagonal())),
            "min_votes":int(np.diff(w.indptr).min()),"max_votes":int(np.diff(w.indptr).max()),
            "relation_fingerprint":fingerprint(w.indptr,w.indices,w.data)} for k,w in relations.items()})
        # Geometry complete. From here, only observed training labels enter target generation.
        all_rows=[]
        for kind,rate in CONFIG["noise_conditions"].items():
            data=inject_train_label_noise(data,"uniform" if kind=="clean" else kind,rate,seed)
            observed=data.noisy_y[tr].numpy()
            condition=folder/kind;condition.mkdir(exist_ok=True)
            np.save(condition/"observed_train.npy",observed)
            bundles={k:vote(w,observed,data.num_classes) for k,w in relations.items()}
            artifacts={}
            for arm,(r,q,a,c,eff) in bundles.items():
                artifacts[arm]=(r,q,a,c,eff,arm)
            r,q,a,c,eff=bundles["GB128"]
            rc,_,_,cc,_=bundles["GB64"]
            controls=alpha_controls(a,r,rc,eff,c,cc,seed)
            for label,aa in controls.items():
                if label!="original": artifacts["GB128_"+label]=(r,q,aa,c,eff,"GB128")
            artifacts["CE"]=(np.eye(data.num_classes)[observed],np.eye(data.num_classes)[observed],
                             np.zeros(len(tr)),np.zeros(len(tr)),np.zeros(len(tr)),None)
            target_hashes={}
            for arm,(r,q,a,c,eff,relation) in artifacts.items():
                t=mix(q,a,observed)
                if not np.isfinite(t).all() or np.min(t)<-1e-8 or not np.allclose(t.sum(1),1,atol=1e-6):
                    raise ValueError("invalid target")
                dest=condition/(arm+".npz")
                np.savez_compressed(dest,q=q.astype(np.float32),alpha=a.astype(np.float32),
                     target=t.astype(np.float32),support=c,effective=eff)
                target_hashes[arm]=sha(dest)
            write(condition/"frozen_targets.json",target_hashes)
            # Diagnostic ground truth is first used AFTER saving all deployable target artifacts.
            clean=data.y[tr].numpy()
            for arm,(r,q,a,c,eff,relation) in artifacts.items():
                t=mix(q,a,observed)
                metric=target_metrics(t,q,a,clean,observed)
                if relation:
                    true_r=vote(relations[relation],clean,data.num_classes)[0]
                    competitor=true_r.copy();competitor[np.arange(len(clean)),clean]=-np.inf
                    margin=true_r[np.arange(len(clean)),clean]-competitor.max(1)
                    margin[c<2]=np.nan
                    valid=c>=2
                    positive=float(((margin>0)&valid).mean())
                    meanmargin=float(np.nanmean(margin)) if valid.any() else None
                else: positive=None;meanmargin=None
                all_rows.append(dict(dataset=name,representation=representation,seed=seed,noise=kind,arm=arm,
                  **metric,mean_support=float(np.mean(c)),mean_effective=float(np.mean(eff)),
                  eligible_rate=float(np.mean(c>=2)),positive_clean_margin_fraction=positive,mean_clean_margin=meanmargin))
            write(condition/"protocol.json",dict(dataset=name,seed=seed,noise=kind,rate=rate,
                  split_fingerprint=fingerprint(tr,data.val_index.numpy(),data.test_index.numpy()),
                  noisy_train_fingerprint=fingerprint(observed),source_manifest="see parent",
                  clean_val_unchanged=bool(torch.equal(data.noisy_y[data.val_mask],data.y[data.val_mask])),
                  clean_test_unchanged=bool(torch.equal(data.noisy_y[data.test_mask],data.y[data.test_mask]))))
            logging.info("%s/%s seed%d %s frozen+evaluated %d arms",name,representation,seed,kind,len(artifacts))
        write(folder/"metrics.json",all_rows)
        summarize()
    for p,digest in inputs.items():
        if sha(p)!=digest: raise RuntimeError("input changed during diagnostic: "+p)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=["corafull","coauthorcs"])
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(5)))
    parser.add_argument("--representation",default="grace")
    parser.add_argument("--h-path",type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/"diagnostic_raw.log",encoding="utf-8")])
    if args.summarize_only:summarize();return
    if args.h_path and len(args.datasets)!=1:parser.error("--h-path requires one dataset")
    torch.set_num_threads(4)
    write(ROOT/"config.json",CONFIG)
    write(ROOT/"environment.json",dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    start=time.time()
    with threadpool_limits(limits=4):
        for name in args.datasets:
            hp=args.h_path or CACHED_H.get(name)
            if hp is None or not hp.is_file(): raise FileNotFoundError(f"representation not ready: {name}/{args.representation}")
            run_dataset(name,args.representation,hp,args.seeds,args.device)
    write(ROOT/"last_execution.json",dict(datasets=args.datasets,representation=args.representation,
          seeds=args.seeds,elapsed_seconds=time.time()-start,status="completed"))
    logging.info("done seconds=%.2f",time.time()-start)

if __name__=="__main__":main()

