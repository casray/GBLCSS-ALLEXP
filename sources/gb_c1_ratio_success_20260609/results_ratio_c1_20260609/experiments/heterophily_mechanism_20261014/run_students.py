"""Run preregistered X-only MLP pilots from frozen diagnostic targets."""
from pathlib import Path
import argparse
import json
import logging
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import diagnostic as d
from gblcss.data import inject_train_label_noise, set_seed
from gblcss.training import train_x_only_mlp

TRAINING=dict(hidden=64,layers=2,dropout=.5,learning_rate=.01,weight_decay=.0005,epochs=200)

def full_targets(noisy,train,n_classes,local):
    t=torch.nn.functional.one_hot(noisy,n_classes).float()
    values=torch.as_tensor(local,dtype=torch.float32)
    if values.shape!=(len(train),n_classes) or not torch.allclose(values.sum(1),torch.ones(len(train)),atol=1e-6):
        raise ValueError("target shape/probability mismatch")
    t[train]=values
    return t

def aggregate(rows):
    keys=sorted(set((r["dataset"],r["representation"],r["noise"],r["arm"]) for r in rows))
    out=[]
    for key in keys:
        values=[r["test_accuracy"] for r in rows if tuple(r[k] for k in ("dataset","representation","noise","arm"))==key]
        out.append(dict(zip(("dataset","representation","noise","arm"),key),
                        n=len(values),test_mean=float(np.mean(values)),
                        test_std=float(np.std(values,ddof=1)) if len(values)>1 else None))
    return out

def summarize():
    rows=[json.loads(p.read_text(encoding="utf-8")) for p in (ROOT/"students").glob("*/*/seed*/*/*/result.json")]
    d.csv_write(ROOT/"student_results.csv",rows)
    if rows:
        summary=aggregate(rows)
        d.write(ROOT/"student_summary.json",summary)
        d.csv_write(ROOT/"student_summary.csv",summary)
        lookup={(r["dataset"],r["representation"],r["seed"],r["noise"],r["arm"]):r for r in rows}
        pairs=[]
        for key,r in lookup.items():
            baseline=lookup.get((*key[:4],"CE"))
            if baseline and key[-1]!="CE":
                pairs.append(dict(dataset=key[0],representation=key[1],seed=key[2],noise=key[3],arm=key[4],
                                  accuracy_minus_ce_pp=100*(r["test_accuracy"]-baseline["test_accuracy"])))
        d.csv_write(ROOT/"student_paired_deltas.csv",pairs)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",nargs="+",default=["corafull","coauthorcs"])
    p.add_argument("--representation",default="grace")
    p.add_argument("--seeds",type=int,nargs="+",default=[0])
    p.add_argument("--conditions",nargs="+",default=["clean","uniform","pair"])
    p.add_argument("--arms",nargs="+",default=["CE","GB64","GB128","KNN128"])
    p.add_argument("--device",default="cuda")
    p.add_argument("--summarize-only",action="store_true")
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/"student_raw.log",encoding="utf-8")])
    if args.summarize_only:summarize();return
    if any(a not in ("CE","GB64","GB128","KNN128") for a in args.arms):
        p.error("new-method training requires a recorded mechanism gate; only preregistered baseline arms enabled")
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for name in args.datasets:
            for seed in args.seeds:
                data=d.load_graph(name,seed);tr=data.train_index
                parent=ROOT/"runs"/name/args.representation/f"seed{seed}"
                if not (parent/"metrics.json").exists():raise RuntimeError("diagnostic must finish before student: "+str(parent))
                for condition in args.conditions:
                    rate=d.CONFIG["noise_conditions"][condition]
                    data=inject_train_label_noise(data,"uniform" if condition=="clean" else condition,rate,seed)
                    src=parent/condition
                    recorded=json.loads((src/"protocol.json").read_text(encoding="utf-8"))
                    if d.fingerprint(data.noisy_y[tr].numpy())!=recorded["noisy_train_fingerprint"]:
                        raise RuntimeError("noise mismatch")
                    if d.fingerprint(tr.numpy(),data.val_index.numpy(),data.test_index.numpy())!=recorded["split_fingerprint"]:
                        raise RuntimeError("split mismatch")
                    frozen=json.loads((src/"frozen_targets.json").read_text(encoding="utf-8"))
                    for arm in args.arms:
                        dest=ROOT/"students"/name/args.representation/f"seed{seed}"/condition/arm
                        if (dest/"result.json").exists():continue
                        dest.mkdir(parents=True,exist_ok=True)
                        artifact=src/(arm+".npz")
                        if d.sha(artifact)!=frozen[arm]:raise RuntimeError("target changed after freeze")
                        with np.load(artifact) as a: targets=full_targets(data.noisy_y,tr,data.num_classes,a["target"])
                        config=dict(dataset=name,representation=args.representation,protocol_seed=seed,
                            student_seed=seed,condition=condition,arm=arm,training=TRAINING,device=args.device,
                            split_hash=recorded["split_fingerprint"],noise_hash=recorded["noisy_train_fingerprint"],
                            target_path=str(artifact),target_hash=frozen[arm],
                            training_source_hash=d.sha(d.WORKSPACE/"GB-LCSS_code/src/gblcss/training.py"),
                            model_source_hash=d.sha(d.WORKSPACE/"GB-LCSS_code/src/gblcss/models.py"),
                            pilot=len(args.seeds)==1,checkpoint_selection="clean validation only")
                        d.write(dest/"config.json",config)
                        logging.info("START %s %s seed%d %s %s",name,args.representation,seed,condition,arm)
                        start=time.time();set_seed(seed)
                        try:
                            result=train_x_only_mlp(data,targets,device=args.device,**TRAINING)
                        except Exception as exc:
                            d.write(dest/"failure.json",dict(type=type(exc).__name__,message=str(exc)))
                            raise
                        elapsed=time.time()-start
                        torch.save(result.logits,dest/"logits.pt")
                        row=dict(dataset=name,representation=args.representation,seed=seed,noise=condition,arm=arm,
                            train_accuracy=result.train_accuracy,validation_accuracy=result.validation_accuracy,
                            test_accuracy=result.test_accuracy,best_epoch=result.best_epoch,elapsed_seconds=elapsed,
                            split_hash=recorded["split_fingerprint"],noise_hash=recorded["noisy_train_fingerprint"],
                            target_hash=frozen[arm])
                        d.write(dest/"result.json",row);summarize()
                        logging.info("DONE %s seed%d %s %s test=%.6f val=%.6f epoch=%d seconds=%.2f",
                            name,seed,condition,arm,result.test_accuracy,result.validation_accuracy,result.best_epoch,elapsed)
                        del targets,result
                        if torch.cuda.is_available():torch.cuda.empty_cache()

if __name__=="__main__":main()

