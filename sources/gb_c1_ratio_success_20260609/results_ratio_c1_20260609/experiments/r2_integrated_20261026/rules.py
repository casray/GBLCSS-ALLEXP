"""Frozen validation-only ranking and causal trajectory replay."""
import statistics
from pathlib import Path

import torch

from targets import groups

SUPPORTS = (4,8,16,32,64,128)
PAIRS = ((0.,1.),(.25,1.),(1.,.25),(1.,0.))
ALLOWED = {'arm','valid_correct','valid_total','valid_nll','mean_lambda','support','best_epoch','epoch'}


def candidates():
    out={}
    for s in SUPPORTS:
        for l in (.25,.5,1.):
            out[f'S{s}_L{l}']=dict(support=s,kind='uniform',strength=l)
        for a,c in PAIRS:
            out[f'S{s}_G_A{a}_C{c}']=dict(support=s,kind='group',agree=a,conflict=c)
    return out


def weights(e, spec):
    if spec['kind']=='uniform':
        return torch.full_like(e['m'],spec['strength'])
    g=groups(e)
    return torch.where(g==0,spec['agree'],torch.where(g==1,spec['conflict'],1.)).double()


def fixed_id(arm):
    return (0 if arm=='CE' else 1 if '_L' in arm else 2,arm)


def rank_key(row):
    if set(row)-ALLOWED:
        raise ValueError('Non-validation selector field')
    return (-row['valid_correct']/row['valid_total'],row['valid_nll'],row['mean_lambda'],
            -row['support'],row['best_epoch'],fixed_id(row['arm']))


def prefix_score(logs, prefix, arm, support, mean_lambda, total):
    eligible=[r for r in logs if r['epoch']<=prefix]
    if [r['epoch'] for r in eligible]!=list(range(1,prefix+1)):
        raise ValueError('Incomplete causal prefix')
    best=max(eligible,key=lambda r:r['validation_correct'])
    return dict(arm=arm,support=support,mean_lambda=mean_lambda,epoch=prefix,
                best_epoch=best['epoch'],valid_correct=best['validation_correct'],
                valid_total=total,valid_nll=best['validation_nll'])


def survivors(rows,count,prefix):
    if any(r['epoch']!=prefix for r in rows):
        raise ValueError('Future score entered race')
    ranked=sorted(rows,key=rank_key)
    chosen=[r['arm'] for r in ranked[:count]]
    if 'S32_L1.0' in {r['arm'] for r in rows} and 'S32_L1.0' not in chosen:
        chosen[-1]='S32_L1.0'
    assert len(set(chosen))==count
    return chosen


def require_joint_frozen(path):
    if not Path(path).exists():
        raise RuntimeError('Fixed control completion requires frozen joint decision')


def mean_std(values):
    return statistics.mean(values),statistics.stdev(values) if len(values)>1 else None
