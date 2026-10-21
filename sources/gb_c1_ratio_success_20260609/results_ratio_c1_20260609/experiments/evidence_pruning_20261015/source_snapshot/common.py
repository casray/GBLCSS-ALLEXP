"""Explicit paths for this experiment; previous experiment is read-only."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
PREVIOUS=ROOT.parent/'ball_count_diagnostic_20261015'
sys.path.insert(0,str(PREVIOUS))
import data_tools as dt
sys.path.insert(0,str(ROOT))
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def target_key(t):return hashlib.sha256(np.ascontiguousarray(t,dtype=np.float32).tobytes()).hexdigest()


def case_path(name,seed,noise):return ROOT/'targets'/name/f'seed{seed}'/noise


def old_case(name,seed,noise):return PREVIOUS/'targets'/name/f'seed{seed}'/noise


def load_tree(name,seed):
    from gblcss.granular_ball_pruning import EvidenceTree
    graph,protocol=dt.load_case(name,seed,'clean');train=graph.train_index.numpy()
    folder=PREVIOUS/'partitions'/name/f'seed{seed}'/'GREET'
    with np.load(folder/'assignments.npz') as a:final=a['B256'][train].copy()
    tree=EvidenceTree.from_trace(final,read(folder/'trajectory_trace.json'))
    # Validate decoded tree against every cached snapshot, not just count.
    with np.load(folder/'assignments.npz') as a:
        for budget in [1,64,256]:
            labels=a[f'B{budget}'][train]
            for node in tree.frontier(budget):
                members=tree.members[node]
                if len(np.unique(labels[members]))!=1 or int(np.sum(labels==labels[members[0]]))!=len(members):
                    raise ValueError('Tree reconstruction differs from cached snapshot')
    return tree
