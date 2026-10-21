"""Training-only evidence selection on a fixed, label-free binary region tree."""
from dataclasses import dataclass
import numpy as np


@dataclass
class EvidenceTree:
    members: dict
    children: dict
    parent: dict
    n: int

    @classmethod
    def from_trace(cls, final_assignment, trace):
        final=np.asarray(final_assignment)
        if final.ndim!=1 or not final.size or final.dtype.kind not in 'iu':
            raise ValueError('Invalid final assignment')
        children={};parent={};leaves={0}
        for row in trace:
            if not row['accepted']: continue
            p=int(row['parent_id']); cs=tuple(map(int,row['child_ids']))
            if p not in leaves or len(cs)!=2 or len(set(cs))!=2 or any(c in parent or c==0 for c in cs):
                raise ValueError('Invalid binary split trace')
            children[p]=cs; leaves.remove(p);leaves.update(cs)
            parent.update({c:p for c in cs})
        ids=sorted(leaves)
        if not np.array_equal(np.unique(final),np.arange(len(ids))):
            raise ValueError('Final labels must index sorted terminal tree IDs')
        members={leaf:np.flatnonzero(final==i) for i,leaf in enumerate(ids)}
        def expand(node):
            if node in members:return members[node]
            members[node]=np.sort(np.concatenate([expand(c) for c in children[node]]))
            return members[node]
        expand(0)
        if not np.array_equal(members[0],np.arange(len(final))):raise ValueError('Incomplete tree')
        return cls(members,children,parent,len(final))

    def frontier(self,budget):
        leaves={0}
        for p,cs in self.children.items():
            if len(leaves)>=budget:break
            if p not in leaves:raise ValueError('Split order is not a valid trajectory')
            leaves.remove(p);leaves.update(cs)
        return sorted(leaves)


def optimal_pruning(children,cost):
    """Dynamic program; ties retain the parent, invalid cuts cost infinity."""
    def solve(node):
        keep=([node],float(cost[node]))
        if node not in children:return keep
        left,right=[solve(c) for c in children[node]]
        split=left[1]+right[1]
        return (left[0]+right[0],split) if split<keep[1]-1e-12 else keep
    return solve(0)


def _distribution(labels,classes):
    return np.bincount(labels,minlength=classes).astype(float)/len(labels) if len(labels) else np.full(classes,1/classes)


def _vote(tree,z,classes,query,voters,regions,min_voters,all_votes=False):
    size=len(query);r=np.full((size,classes),1/classes);support=np.zeros(size,dtype=int)
    selected=np.full(size,-1,dtype=int);query_pos=np.full(tree.n,-1,dtype=int);query_pos[query]=np.arange(size)
    voter_mask=np.zeros(tree.n,dtype=bool);voter_mask[voters]=True
    for node in regions:
        m=tree.members[node];local=query_pos[m];local=local[local>=0]
        if not len(local):continue
        actual=node
        while voter_mask[tree.members[actual]].sum()<min_voters and actual in tree.parent:
            actual=tree.parent[actual]
        v=tree.members[actual][voter_mask[tree.members[actual]]]
        if len(v)<min_voters:continue
        if all_votes:
            v=tree.members[actual]
            counts=np.bincount(z[v],minlength=classes)
            support[local]=len(v)-1
            r[local]=(counts-np.eye(classes)[z[query[local]]])/(len(v)-1)
        else:
            support[local]=len(v);r[local]=_distribution(z[v],classes)
        selected[local]=actual
    top=r.max(1);alpha=np.clip((classes*top-1)/(classes-1),0,1);alpha[support<2]=0
    q=r.copy();unique=(np.abs(r-top[:,None])<=1e-12).sum(1)==1
    q[unique]=.5*r[unique]+.5*np.eye(classes)[r[unique].argmax(1)]
    target=(1-alpha[:,None])*np.eye(classes)[z[query]]+alpha[:,None]*q
    return dict(r=r,q=q,alpha=alpha,support=support,region=selected,target=target)


def crossfit_targets(tree,observed,classes,*,seed,min_voters=16,delta=.1):
    """No clean, validation or test labels enter this interface.

    Input order is the original training-index order. Each query label is absent
    from its region selection and independent voting folds.
    """
    z=np.asarray(observed)
    if z.shape!=(tree.n,) or z.dtype.kind not in 'iu' or classes<2 or np.any(z<0) or np.any(z>=classes):
        raise ValueError('Invalid observed labels/classes')
    if not 0<delta<1 or int(min_voters)!=min_voters or min_voters<2:raise ValueError('Invalid evidence settings')
    if tree.n<3:raise ValueError('At least three training positions required')
    folds=np.empty(tree.n,dtype=int)
    for k,positions in enumerate(np.array_split(np.random.default_rng(seed).permutation(tree.n),3)):folds[positions]=k
    arms=['CF_ROOT','CF_64','CF_FINE','PRUNE','PRUNE_ZERO','PRUNE_ALL']
    components={a:{k:np.zeros((2,tree.n,classes) if k in ['r','q','target'] else (2,tree.n),dtype=int if k in ['support','region'] else float)
                   for k in ['r','q','target','alpha','support','region']} for a in arms}
    decisions=[];M=len(tree.members)
    for qfold in range(3):
        query=np.flatnonzero(folds==qfold);other=[k for k in range(3) if k!=qfold]
        for orientation in range(2):
            selector=np.flatnonzero(folds==other[orientation]);voters=np.flatnonzero(folds==other[1-orientation])
            shuffled=np.random.default_rng(seed+1009+qfold*101+orientation).permutation(selector)
            a,d=np.array_split(shuffled,2)
            amask=np.zeros(tree.n,dtype=bool);amask[a]=True
            dmask=np.zeros(tree.n,dtype=bool);dmask[d]=True
            vmask=np.zeros(tree.n,dtype=bool);vmask[voters]=True
            costs={};zero={};score={}
            for node,m in tree.members.items():
                aa=m[amask[m]];dd=m[dmask[m]];vc=int(vmask[m].sum())
                valid=len(aa)>0 and len(dd)>0 and vc>=min_voters
                pred=int(np.bincount(z[aa],minlength=classes).argmax()) if len(aa) else -1
                error=int(np.count_nonzero(z[dd]!=pred)) if pred>=0 else len(dd)
                penalty=float(np.sqrt(len(dd)*np.log(2*M/delta)/2))
                costs[node]=error+penalty if valid else float('inf')
                zero[node]=float(error) if valid else float('inf')
                score[node]=dict(fit_count=len(aa),score_count=len(dd),voter_count=vc,prediction=pred,
                                 errors=error,penalty=penalty,valid=bool(valid))
            regions,value=optimal_pruning(tree.children,costs)
            regions0,value0=optimal_pruning(tree.children,zero)
            # An unscorable root means no selectable partition: _vote returns CE
            # if it also lacks voters; otherwise use the root (no learned cut).
            chosen=dict(CF_ROOT=[0],CF_64=tree.frontier(64),CF_FINE=tree.frontier(256),
                        PRUNE=regions,PRUNE_ZERO=regions0,PRUNE_ALL=regions)
            for arm,cut in chosen.items():
                b=_vote(tree,z,classes,query,voters,cut,min_voters,arm=='PRUNE_ALL')
                for key,array in b.items():components[arm][key][orientation,query]=array
            decisions.append(dict(query_fold=qfold,orientation=orientation,query=query.tolist(),selector=selector.tolist(),
                voters=voters.tolist(),fit=a.tolist(),score=d.tolist(),scores=score,regions=regions,regions_zero=regions0,
                cost=float(value) if np.isfinite(value) else None,cost_zero=float(value0) if np.isfinite(value0) else None))
    bundles={}
    for arm,c in components.items():
        alpha=c['alpha'].mean(0);weighted=(c['alpha'][:,:,None]*c['q']).mean(0)
        q=c['q'].mean(0);active=alpha>0;q[active]=weighted[active]/alpha[active,None]
        bundles[arm]=dict(target=c['target'].mean(0),q=q,alpha=alpha,r=c['r'].mean(0),support=c['support'].mean(0))
    return dict(bundles=bundles,components=components,decisions=decisions,folds=folds)
