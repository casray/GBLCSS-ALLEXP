"""Independent scalar voting, scoring and cut optimality checks."""
import numpy as np


def reference_vote(tree,z,classes,query,voters,regions,min_voters,all_votes):
    voter_set=set(map(int,voters)); rows=[];mapping={};cached={}
    for node in regions:
        for index in tree.members[node]:
            assert int(index) not in mapping,'Overlapping selected regions'
            mapping[int(index)]=node
    def evidence(node):
        if node not in cached:
            v=sorted(voter_set.intersection(tree.members[node].tolist()))
            used=tree.members[node] if all_votes else np.asarray(v,dtype=int)
            cached[node]=(len(v),len(used),np.bincount(z[used],minlength=classes).astype(float))
        return cached[node]
    for index in query:
        assert int(index) in mapping,'Selected partition does not cover a query'
        node=mapping[int(index)]
        while evidence(node)[0]<min_voters and node in tree.parent:
            node=tree.parent[node]
        count,total,hist=evidence(node)
        r=np.full(classes,1/classes);support=0;region=-1
        if count>=min_voters:
            support=total-1 if all_votes else total
            r=(hist-np.eye(classes)[z[index]])/support if all_votes else hist/support
            region=node
        alpha=max(0.,min(1.,(classes*float(max(r))-1)/(classes-1))) if support>=2 else 0.
        q=r.copy();winners=np.flatnonzero(np.abs(r-r.max())<=1e-12)
        if len(winners)==1:q=.5*r+.5*np.eye(classes)[winners[0]]
        t=(1-alpha)*np.eye(classes)[z[index]]+alpha*q
        rows.append(dict(r=r,q=q,alpha=alpha,support=support,region=region,target=t))
    return {key:np.asarray([row[key] for row in rows]) for key in rows[0]}


def verify_output(tree,z,classes,out,*,min_voters,delta):
    checked=0;M=len(tree.members)
    for decision in out['decisions']:
        q,a,d,v,s=[np.array(decision[k],dtype=int) for k in ['query','fit','score','voters','selector']]
        assert len(set(q)&set(s))==len(set(q)&set(v))==len(set(s)&set(v))==0
        assert sorted(np.r_[q,s,v].tolist())==list(range(tree.n))
        assert sorted(np.r_[a,d].tolist())==sorted(s.tolist()) and not set(a)&set(d)
        costs={};zero={}
        for node,m in tree.members.items():
            aa=sorted(set(a)&set(m));dd=sorted(set(d)&set(m));vv=sorted(set(v)&set(m))
            pred=int(np.argmax([sum(z[j]==c for j in aa) for c in range(classes)])) if aa else -1
            errors=sum(z[j]!=pred for j in dd)
            valid=bool(aa and dd and len(vv)>=min_voters)
            penalty=float(np.sqrt(len(dd)*np.log(2*M/delta)/2))
            recorded=decision['scores'][node] if node in decision['scores'] else decision['scores'][str(node)]
            for k,value in dict(fit_count=len(aa),score_count=len(dd),voter_count=len(vv),prediction=pred,errors=int(errors),valid=valid).items():
                assert recorded[k]==value,(node,k)
            assert abs(recorded['penalty']-penalty)<1e-12
            costs[node]=errors+penalty if valid else np.inf;zero[node]=float(errors) if valid else np.inf
        def solve(node,cost):
            value=cost[node];cut=[node]
            if node in tree.children:
                sub=[solve(c,cost) for c in tree.children[node]]
                if sum(x[0] for x in sub)<value-1e-12:value=sum(x[0] for x in sub);cut=sum((x[1] for x in sub),[])
            return value,cut
        for costs_,key in [(costs,'regions'),(zero,'regions_zero')]:
            assert solve(0,costs_)[1]==decision[key],'Cut is not optimal under recorded criterion'
        regions=dict(CF_ROOT=[0],CF_64=tree.frontier(64),CF_FINE=tree.frontier(256),
                     PRUNE=decision['regions'],PRUNE_ZERO=decision['regions_zero'],PRUNE_ALL=decision['regions'])
        orientation=decision['orientation']
        for arm,cut in regions.items():
            ref=reference_vote(tree,z,classes,q,v,cut,min_voters,arm=='PRUNE_ALL')
            for key,values in ref.items():
                np.testing.assert_allclose(out['components'][arm][key][orientation,q],values,atol=2e-12,rtol=0)
            checked+=1
    for arm,c in out['components'].items():
        b=out['bundles'][arm]
        np.testing.assert_allclose(b['target'],c['target'].mean(0),atol=2e-12,rtol=0)
        np.testing.assert_allclose(b['target'],(1-b['alpha'][:,None])*np.eye(classes)[z]+b['alpha'][:,None]*b['q'],atol=2e-12,rtol=0)
        np.testing.assert_allclose(b['alpha'],c['alpha'].mean(0),atol=2e-12,rtol=0)
        np.testing.assert_allclose(b['support'],c['support'].mean(0),atol=2e-12,rtol=0)
        np.testing.assert_allclose(b['r'],c['r'].mean(0),atol=2e-12,rtol=0)
        np.testing.assert_allclose(b['target'].sum(1),1.,atol=2e-12,rtol=0)
        assert np.min(b['target'])>=-2e-12
    return dict(status='passed',components_checked=checked)
