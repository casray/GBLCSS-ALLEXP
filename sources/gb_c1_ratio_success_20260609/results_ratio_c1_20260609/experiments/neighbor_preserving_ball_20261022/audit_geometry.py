"""Read-only replication and independent distance/cut checks; no target outcomes."""
import numpy as np
import torch
from scipy.spatial.distance import cdist

from run import ROOT, CONFIG, load_dataset, read, write, verify, sha, g
from gblcss.data import percent_split_indices


def main():
    verify()
    rows = []
    for name in CONFIG['datasets']:
        data = load_dataset(name, 'cpu')
        geo = torch.load(ROOT/'inputs'/name/'geometry.pt', weights_only=True)
        saved = torch.load(ROOT/'inputs'/name/'tree.pt', weights_only=True)
        train = geo['train'].numpy()
        near = geo['near'].numpy()
        h = g.normalize(data['h'].numpy())
        rng = np.random.default_rng(101)
        sample = rng.choice(len(train), min(32,len(train)), replace=False)
        dist = cdist(h[train[sample]], h[train], metric='sqeuclidean')
        exact = 0
        for pos, query in enumerate(sample):
            dist[pos,query] = np.inf
            expected = np.lexsort((train,dist[pos]))[:-1]
            assert query not in near[query]
            assert np.array_equal(np.sort(near[query]),np.delete(np.arange(len(train)),query))
            if np.array_equal(expected,near[query]):
                exact += 1
            else:
                # Independent summation may reorder only numerically tied distances.
                assert np.allclose(dist[pos,expected],dist[pos,near[query]],rtol=0,atol=1e-12)
        tree = g.build_tree(h, train[::-1], CONFIG['max_leaves'], CONFIG['snapshot_leaves'],
                            CONFIG['min_nodes'],CONFIG['min_train'],seed=0)
        assert len(tree['nodes']) == len(saved['nodes'])
        for a,b in zip(tree['nodes'],saved['nodes']):
            assert np.array_equal(a['members'],b['members'].numpy())
            assert a['children'] == b['children'] and a['train_count']==b['train_count']
        for arm,key in [('TREE64','snapshot_assignment'),('FINE','fine_assignment')]:
            assert np.array_equal(tree[key],geo['partitions'][arm].numpy())
        costs = g.node_costs(tree['nodes'],train,near[:,:geo['k']])
        assert np.array_equal(costs,saved['costs'].numpy())
        chosen,value = g.select_cut(tree['nodes'],costs)
        assert chosen==saved['np_leaves'] and value==saved['cost']
        assert np.array_equal(g.assignment(tree['nodes'],chosen,len(h)),geo['partitions']['NP'].numpy())
        ids = np.r_[geo['train'],geo['valid'],geo['test']]
        assert len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<len(h)
        for key, reference in zip(('train','valid','test'),percent_split_indices(len(h),0)):
            assert np.array_equal(geo[key].numpy(),reference)
        rows.append(dict(dataset=name,nodes=len(h),training=len(train),distance_checks=len(sample),
                         exact_distance_order=exact,tree_reconstruction=True,train_order_invariant=True,
                         partition_coverage=True,dp_reconstruction=True,ordered_split_match=True,
                         outside_evaluation_splits=len(h)-len(ids)))
        print('GEOMETRY AUDIT '+name+' passed',flush=True)
    verify()
    write(ROOT/'geometry_audit.json',dict(passed=True,checks=rows,source_sha256=sha(__file__)))


if __name__=='__main__':
    main()
