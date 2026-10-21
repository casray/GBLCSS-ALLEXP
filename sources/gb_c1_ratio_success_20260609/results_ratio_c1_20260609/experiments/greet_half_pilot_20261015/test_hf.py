import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hf_proxy import filtered_features, cross_fit_scores


class HFTests(unittest.TestCase):
    def test_features_match_dense_filter_and_isolate_h(self):
        x=np.array([[1.,0.],[1.,1.],[0.,1.],[0.,0.]],dtype=np.float32)
        edges=np.array([[0,1,1,2],[1,0,2,1]])
        low,high,h=filtered_features(x,edges)
        adj=np.eye(4,dtype=np.float32); adj[edges[0],edges[1]]=1
        inv=adj.sum(1)**-.5
        expected=(adj*inv[:,None]*inv[None,:])@x
        np.testing.assert_allclose(low,expected,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(high,x-low)
        np.testing.assert_allclose(h[:3],1/np.sqrt(2),rtol=1e-6)
        self.assertEqual(h[3],.5)

    def test_crossfit_never_receives_holdout_labels(self):
        features={k:np.ones((15,3),dtype=np.float32) for k in ['attribute','low','high']}
        tr=np.array([1,3,5,7,9,11]); observed=np.array([0,1,0,1,0,1])
        calls=[]
        def fake(features,fit_indices,fit_labels,hold_indices,n_classes,config,seed,folder,device):
            self.assertEqual(len(set(fit_indices)&set(hold_indices)),0)
            lookup=dict(zip(tr,observed))
            np.testing.assert_array_equal(fit_labels,[lookup[i] for i in fit_indices])
            calls.append((fit_indices.copy(),hold_indices.copy()))
            return {epoch:np.zeros((len(hold_indices),n_classes)) for epoch in config['saved_epochs']}
        with tempfile.TemporaryDirectory() as directory, patch('hf_proxy.fit_channel',side_effect=fake):
            losses,folds=cross_fit_scores(features,observed,tr,2,
                dict(folds=3,epochs=30,saved_epochs=[10,20,30]),0,Path(directory),'cpu')
        self.assertEqual(len(calls),9)
        for channel in losses:
            for scores in losses[channel].values():np.testing.assert_allclose(scores,np.log(2))
        self.assertEqual(sorted(np.concatenate(folds).tolist()),list(range(6)))


if __name__=='__main__':unittest.main()
