from pathlib import Path
import tempfile
import unittest
import numpy as np
import scipy.sparse as sp
import torch
from experiment import file_hash
from prepare_data import load_verified_edges


class PrepareTests(unittest.TestCase):
    def test_raw_features_labels_and_propagation_match_original_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'raw.pt'
            x=np.array([[1.,2.],[3.,4.],[5.,6.]],dtype=np.float32)
            y=np.array([0,1,0],dtype=np.int64)
            edges=np.array([[0,1,1,2],[1,0,2,1]],dtype=np.int64)
            torch.save((dict(x=torch.from_numpy(x),y=torch.from_numpy(y),edge_index=torch.from_numpy(edges)),{}),path)
            a=sp.csr_matrix(np.array([[1,1,0],[1,1,1],[0,1,1]],dtype=np.float32))
            inv=sp.diags(1/np.sqrt(np.asarray(a.sum(1)).ravel()));p=inv@a@inv
            np.testing.assert_array_equal(load_verified_edges(path,file_hash(path),x,y,p),edges)
            with self.assertRaisesRegex(ValueError,'features'):
                load_verified_edges(path,file_hash(path),x+1,y,p)
            with self.assertRaisesRegex(ValueError,'propagation'):
                load_verified_edges(path,file_hash(path),x,y,sp.eye(3))
            with self.assertRaisesRegex(ValueError,'source'):
                load_verified_edges(path,'wrong',x,y,p)


if __name__=='__main__':unittest.main()
