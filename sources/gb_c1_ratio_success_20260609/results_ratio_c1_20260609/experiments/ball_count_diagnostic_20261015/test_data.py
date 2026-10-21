import tempfile
import unittest
from pathlib import Path
import numpy as np
from data_tools import load_new_graph


class DataTests(unittest.TestCase):
    def test_official_order_and_masks_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tiny.npz'
            x=np.arange(20,dtype=np.float32).reshape(5,4)
            masks=[np.array([[1,1,0,0,0]],dtype=bool),np.array([[0,0,1,0,0]],dtype=bool),np.array([[0,0,0,1,1]],dtype=bool)]
            np.savez(path,node_features=x,node_labels=np.array([0,1,0,1,0]),edges=np.array([[0,1],[1,2]]),
                     train_masks=masks[0],val_masks=masks[1],test_masks=masks[2])
            graph=load_new_graph(path,'tiny',0)
            np.testing.assert_array_equal(graph.x.numpy(),x)
            np.testing.assert_array_equal(graph.train_index.numpy(),[0,1])
            self.assertEqual(set(map(tuple,graph.edge_index.T.tolist())),{(0,1),(1,0),(1,2),(2,1)})

    def test_overlapping_official_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.npz'
            m=np.ones((1,4),dtype=bool)
            np.savez(path,node_features=np.ones((4,3)),node_labels=np.array([0,1,0,1]),
                     edges=np.array([[0,1]]),train_masks=m,val_masks=m,test_masks=m)
            with self.assertRaises(ValueError):load_new_graph(path,'bad',0)


if __name__=='__main__':unittest.main()
