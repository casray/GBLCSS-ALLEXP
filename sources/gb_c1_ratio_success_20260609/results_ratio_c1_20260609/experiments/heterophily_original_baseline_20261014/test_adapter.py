import numpy as np
import pytest
from run_original_baseline import load_npz_graph


def test_official_npz_masks_and_row_order(tmp_path):
    file=tmp_path/'graph.npz'
    x=np.arange(12,dtype=np.float32).reshape(4,3)
    masks=np.array([[1,0,1,0],[0,1,0,1]],dtype=bool)
    np.savez(file,node_features=x,node_labels=np.array([0,1,1,0]),
        edges=np.array([[0,1],[1,2]]),train_masks=masks,
        val_masks=np.array([[0,1,0,0],[1,0,0,0]],dtype=bool),
        test_masks=np.array([[0,0,0,1],[0,0,1,0]],dtype=bool))
    graph=load_npz_graph(file,'toy',1)
    np.testing.assert_array_equal(graph.x.numpy(),x)
    np.testing.assert_array_equal(graph.train_index.numpy(),[1,3])
    assert graph.edge_index.shape==(2,4)


def test_overlapping_split_rejected(tmp_path):
    file=tmp_path/'bad.npz'
    np.savez(file,node_features=np.ones((2,3),dtype=np.float32),node_labels=np.array([0,1]),
        edges=np.array([[0,1]]),train_masks=np.array([[1,0]],dtype=bool),
        val_masks=np.array([[1,0]],dtype=bool),test_masks=np.array([[0,1]],dtype=bool))
    with pytest.raises(ValueError,match='disjoint'):
        load_npz_graph(file,'toy',0)
