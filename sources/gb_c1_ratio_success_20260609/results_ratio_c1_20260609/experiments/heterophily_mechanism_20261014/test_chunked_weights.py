"""Regression: bounded pair expansion preserves the frozen diagnostic formula."""
import numpy as np
import diagnostic
import diagnostic_v2


def test_weight_chunks_are_bit_identical_to_original():
    rng=np.random.RandomState(59)
    h=rng.normal(size=(29,512)).astype(np.float32)
    h/=np.linalg.norm(h,axis=1,keepdims=True)
    groups=np.r_[np.zeros(25,dtype=int),[1,1,2,3]]
    expected=diagnostic.group_weights(groups,h)
    for chunk in (1,7,31,16384):
        actual=diagnostic_v2.group_weights(groups,h,pair_chunk=chunk)
        np.testing.assert_array_equal(actual.indptr,expected.indptr)
        np.testing.assert_array_equal(actual.indices,expected.indices)
        np.testing.assert_array_equal(actual.data,expected.data)
