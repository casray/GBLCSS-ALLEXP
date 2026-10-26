import unittest
import numpy as np

from routing_kernels import choose_weight, choose_threshold, student_features, region_features


class RoutingTests(unittest.TestCase):
    def test_global_weight_tie_prefers_default(self):
        p=np.array([[.8,.2],[.2,.8]])
        self.assertEqual(choose_weight(np.array([0,1]),p,p),1.0)

    def test_soft_average_can_correct_two_wrong_argmaxes(self):
        a=np.array([[.5,.1,.4]])
        c=np.array([[.1,.5,.4]])
        self.assertEqual(np.argmax((a+c)/2,axis=1)[0],2)

    def test_threshold_preserves_default_on_no_net_gain(self):
        a=np.array([0,1]);c=np.array([1,0]);y=np.array([0,0])
        self.assertTrue(np.isinf(choose_threshold(y,a,c,np.array([.7,.7]))))

    def test_features_have_no_label_input_and_are_class_permutation_invariant(self):
        a=np.array([[.1,.2,.7],[.3,.6,.1]])
        c=np.array([[.4,.3,.3],[.7,.1,.2]])
        np.testing.assert_allclose(student_features(a,c),student_features(a[:,[2,0,1]],c[:,[2,0,1]]))

    def test_region_statistics_exclude_query_training_label(self):
        assignment=np.array([0,0,0,1]);teacher=np.array([0,1,0,1]);c=np.array([0,0,0,1]);train=np.array([0,1]);z=np.array([0,1])
        first=region_features(assignment,teacher,c,train,z)
        changed=region_features(assignment,teacher,c,train,np.array([1,1]))
        np.testing.assert_allclose(first[0],changed[0])
        self.assertNotEqual(first[1,1],changed[1,1])
        self.assertEqual(first[3,3],1)


if __name__=='__main__':
    unittest.main()
