import unittest
import numpy as np
from run import target_key,arm_metadata


class ExecutionTests(unittest.TestCase):
    def test_alias_identity_matches_actual_float32_training_values(self):
        a=np.array([[.4,.6],[.2,.8]],dtype=np.float64)
        self.assertEqual(target_key(a),target_key(a.astype(np.float32)))
        b=a.copy();b[0]=[.41,.59]
        self.assertNotEqual(target_key(a),target_key(b))

    def test_ce_old_and_budget_arms_are_unambiguous(self):
        self.assertEqual(arm_metadata('GREET_B128'),dict(space='GREET',kind='B',requested_balls=128))
        self.assertEqual(arm_metadata('GRACE_OLD64')['kind'],'OLD64')
        self.assertEqual(arm_metadata('CE')['kind'],'CE')
        with self.assertRaises(ValueError):arm_metadata('GREET_KNN_MATCH')


if __name__=='__main__':unittest.main()
