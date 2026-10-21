import unittest
import pandas as pd
from validation_budget_diagnostic import select_by_validation


class ValidationBudgetTests(unittest.TestCase):
    def test_test_scores_cannot_choose_the_budget(self):
        f=pd.DataFrame([dict(dataset='toy',seed=0,noise='uniform',space='GREET',kind='B',arm='GREET_B'+str(b),
                            requested_balls=b,actual_balls=b,validation_accuracy=v,test_accuracy=t)
                        for b,v,t in [(1,.7,.99),(32,.8,.1),(64,.8,.9)]])
        first=select_by_validation(f)
        self.assertEqual(first.iloc[0].arm,'GREET_B32')
        f['test_accuracy']=[0,.99,0]
        pd.testing.assert_frame_equal(first,select_by_validation(f))

    def test_capped_aliases_do_not_break_coarse_tie_rule(self):
        f=pd.DataFrame([dict(dataset='toy',seed=0,noise='pair',space='GRACE',kind='B',arm='GRACE_B'+str(b),
                            requested_balls=b,actual_balls=20,validation_accuracy=.6) for b in [128,64]])
        self.assertEqual(select_by_validation(f).iloc[0].requested_balls,64)


if __name__=='__main__':unittest.main()
