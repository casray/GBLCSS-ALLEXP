import unittest
import torch
import analyze_fits


class PosthocTests(unittest.TestCase):
    def test_selector_overrides_arm_without_mutating_source(self):
        original=dict(arm='S32',accuracy=.5)
        selected=analyze_fits.selector_row(original,'SC',32)
        self.assertEqual(selected,dict(arm='SC',accuracy=.5,selected_support=32))
        self.assertEqual(original,dict(arm='S32',accuracy=.5))

    def test_macro_f1(self):
        y=torch.tensor([0,0,1,1])
        p=torch.tensor([0,1,1,1])
        result=analyze_fits.metrics(y,p,2)
        self.assertEqual(result['accuracy'],.75)
        self.assertAlmostEqual(result['macro_f1'],(2/3+4/5)/2)


if __name__=='__main__':
    unittest.main()
