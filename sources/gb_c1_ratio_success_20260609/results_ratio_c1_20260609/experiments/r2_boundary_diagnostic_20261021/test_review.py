import unittest

import finish_review as f


class OptionalArmTests(unittest.TestCase):
    def test_optional_checker_exists(self):
        self.assertTrue(callable(getattr(f,'check_optional_arm',None)))

    def test_blank_is_not_zero_or_available_result(self):
        self.assertFalse(f.check_optional_arm('',[]))

    def test_partial_seed_set_is_an_error(self):
        with self.assertRaises(AssertionError):
            f.check_optional_arm('50',[50,51])

    def test_present_three_seed_mean_verified(self):
        self.assertTrue(f.check_optional_arm('50',[49,50,51]))


if __name__=='__main__':
    unittest.main(verbosity=2)
