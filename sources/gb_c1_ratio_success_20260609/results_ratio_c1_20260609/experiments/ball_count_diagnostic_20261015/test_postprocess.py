import unittest
import numpy as np
from finish_diagnostic import compare_csv_metric


class CsvAuditTests(unittest.TestCase):
    def test_numeric_strings_and_missing_values_are_compared_numerically(self):
        compare_csv_metric('0.2431469298245614',.2431469298245614,'margin')
        compare_csv_metric('',None,'missing')
        compare_csv_metric(np.nan,None,'missing')
        with self.assertRaises(ValueError):compare_csv_metric('0.3',.4,'mismatch')
        with self.assertRaises(ValueError):compare_csv_metric('not-a-number',.4,'bad')


if __name__=='__main__':unittest.main()
