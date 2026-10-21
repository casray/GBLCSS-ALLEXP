from pathlib import Path
import unittest


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parent/'selection.py').exists(), 'selector not implemented')
        from selection import choose
        self.choose = choose

    def test_validation_only(self):
        rows = [dict(valid_correct=8, valid_total=10, test_correct=1),
                dict(valid_correct=7, valid_total=10, test_correct=10)]
        self.assertIs(self.choose(rows), rows[0])
        rows[0]['test_correct'], rows[1]['test_correct'] = 0, 100000
        self.assertIs(self.choose(rows), rows[0])

    def test_ties_use_order(self):
        rows = [dict(valid_correct=8, valid_total=10, name='first'),
                dict(valid_correct=8, valid_total=10, name='second')]
        self.assertEqual(self.choose(rows)['name'], 'first')

    def test_mismatched_split_is_rejected(self):
        with self.assertRaises(ValueError):
            self.choose([dict(valid_correct=8, valid_total=10), dict(valid_correct=8, valid_total=11)])


if __name__ == '__main__':
    unittest.main()
