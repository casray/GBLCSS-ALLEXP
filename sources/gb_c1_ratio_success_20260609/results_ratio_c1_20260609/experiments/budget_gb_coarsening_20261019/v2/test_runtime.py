import unittest

from runtime import Checkpoints, validation_due


class RuntimeTest(unittest.TestCase):
    def test_original_and_lift_checkpoints_are_separate(self):
        bank = Checkpoints(patience_checks=2)
        bank.observe(5, 0.9, 0.5, {"value": [5]})
        bank.observe(10, 0.8, 0.95, {"value": [10]})
        self.assertEqual(bank.primary_epoch, 5)
        self.assertEqual(bank.secondary_epoch, 10)
        self.assertEqual(bank.primary_state["value"], [5])
        self.assertEqual(bank.secondary_state["value"], [10])
        self.assertFalse(bank.stop)
        bank.observe(15, 0.85, 0.99, {"value": [15]})
        self.assertTrue(bank.stop)
        self.assertEqual(bank.primary_epoch, 5)

    def test_snapshot_is_not_changed_by_later_mutation(self):
        bank = Checkpoints(patience_checks=2)
        state = {"value": [5]}
        bank.observe(5, 0.8, None, state)
        state["value"][0] = 100
        self.assertEqual(bank.primary_state["value"], [5])

    def test_validation_schedule_includes_final_partial_block(self):
        self.assertEqual([i for i in range(1, 13) if validation_due(i, 12, 5)], [5, 10, 12])


if __name__ == "__main__":
    unittest.main()
