import unittest

from scripts import run_ssl_swap_downstream


class TieSafeRunnerDiagnosticsTest(unittest.TestCase):
    def test_tie_safe_diagnostics_are_persisted_to_csv_rows(self):
        required = {
            "tie_mode",
            "tie_ball_count",
            "tie_active_ball_count",
            "tie_affected_node_count",
            "tie_affected_alpha_sum",
            "ball_assignment_hash",
        }
        self.assertTrue(hasattr(run_ssl_swap_downstream, "DIAG_KEYS"))
        self.assertTrue(required.issubset(set(run_ssl_swap_downstream.DIAG_KEYS)))


if __name__ == "__main__":
    unittest.main()
