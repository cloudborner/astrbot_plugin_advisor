import importlib.util
import unittest
from pathlib import Path

from advisor.market import GitHubObservation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scan_market.py"
SPEC = importlib.util.spec_from_file_location("scan_market", SCRIPT)
scan_market = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan_market)


class ScanBudgetTests(unittest.TestCase):
    def test_empty_cache_full_market_fits_classic_pat_mandatory_phase(self):
        pending = [(str(index), None) for index in range(1834)]
        selected, deferred = scan_market.select_checkpoint_batch(
            pending, remaining=5000
        )
        self.assertEqual(len(selected), 1834)
        self.assertEqual(deferred, 0)

    def test_low_quota_saves_deterministic_partial_checkpoint_only(self):
        pending = [(str(index), None) for index in range(100)]
        selected, deferred = scan_market.select_checkpoint_batch(
            pending, remaining=150, reserve=50
        )
        self.assertEqual(len(selected), 50)
        self.assertEqual(deferred, 50)
        self.assertEqual(selected[0][0], "0")
        self.assertEqual(selected[-1][0], "49")

    def test_cached_tree_and_commit_cost_zero(self):
        complete = GitHubObservation(
            "a" * 40,
            [],
            [],
            True,
            False,
            [],
            commit_ok=True,
            commit_api="list_commits_metadata",
        )
        pending = [("cached", complete), ("new", None)]
        selected, deferred = scan_market.select_checkpoint_batch(
            pending, remaining=50, reserve=50
        )
        self.assertEqual([item[0] for item in selected], ["cached"])
        self.assertEqual(deferred, 1)


if __name__ == "__main__":
    unittest.main()
