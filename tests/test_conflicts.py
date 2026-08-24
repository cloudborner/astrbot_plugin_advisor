import unittest

from advisor.conflicts import detect_capacity_conflicts
from advisor.models import ResourceProfile, ServerProfile


def profile(plugin_id, *, memory=1, background="no", processes=None):
    scores = {
        "idle_memory": 0,
        "peak_memory": memory,
        "idle_cpu": 0,
        "peak_cpu": 1,
        "disk": 1,
        "network": 1,
    }
    return ResourceProfile(
        plugin_id=plugin_id,
        version="1",
        commit_sha="abc",
        levels={key: f"L{value}" for key, value in scores.items()},
        scores=scores,
        features=[],
        external_processes=processes or [],
        background_tasks=background,
        evidence=[],
        unknowns=[],
        confidence=0.7,
        evidence_level="test",
        scanned_at="2026-08-23T00:00:00+00:00",
    )


class ConflictTests(unittest.TestCase):
    def test_reports_oom_stacking_on_small_server(self):
        server = ServerProfile(1843, 700, 1024, 700, 2, 10_000)
        warnings = detect_capacity_conflicts(
            profile("a/new", memory=4), [profile("a/old", memory=3)], server
        )
        self.assertTrue(any("OOM" in item for item in warnings))

    def test_shared_lightweight_dependency_is_not_called_conflict(self):
        server = ServerProfile(8192, 6000, 1024, 1024, 4, 10_000)
        warnings = detect_capacity_conflicts(
            profile("a/new", processes=["redis"]),
            [profile("a/old", processes=["redis"])],
            server,
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
