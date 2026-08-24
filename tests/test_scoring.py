import unittest
from datetime import UTC, datetime

from advisor.models import PluginRecord, ResourceProfile, ServerProfile
from advisor.scoring import ScoreEngine


def plugin(plugin_id, downloads, stars, *, platforms=None, astrbot_version=""):
    author, name = plugin_id.split("/", 1)
    return PluginRecord(
        plugin_id=plugin_id,
        author=author,
        name=name,
        version="1.0.0",
        repo=f"https://github.com/{plugin_id}",
        desc="视频下载",
        download_count=downloads,
        stars=stars,
        updated_at="2026-08-01T00:00:00+00:00",
        support_platforms=platforms or [],
        astrbot_version=astrbot_version,
    )


def profile(
    plugin_id,
    *,
    peak_memory=1,
    peak_cpu=1,
    idle_cpu=0,
    disk=1,
    network=1,
    confidence=0.7,
):
    levels = {
        "idle_memory": "L0",
        "peak_memory": f"L{peak_memory}",
        "idle_cpu": f"L{idle_cpu}",
        "peak_cpu": f"L{peak_cpu}",
        "disk": f"L{disk}",
        "network": f"L{network}",
    }
    scores = {key: int(value[1]) for key, value in levels.items()}
    return ResourceProfile(
        plugin_id=plugin_id,
        version="1.0.0",
        commit_sha="abc",
        levels=levels,
        scores=scores,
        features=[],
        external_processes=[],
        background_tasks="no",
        evidence=[],
        unknowns=[],
        confidence=confidence,
        evidence_level="test",
        scanned_at="2026-08-01T00:00:00+00:00",
    )


SERVER = ServerProfile(2048, 900, 1024, 700, 2.0, 10_000, "aiocqhttp", "4.5.7")


class ScoringTests(unittest.TestCase):
    def test_market_usage_is_13_download_plus_7_stars(self):
        low = plugin("a/low", 0, 0)
        mid = plugin("a/mid", 10, 10)
        high = plugin("a/high", 1000, 100)
        engine = ScoreEngine([low, mid, high], now=datetime(2026, 8, 23, tzinfo=UTC))
        self.assertEqual(engine._market_score(low), 0.0)
        self.assertEqual(engine._market_score(high), 20.0)
        self.assertAlmostEqual(engine._market_score(mid), 10.0)

    def test_percentile_counts_duplicate_plugins(self):
        records = [plugin(f"a/low{i}", 0, 0) for i in range(8)]
        high = plugin("a/high", 100, 100)
        records.append(high)
        engine = ScoreEngine(records)
        # The high plugin is at the empirical 100th percentile, even though
        # most plugins share one low value.
        self.assertEqual(engine._market_score(high), 20.0)
        self.assertEqual(engine._market_score(records[0]), 0.0)

    def test_total_equals_fixed_components(self):
        item = plugin("a/test", 10, 10)
        engine = ScoreEngine([item], now=datetime(2026, 8, 23, tzinfo=UTC))
        result = engine.score(item, profile(item.plugin_id), SERVER, {"download": 10})
        expected = (
            result.demand
            + result.market_usage
            + result.compatibility
            + result.resource_fit
            + result.maintenance
            + result.deployment
        )
        self.assertAlmostEqual(result.total, expected)

    def test_incompatible_platform_caps_recommendation(self):
        item = plugin("a/test", 1000, 100, platforms=["telegram"])
        result = ScoreEngine([item]).score(
            item, profile(item.plugin_id), SERVER, {"download": 10}
        )
        self.assertLessEqual(result.total, 39.0)
        self.assertEqual(result.compatibility, 0.0)

    def test_incompatible_astrbot_version_caps_recommendation(self):
        item = plugin("a/test", 1000, 100, astrbot_version=">=5")
        result = ScoreEngine([item]).score(
            item, profile(item.plugin_id), SERVER, {"download": 10}
        )
        self.assertLessEqual(result.total, 39.0)
        self.assertEqual(result.compatibility, 0.0)

    def test_compatible_astrbot_version_gets_full_compatibility(self):
        item = plugin(
            "a/test",
            1,
            1,
            platforms=["aiocqhttp"],
            astrbot_version=">=4.5,<5",
        )
        result = ScoreEngine([item]).score(
            item, profile(item.plugin_id), SERVER, {"download": 10}
        )
        self.assertEqual(result.compatibility, 20.0)

    def test_heavy_profile_scores_lower_on_small_server(self):
        item = plugin("a/test", 10, 10)
        engine = ScoreEngine([item])
        light = engine.score(item, profile(item.plugin_id), SERVER, {"download": 10})
        heavy = engine.score(
            item,
            profile(item.plugin_id, peak_memory=4, peak_cpu=4),
            SERVER,
            {"download": 10},
        )
        self.assertLess(heavy.resource_fit, light.resource_fit)

    def test_disk_network_and_uncertainty_reduce_resource_fit(self):
        item = plugin("a/test", 10, 10)
        engine = ScoreEngine([item])
        light = engine.score(item, profile(item.plugin_id), SERVER, {"download": 10})
        risky = engine.score(
            item,
            profile(item.plugin_id, disk=4, network=4, idle_cpu=3, confidence=0.28),
            SERVER,
            {"download": 10},
        )
        self.assertLess(risky.resource_fit, light.resource_fit)
        self.assertTrue(any("证据不足" in item for item in risky.warnings))

    def test_topic_match_can_supply_full_30_point_demand_score(self):
        item = plugin("a/robomaster", 10, 10)
        result = ScoreEngine([item]).score(
            item,
            profile(item.plugin_id),
            SERVER,
            {},
            topic_match_strength=1.0,
            matched_topics=["RoboMaster"],
        )
        self.assertEqual(result.demand, 30.0)
        self.assertTrue(any("RoboMaster" in reason for reason in result.reasons))

    def test_short_ai_keyword_does_not_match_email_or_daily(self):
        item = plugin("a/mail", 10, 10)
        item.desc = "email daily waiting farm formatter"
        item.short_desc = "mail tools"
        score, reasons = ScoreEngine._demand_score(item, {"ai": 10})
        self.assertEqual(score, 3.0)
        self.assertEqual(reasons, ["未匹配到当前群聊的主要需求"])


if __name__ == "__main__":
    unittest.main()
