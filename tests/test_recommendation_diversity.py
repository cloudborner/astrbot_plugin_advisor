from dataclasses import replace

import pytest

from advisor.analysis_checkpoint import report_from_payload, report_to_payload
from advisor.config import parse_config
from advisor.recommendation_diversity import select_diverse_recommendations
from advisor.reports import analysis_report_text, render_analysis_report_html
from tests.test_analysis_checkpoint import _report


def review(*caps, status="supported"):
    return {"capability_checks": [
        {"need_index": 1, "capability_index": c, "status": status} for c in caps
    ]}


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_limit_includes_main_choice_and_groups_before_final_cutoff(limit):
    ids = [f"parser/{i}" for i in range(6)] + ["summary/new"]
    reviews = {pid: review(1) for pid in ids}
    reviews[ids[-1]] = review(1, 2)
    kept, folded = select_diverse_recommendations(ids, reviews, tolerance=limit)
    assert kept == ids[:limit] + [ids[-1]]
    assert folded == {ids[0]: ids[limit:6]}
    assert ids[-1] in kept[:limit + 1]  # complementary candidate fills freed slot


def test_unknown_partial_missing_proof_and_names_do_not_prove_duplicates():
    reviews = {"full": review(1), "partial": review(1, status="partial"),
               "unknown": review(1, status="unknown"), "legacy": {}}
    ids = list(reviews)
    assert select_diverse_recommendations(ids, reviews, tolerance=1) == (ids, {})
    assert select_diverse_recommendations(["a", "b"], {"a": review(1), "b": review(1)},
                                         tolerance=1, capability_counts={1: 2}) == (["a", "b"], {})


def test_disabled_keeps_order_and_does_not_fold():
    ids = ["b", "a"]
    assert select_diverse_recommendations(ids, {p: review(1) for p in ids}, enabled=False) == (ids, {})


def test_local_mode_uses_confident_matching_profiles_only():
    ids = ["a", "b", "c", "d"]
    reviews = {p: {"functional_fit": 0.9, "matched_need_titles": ["视频"], "risks": []} for p in ids}
    profiles = {p: {"capabilities": ["链接解析"], "confidence": 0.9} for p in ids}
    profiles["c"] = {**profiles["c"], "capabilities": ["视频摘要"]}
    profiles["d"] = {**profiles["d"], "confidence": 0.4}
    assert select_diverse_recommendations(ids, reviews, tolerance=1, profiles=profiles) == (["a", "c", "d"], {"a": ["b"]})


def test_config_defaults_bounds_and_off():
    assert parse_config({}).deduplicate_similar_functions is True
    assert parse_config({}).similar_function_limit == 3
    assert parse_config({"general": {"similar_function_limit": 0}}).similar_function_limit == 1
    assert parse_config({"general": {"similar_function_limit": 99}}).similar_function_limit == 20
    assert parse_config({"general": {"similar_function_limit": "bad"}}).similar_function_limit == 3
    assert not parse_config({"general": {"deduplicate_similar_functions": False}}).deduplicate_similar_functions


def grouped_report():
    data = _report()
    card = replace(data.recommendations[0], plugin_id="demo/parser", similar_count=2,
                   similar_plugins=("同功能甲<script>", "同功能乙"),
                   reason="第一行\n第二行；依据（标题；简介）")
    return replace(data, recommendations=(card,))


def test_group_and_newlines_survive_checkpoint_and_escape_html():
    data = grouped_report()
    restored = report_from_payload(report_to_payload(data), group_label="测试")
    assert restored.recommendations == data.recommendations
    html = render_analysis_report_html(restored)
    text = analysis_report_text(restored)
    assert "第一行<br>第二行<br>依据（标题；简介）" in html
    assert "同功能甲<br>同功能乙" in html
    assert "<script>" not in html
    assert "同功能合并 2 项" in html
    assert "第一行\n第二行\n依据（标题；简介）" in text
    old = report_to_payload(_report())
    for k in ("plugin_id", "similar_count", "similar_plugins"):
        old["recommendations"][0].pop(k)
    assert report_from_payload(old, group_label="旧报告").recommendations[0].similar_count == 0
