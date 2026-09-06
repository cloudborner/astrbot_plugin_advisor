import json
import unittest

from advisor.grounded_review import (
    ReviewShapeError,
    build_grounded_review_prompt,
    ground_need_capabilities,
    parse_grounded_review,
)
from advisor.llm_fallback import build_analysis_response_format, parse_context_analysis


class GroundedReviewTests(unittest.TestCase):
    def test_duplicate_capability_status_is_rejected_as_repairable_structure(self):
        payload = self.payload()
        _, _, facts = build_grounded_review_prompt(payload)
        check = {"need_index": 1, "capability_index": 1, "status": "supported", "source_id": "f1", "quote": "支持查询余额"}
        with self.assertRaises(ReviewShapeError):
            parse_grounded_review(json.dumps({"assessments": [{"plugin_id": "owner/balance", "checks": [check, check]}]}), payload, facts)

    def test_comparison_cannot_create_a_memory_requirement_even_with_cherry_picked_quote(self):
        source = "长期记忆能记住我的偏好，但记住信息不等于会主动发起关心。"
        need = {"capabilities": ["记忆偏好"], "evidence_ids": ["m1"], "capability_evidence": [
            {"capability": "记忆偏好", "evidence_id": "m1", "quote": "长期记忆能记住我的偏好", "intent": "request"},
        ]}
        self.assertEqual(ground_need_capabilities(need, {"m1": source}, set())[0], [])

    def test_negation_and_unrelated_quote_cannot_prove_a_claim(self):
        for cap, source, quote in (
            ("记忆偏好", "不要记忆偏好", "记忆偏好"),
            ("统计API消耗", "希望查询API余额", "希望查询API余额"),
            ("情绪支持", "希望保存用户记忆", "希望保存用户记忆"),
        ):
            with self.subTest(cap=cap):
                need = {"capabilities": [cap], "evidence_ids": ["m1"], "capability_evidence": [
                    {"capability": cap, "evidence_id": "m1", "quote": quote, "intent": "request"},
                ]}
                self.assertEqual(ground_need_capabilities(need, {"m1": source}, set())[0], [])

    def test_explicit_request_retains_its_own_provenance(self):
        need = {"capabilities": ["主动关怀"], "evidence_ids": ["m1"], "capability_evidence": [
            {"capability": "主动关怀", "evidence_id": "m1", "quote": "希望主动关怀", "intent": "request"},
        ]}
        caps, proof = ground_need_capabilities(need, {"m1": "希望主动关怀，但不要频繁打扰"}, set())
        self.assertEqual(caps, ["主动关怀"])
        self.assertEqual(proof[0]["evidence_id"], "m1")

    def test_image_evidence_must_be_actually_attached(self):
        need = {"capabilities": ["图片搜索"], "evidence_ids": ["i1"], "capability_evidence": [
            {"capability": "图片搜索", "evidence_id": "i1", "quote": "可见图片搜索需求", "intent": "visual"},
        ]}
        self.assertEqual(ground_need_capabilities(need, {}, set())[0], [])
        self.assertEqual(ground_need_capabilities(need, {}, {"i1"})[0], ["图片搜索"])

    def test_request_and_separate_constraint_quote_retains_only_positive_clause(self):
        source = "只等我发消息还不够，希望能在群聊安静后主动关怀，但不要频繁打扰。"
        need = {"capabilities": ["群聊安静后主动关怀"], "evidence_ids": ["m1"], "capability_evidence": [
            {"capability": "群聊安静后主动关怀", "evidence_id": "m1",
             "quote": "希望能在群聊安静后主动关怀，但不要频繁打扰", "intent": "request"},
        ]}
        caps, proofs = ground_need_capabilities(need, {"m1": source}, set())
        self.assertEqual(caps, ["群聊安静后主动关怀"])
        self.assertEqual(proofs[0]["quote"], "希望能在群聊安静后主动关怀")
        need["capabilities"] = ["记忆偏好"]
        need["capability_evidence"][0].update(capability="记忆偏好", quote="希望主动关怀，但不要记忆偏好")
        self.assertEqual(ground_need_capabilities(need, {"m1": "希望主动关怀，但不要记忆偏好"}, set())[0], [])

    def test_removed_capability_is_also_removed_from_summary(self):
        raw = {"group_profile": "主动关怀与记忆", "needs": [{
            "title": "主动关怀与记忆", "importance": "中", "capabilities": ["主动关怀", "记忆偏好"],
            "evidence_ids": ["m1", "m2"], "evidence_summary": "希望主动关怀和记忆偏好",
            "capability_evidence": [
                {"capability": "主动关怀", "evidence_id": "m1", "quote": "希望主动关怀", "intent": "request"},
                {"capability": "记忆偏好", "evidence_id": "m2", "quote": "长期记忆能记住偏好", "intent": "comparison"},
            ],
        }], "unsuitable_capabilities": [], "uncertainties": [], "confidence": 0.8, "search_terms": ["主动关怀"]}
        parsed = parse_context_analysis(json.dumps(raw), allowed_evidence_ids={"m1", "m2"},
                                        evidence_text_by_id={"m1": "希望主动关怀", "m2": "长期记忆能记住偏好，但不等于主动关怀"},
                                        require_capability_evidence=True)
        self.assertEqual(parsed["needs"][0]["capabilities"], ["主动关怀"])
        self.assertNotIn("记忆", parsed["needs"][0]["evidence_summary"])
        self.assertEqual(parsed["needs"][0]["evidence_ids"], ["m1"])

    def payload(self):
        return {"confirmed_needs": [{"title": "官方账户查询", "capabilities": ["查询余额", "统计调用消耗"], "evidence_ids": ["m1"]}],
                "candidates": [{"plugin_id": "owner/balance", "name": "余额工具", "description": "支持查询余额", "semantic_profile": {}}]}

    def test_balance_quote_does_not_support_consumption_and_reason_is_locally_derived(self):
        payload = self.payload()
        _, _, facts = build_grounded_review_prompt(payload)
        raw = {"assessments": [{"plugin_id": "owner/balance", "checks": [
            {"need_index": 1, "capability_index": i, "status": "supported", "source_id": "f1", "quote": "支持查询余额"}
            for i in (1, 2)
        ]}]}
        result = parse_grounded_review(json.dumps(raw), payload, facts)["assessments"][0]
        self.assertEqual(result["functional_fit"], 0.5)
        self.assertIn("资料未证实统计调用消耗", result["reason"])
        self.assertNotIn("支持统计调用消耗", result["reason"])

    def test_invented_quotes_and_unknown_ids_cannot_produce_recommendations(self):
        payload = self.payload()
        _, _, facts = build_grounded_review_prompt(payload)
        item = {"plugin_id": "owner/balance", "checks": [
            {"need_index": 1, "capability_index": 1, "status": "supported", "source_id": "f1", "quote": "不存在的查询余额依据"},
        ]}
        self.assertEqual(parse_grounded_review(json.dumps({"assessments": [item]}), payload, facts)["assessments"], [])
        item["plugin_id"] = "invented/plugin"
        with self.assertRaises(ValueError):
            parse_grounded_review(json.dumps({"assessments": [item]}), payload, facts)

    def test_absence_of_evidence_is_not_proof_of_no_support(self):
        payload = self.payload()
        _, _, facts = build_grounded_review_prompt(payload)
        raw = {"assessments": [{"plugin_id": "owner/balance", "checks": [
            {"need_index": 1, "capability_index": 1, "status": "supported", "source_id": "f1", "quote": "支持查询余额"},
            {"need_index": 1, "capability_index": 2, "status": "unsupported", "source_id": "", "quote": ""},
        ]}]}
        result = parse_grounded_review(json.dumps(raw), payload, facts)["assessments"][0]
        self.assertIn("资料未证实统计调用消耗", result["reason"])
        self.assertNotIn("不支持统计调用消耗", result["reason"])

    def test_new_native_schemas_require_capability_evidence(self):
        schema = build_analysis_response_format("context_analysis", grounded=True)["json_schema"]["schema"]
        self.assertIn("capability_evidence", schema["properties"]["needs"]["items"]["required"])
        schema = build_analysis_response_format("capability_review")["json_schema"]["schema"]
        self.assertIn("checks", schema["properties"]["assessments"]["items"]["required"])
