import asyncio
import json
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

from advisor.chat_history import HistoryMessage
from advisor.llm_fallback import parse_context_analysis
from tests import test_main_integration as fixtures


def response(evidence_id="消息0001", aggregate_id=None):
    return {"group_profile": "资料搜索需求", "needs": [{
        "title": "资料搜索", "importance": "中", "capabilities": ["资料搜索"],
        "evidence_ids": [aggregate_id or evidence_id], "evidence_summary": "希望资料搜索",
        "capability_evidence": [{"capability": "资料搜索", "evidence_id": evidence_id,
                                 "quote": "希望资料搜索", "intent": "request"}],
    }], "unsuitable_capabilities": [], "uncertainties": [], "confidence": 0.8,
        "search_terms": ["资料搜索"]}


class EvidencePreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = fixtures._load_main_module()

    _plugin = fixtures.MainIntegrationTests._plugin

    def test_valid_atomic_proof_survives_invalid_redundant_aggregate_id(self):
        result = parse_context_analysis(json.dumps(response(aggregate_id="unknown")),
            allowed_evidence_ids={"消息0001"}, evidence_text_by_id={"消息0001": "希望资料搜索"},
            require_capability_evidence=True)
        self.assertEqual(result["needs"][0]["evidence_ids"], ["消息0001"])

    def test_unknown_proof_id_or_wrong_quote_never_recovers_need(self):
        for raw in (response("unknown"), response()):
            if raw["needs"][0]["evidence_ids"] == ["消息0001"]:
                raw["needs"][0]["capability_evidence"][0]["quote"] = "希望资料搜索不存在的原文"
            result = parse_context_analysis(json.dumps(raw), allowed_evidence_ids={"消息0001"},
                evidence_text_by_id={"消息0001": "希望资料搜索"}, require_capability_evidence=True)
            self.assertEqual(result["needs"], [])

    def test_stringified_object_fields_use_only_verified_atomic_proofs(self):
        raw = response(aggregate_id="{quote: '希望资料搜索', intent: 'request'}")
        raw["needs"][0]["capabilities"] = ["{capability: '资料搜索', evidence_id: '"]
        result = parse_context_analysis(json.dumps(raw), allowed_evidence_ids={"消息0001"},
            evidence_text_by_id={"消息0001": "希望资料搜索"}, require_capability_evidence=True)
        self.assertEqual(result["needs"][0]["capabilities"], ["资料搜索"])
        self.assertEqual(result["needs"][0]["evidence_ids"], ["消息0001"])

    def test_empty_or_invalid_synthesis_keeps_verified_windows(self):
        for invalid in (False, True, "partial"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                plugin = self._plugin(directory, config={"general": {"provider_id": "mock"},
                    "advanced": {"candidate_selection_mode": "full_market"}})
                messages = [HistoryMessage(message_id=str(i), sequence=i, timestamp=1700000000+i,
                    group_id="123456789", sender_id="10001", sender_name="成员", text="希望资料搜索",
                    segments=(), component_types=("text",)) for i in (1, 2)]
                draft = plugin.analysis_drafts.create(owner_id="10001", platform="aiocqhttp",
                    group_id="123456789", messages=messages, phrases=[])
                windows = [{"messages": [{"evidence_id": f"消息{i:04d}", "text": "希望资料搜索"}],
                            "phrases": [], "images": []} for i in (1, 2)]
                merged = response("unknown") if invalid else {**response(), "needs": []}
                second = response("消息0002")
                if invalid == "partial":
                    second = json.loads(json.dumps(second, ensure_ascii=False).replace("资料搜索", "资料整理"))
                    windows[1]["messages"][0]["text"] = "希望资料整理"
                    # Draft grounding uses the complete source map for synthesis.
                    draft = plugin.analysis_drafts.create(owner_id="10001", platform="aiocqhttp",
                        group_id="123456789", messages=[messages[0],
                            fixtures.HistoryMessage(message_id="2", sequence=2, timestamp=1700000002,
                                group_id="123456789", sender_id="10001", sender_name="成员",
                                text="希望资料整理", segments=(), component_types=("text",))], phrases=[])
                    merged = response()
                plugin.context.llm_generate = AsyncMock(side_effect=[types.SimpleNamespace(
                    completion_text=json.dumps(r)) for r in (response(), second, merged)])
                state = {}
                with patch.object(self.module, "build_context_analysis_windows", return_value=windows):
                    result = asyncio.run(plugin._run_confirmed_model(fixtures._Event(), draft, run_state=state))[0]
                if invalid == "partial":
                    self.assertEqual({c for n in result["needs"] for c in n["capabilities"]}, {"资料搜索", "资料整理"})
                else:
                    self.assertEqual(result["needs"][0]["evidence_ids"], ["消息0001", "消息0002"])
                self.assertEqual(state["synthesis_fallbacks"], 1)
                self.assertEqual(plugin.context.llm_generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
