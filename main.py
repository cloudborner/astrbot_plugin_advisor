from __future__ import annotations

import asyncio
import functools
import html
import json
import secrets
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from astrbot import __version__ as ASTRBOT_VERSION
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Plain
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import GreedyStr

from .advisor.analysis_audit import (
    AnalysisAuditLog,
    AnalysisAuditRecord,
    audit_id,
    result_digest,
    utc_now_text,
)
from .advisor.analysis_checkpoint import (
    AnalysisCheckpointStore,
    report_to_payload,
)
from .advisor.analysis_draft import (
    AnalysisDraft,
    AnalysisDraftStore,
    created_at_text,
    phrase_sources,
)
from .advisor.capabilities import CapabilityIndex, load_capability_index
from .advisor.chat_history import (
    HistoryFetchError,
    HistoryFetchResult,
    HistoryImportState,
    HistoryImportSummary,
    HistoryMessage,
    HistoryUnavailableError,
    history_message_from_event,
    provider_for_event,
    write_history_export,
)
from .advisor.chat_stats import ChatStatsStore
from .advisor.config import AdvisorConfig, llm_timeout_clamp_notice, parse_config
from .advisor.conflicts import detect_capacity_conflicts
from .advisor.image_evidence import (
    cleanup_prepared_images,
    prepare_images,
    validate_remote_images,
)
from .advisor.index import (
    atomic_write_json,
    get_profile,
    load_index,
    profile_is_current,
    read_index_generated_at,
    validate_index_semantics,
)
from .advisor.llm_fallback import (
    build_analysis_response_format,
    build_assessment_prompt,
    build_candidate_review_prompt,
    build_context_analysis_prompt,
    build_context_analysis_windows,
    build_context_synthesis_prompt,
    build_contract_repair_prompt,
    is_repairable_contract_error,
    merge_assessment,
    merge_validated_context_results,
    needs_llm_fallback,
    parse_assessment,
    parse_candidate_review,
    parse_context_analysis,
)
from .advisor.market import DEFAULT_MARKET_URL, GitHubClient, load_market
from .advisor.models import MAX_MARKET_PLUGINS, PluginRecord
from .advisor.phrase_extraction import extract_phrases
from .advisor.reports import (
    AnalysisReportData,
    NeedCard,
    PhraseReportData,
    PhraseReportRow,
    RecommendationCard,
    analysis_report_text,
    phrase_confirmation_text,
    render_analysis_report_html,
    render_phrase_confirmation_html,
)
from .advisor.resource_rules import build_resource_profile, load_rules
from .advisor.scoring import RecommendationScore, ScoreEngine
from .advisor.system_probe import probe_server
from .advisor.taxonomy import PluginTaxonomy

_HISTORY_EXPORT_RANGE_HOURS: dict[str, int | None] = {
    "all": None,
    "24h": 24,
    "3d": 72,
    "7d": 168,
    "30d": 720,
}
_HISTORY_EXPORT_RANGE_LABELS = {
    "all": "全部（受消息上限限制）",
    "24h": "最近24小时",
    "3d": "最近3天",
    "7d": "最近7天",
    "30d": "最近30天",
}

PLUGIN_NAME = "astrbot_plugin_advisor"
_MAX_LIVE_HISTORY_MESSAGES = 10_000
_MAX_LIVE_HISTORY_CHARS = 8_000_000
_SAFE_ANALYSIS_VALUE_ERRORS = {
    "confirmed analysis payload exceeds safe prompt size": "prompt_too_large",
    "single confirmed analysis unit exceeds safe window size": "window_too_large",
    "context synthesis payload exceeds safe prompt size": "synthesis_too_large",
    "candidate review payload exceeds safe prompt size": "candidate_prompt_too_large",
    "contract repair payload exceeds safe prompt size": "repair_payload_too_large",
    "context analysis fields mismatch": "response_fields_mismatch",
    "invalid group_profile": "invalid_group_profile",
    "invalid needs": "invalid_needs",
    "invalid need fields": "invalid_need_fields",
    "invalid need title": "invalid_need_title",
    "invalid evidence_summary": "invalid_evidence_summary",
    "invalid importance": "invalid_importance",
    "invalid capabilities": "invalid_capabilities",
    "invalid evidence_ids": "invalid_evidence_ids",
    "need cites unknown evidence": "unknown_evidence",
    "confidence must be numeric": "invalid_confidence_type",
    "confidence out of range": "invalid_confidence_range",
    "candidate review fields mismatch": "candidate_fields_mismatch",
    "invalid candidate assessments": "invalid_candidate_assessments",
    "invalid candidate assessment fields": "invalid_candidate_fields",
    "unknown candidate": "unknown_candidate",
    "duplicate candidate assessment": "duplicate_candidate",
    "invalid functional_fit": "invalid_functional_fit",
    "unknown matched need": "unknown_matched_need",
    "candidate evidence does not support matched need": "unsupported_candidate_evidence",
    "invalid candidate reason": "invalid_candidate_reason",
}


def _qq_whitelist_required(handler):
    """Gate every user-invoked command without affecting passive group statistics."""

    @functools.wraps(handler)
    async def wrapped(self, event: AstrMessageEvent, *args, **kwargs):
        denial = self._whitelist_denial(event)
        if denial:
            yield event.plain_result(denial)
            return
        async for result in handler(self, event, *args, **kwargs):
            yield result

    return wrapped


def _report_html(text: str) -> str:
    """Build a compact, escaped HTML report for AstrBot's image renderer."""

    safe_text = str(text)[:30_000]
    lines = safe_text.splitlines() or ["插件顾问"]
    title = html.escape(lines[0].strip() or "插件顾问")
    rows: list[str] = []
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            rows.append('<div class="gap"></div>')
            continue
        escaped = html.escape(line)
        css_class = "item" if line[:1].isdigit() and ". " in line[:5] else "line"
        rows.append(f'<div class="{css_class}">{escaped}</div>')
    body = "".join(rows) or '<div class="line muted">暂无内容</div>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080">
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 48px; width: 1080px; color: #172033;
  background: linear-gradient(145deg, #eef5ff 0%, #f7f9fc 48%, #effbf6 100%);
  font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
.card {{ background: rgba(255,255,255,.96); border: 1px solid #dce6f4;
  border-radius: 28px; padding: 42px 46px; box-shadow: 0 18px 50px rgba(46,76,120,.12); }}
.eyebrow {{ color: #3976d3; font-size: 20px; font-weight: 700; letter-spacing: 3px; }}
h1 {{ margin: 10px 0 30px; font-size: 42px; line-height: 1.25; color: #13213a; }}
.line, .item {{ margin: 10px 0; padding: 14px 18px; border-radius: 14px;
  background: #f7f9fd; font-size: 25px; line-height: 1.55; overflow-wrap: anywhere; }}
.item {{ background: #f0f6ff; border-left: 6px solid #5c91e6; }}
.gap {{ height: 12px; }}
.muted {{ color: #788399; }}
.footer {{ margin-top: 28px; color: #8792a7; font-size: 18px; text-align: right; }}
</style>
</head>
<body><main class="card"><div class="eyebrow">ASTRBOT PLUGIN ADVISOR</div>
<h1>{title}</h1>{body}<div class="footer">插件顾问 · 建议仅供安装前参考</div>
</main></body></html>"""


class PluginAdvisor(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.settings: AdvisorConfig = parse_config(config)
        timeout_notice = llm_timeout_clamp_notice(config)
        if timeout_notice is not None:
            requested_timeout, effective_timeout = timeout_notice
            self._log_warning(
                "模型最长等待时间配置已限制：requested=%d，effective=%d",
                requested_timeout,
                effective_timeout,
            )
        self.root = Path(__file__).resolve().parent
        self.data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rules = load_rules(self.root / "data" / "resource_rules.json")
        self.taxonomy = PluginTaxonomy.from_file(self.root / "data" / "plugin_taxonomy.json")
        try:
            self.capability_index = load_capability_index(
                self.root / "data" / "plugin_capabilities.json"
            )
        except Exception as error:
            self.capability_index = CapabilityIndex.empty()
            self._log_warning("插件功能语义索引加载失败（%s）", type(error).__name__)
        self.index = self._load_best_index()
        self.records: list[PluginRecord] = []
        self.record_by_id: dict[str, PluginRecord] = {}
        self.classifications = {}
        self._market_lock = asyncio.Lock()
        self._market_inflight_task: asyncio.Task | None = None
        self._fallback_lock = asyncio.Lock()
        self._fallback_cache: OrderedDict[tuple[str, str, str], tuple[float, Any]] = OrderedDict()
        self._market_loaded_at = 0.0
        self._github_inflight_task: asyncio.Task | None = None
        self._github_inflight_key: tuple[str, str, str] | None = None
        self._stats_dirty = 0
        salt_path = self.data_dir / "stats_salt.txt"
        if salt_path.exists():
            salt = salt_path.read_text(encoding="utf-8").strip()
        else:
            salt = secrets.token_hex(32)
            salt_path.write_text(salt, encoding="utf-8")
        self._stats_salt = salt
        self._history_fetch_gate = asyncio.Lock()
        self._analysis_gate = asyncio.Semaphore(1)
        self._live_history: OrderedDict[str, deque[HistoryMessage]] = OrderedDict()
        self._live_history_message_count = 0
        self._live_history_char_count = 0
        self._live_history_message_budget = min(
            _MAX_LIVE_HISTORY_MESSAGES,
            max(2_000, self.settings.history_message_limit * 10),
        )
        self._live_history_char_budget = min(
            _MAX_LIVE_HISTORY_CHARS,
            max(
                2_000_000,
                self.settings.history_message_limit * self.settings.max_message_chars * 2,
            ),
        )
        self.analysis_drafts = AnalysisDraftStore(
            ttl_seconds=self.settings.analysis_draft_ttl_minutes * 60,
            max_entries=self.settings.max_group_buckets,
            max_total_messages=_MAX_LIVE_HISTORY_MESSAGES,
            max_total_images=2_000,
            max_total_text_chars=32_000_000,
            max_message_chars=self.settings.max_message_chars,
        )
        self.analysis_audit = AnalysisAuditLog(
            self.data_dir / "analysis_audit.json",
            maximum_records=self.settings.max_runtime_cache_entries,
        )
        self.analysis_checkpoints = AnalysisCheckpointStore(
            self.data_dir / "analysis_checkpoints.json",
            salt=salt,
            maximum_records=min(
                self.settings.max_group_buckets,
                self.settings.max_runtime_cache_entries,
            ),
        )
        self.history_import_state = HistoryImportState(
            self.data_dir / "history_import_state.json",
            salt=salt,
            max_groups=self.settings.max_group_buckets,
            max_seen_per_group=self.settings.history_message_limit,
        )
        self.stats = ChatStatsStore(
            self.data_dir / "group_stats.json",
            salt=salt,
            retention_days=self.settings.statistics_retention_days,
            stopwords=self.settings.stop_words,
            min_word_length=self.settings.word_min_length,
            ngram_max_length=self.settings.word_ngram_max_length,
            top_n=self.settings.word_frequency_top_n,
            keyword_min_count=self.settings.word_min_count,
            enable_word_frequency=self.settings.enable_word_frequency,
            topic_rules=(
                self.settings.topic_rules if self.settings.enable_topic_classification else ()
            ),
            max_text_length=self.settings.max_message_chars,
            max_group_buckets=self.settings.max_group_buckets,
        )
        if self.stats.migrated_from_schema is not None:
            self.history_import_state.clear_all()
            self.history_import_state.save()
            self.stats.save()
        self._log_info(
            "插件顾问已加载：资源画像 %d 条",
            len(self.index.get("profiles", {})),
        )

    def _log_info(self, message: str, *args: Any) -> None:
        if self.settings.enable_logging:
            logger.info(message, *self._safe_log_args(args))

    def _log_warning(self, message: str, *args: Any) -> None:
        if self.settings.enable_logging:
            logger.warning(message, *self._safe_log_args(args))

    @staticmethod
    def _safe_log_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Keep operational logs useful without copying exception payloads."""

        return tuple(
            type(value).__name__ if isinstance(value, BaseException) else value for value in args
        )

    @staticmethod
    def _safe_analysis_error(error: BaseException) -> str:
        """Return a diagnostic code without copying provider/model content."""

        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, json.JSONDecodeError):
            return "invalid_json"
        if isinstance(error, ValueError):
            return _SAFE_ANALYSIS_VALUE_ERRORS.get(str(error), "invalid_value")
        return type(error).__name__

    @staticmethod
    @contextmanager
    def _analysis_phase(run_state: dict[str, Any], phase: str):
        """Accumulate bounded wall-clock time for a named workflow phase."""

        normalized = str(phase)[:48]
        run_state["phase"] = normalized
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = max(0, int((time.monotonic() - started) * 1000))
            durations = run_state.setdefault("stage_durations_ms", {})
            durations[normalized] = min(
                86_400_000,
                max(0, int(durations.get(normalized) or 0)) + elapsed,
            )

    @staticmethod
    def _response_format_unsupported(error: BaseException) -> bool:
        """Recognize only explicit provider/API rejection of response_format."""

        text = str(error).casefold()
        if "response_format" not in text and "json_schema" not in text:
            return False
        return any(
            marker in text
            for marker in (
                "unsupported",
                "not support",
                "does not support",
                "unexpected keyword",
                "unknown parameter",
                "unknown field",
                "invalid parameter",
                "not allowed",
                "not permitted",
            )
        )

    @staticmethod
    def _response_token_usage(response: Any) -> tuple[int, int, int]:
        """Extract AstrBot/OpenAI-compatible token counters without requiring them."""

        usage = getattr(response, "usage", None)
        if usage is None:
            raw_completion = getattr(response, "raw_completion", None)
            usage = getattr(raw_completion, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0, 0, 0

        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", usage.get("input", 0))
            completion = usage.get("completion_tokens", usage.get("output", 0))
            total = usage.get("total_tokens", usage.get("total", 0))
        else:
            prompt = getattr(usage, "prompt_tokens", getattr(usage, "input", 0))
            completion = getattr(usage, "completion_tokens", getattr(usage, "output", 0))
            total = getattr(usage, "total_tokens", getattr(usage, "total", 0))
        try:
            prompt_value = max(0, int(prompt or 0))
            completion_value = max(0, int(completion or 0))
            total_value = max(0, int(total or 0))
        except (TypeError, ValueError, OverflowError):
            return 0, 0, 0
        if total_value == 0:
            total_value = prompt_value + completion_value
        return prompt_value, completion_value, total_value

    def _record_response_usage(self, run_state: dict[str, Any], response: Any) -> None:
        prompt, completion, total = self._response_token_usage(response)
        run_state["prompt_tokens"] = int(run_state.get("prompt_tokens") or 0) + prompt
        run_state["completion_tokens"] = int(run_state.get("completion_tokens") or 0) + completion
        run_state["total_tokens"] = int(run_state.get("total_tokens") or 0) + total

    async def _llm_generate_analysis(
        self,
        *,
        provider_id: str,
        system_prompt: str,
        prompt: str,
        contract_kind: str,
        phase: str,
        run_state: dict[str, Any],
        image_urls: list[str] | None = None,
    ) -> Any:
        """Call an analysis model with native schema and one compatibility fallback."""

        base_kwargs: dict[str, Any] = {
            "chat_provider_id": provider_id,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": 0,
        }
        if image_urls:
            base_kwargs["image_urls"] = image_urls
        response_format = build_analysis_response_format(contract_kind)

        async def request(*, include_schema: bool) -> Any:
            kwargs = dict(base_kwargs)
            if include_schema:
                kwargs["response_format"] = response_format
            run_state["model_called"] = True
            run_state["llm_calls"] = int(run_state.get("llm_calls") or 0) + 1
            response = await asyncio.wait_for(
                self.context.llm_generate(**kwargs),
                timeout=self._llm_timeout(),
            )
            self._record_response_usage(run_state, response)
            return response

        with self._analysis_phase(run_state, phase):
            try:
                return await request(include_schema=True)
            except Exception as error:
                if not self._response_format_unsupported(error):
                    raise
                run_state["schema_fallbacks"] = int(run_state.get("schema_fallbacks") or 0) + 1
                run_state["retried"] = True
                self._log_warning(
                    "当前 Provider 不支持原生 JSON Schema，已兼容降级为提示词契约（phase=%s）",
                    phase,
                )
                return await request(include_schema=False)

    def _append_analysis_audit(
        self,
        *,
        draft: AnalysisDraft,
        run_state: dict[str, Any],
        started_at: str,
        started_monotonic: float,
        status: str,
        result: Any,
        cache_used: bool = False,
    ) -> None:
        if not self.settings.enable_logging:
            return
        duration_ms = max(0, int((time.monotonic() - started_monotonic) * 1000))
        self.analysis_audit.append(
            AnalysisAuditRecord(
                analysis_id=audit_id(
                    message_count=len(draft.messages),
                    phrase_count=len(draft.active_phrases()),
                    nonce=secrets.token_hex(8),
                ),
                started_at=started_at,
                finished_at=utc_now_text(),
                duration_ms=duration_ms,
                model_called=bool(run_state.get("model_called")),
                cache_used=cache_used,
                retried=bool(run_state.get("retried")),
                text_messages=len(draft.messages),
                phrases=len(draft.active_phrases()),
                detected_images=len(draft.images),
                sent_images=max(0, int(run_state.get("attempted_images") or 0)),
                status=str(status)[:48],
                phase=str(run_state.get("phase") or "unknown")[:48],
                result_hash=result_digest(result),
                llm_calls=max(0, int(run_state.get("llm_calls") or 0)),
                prompt_tokens=max(0, int(run_state.get("prompt_tokens") or 0)),
                completion_tokens=max(0, int(run_state.get("completion_tokens") or 0)),
                total_tokens=max(0, int(run_state.get("total_tokens") or 0)),
                schema_fallbacks=max(0, int(run_state.get("schema_fallbacks") or 0)),
                stage_durations_ms=dict(run_state.get("stage_durations_ms") or {}),
            )
        )
        run_state["audit_finished"] = True
        self._log_info(
            "需求分析结束：status=%s，phase=%s，模型调用=%d，tokens=%d，耗时=%dms",
            status,
            run_state.get("phase") or "unknown",
            int(run_state.get("llm_calls") or 0),
            int(run_state.get("total_tokens") or 0),
            duration_ms,
        )

    async def _repair_contract_completion(
        self,
        *,
        provider_id: str,
        contract_kind: str,
        invalid_output: str,
        run_state: dict[str, Any],
    ) -> str:
        """Make one format-only repair call without resending source evidence."""

        system, prompt = build_contract_repair_prompt(
            invalid_output,
            contract_kind=contract_kind,
        )
        run_state["repair_used"] = True
        run_state["retried"] = True
        run_state["model_called"] = True
        run_state["phase"] = f"{contract_kind}_repair"
        self._log_warning(
            "模型输出契约结构异常，尝试一次格式修复（%s）",
            contract_kind,
        )
        response = await self._llm_generate_analysis(
            provider_id=provider_id,
            system_prompt=system,
            prompt=prompt,
            contract_kind=contract_kind,
            phase=f"{contract_kind}_repair",
            run_state=run_state,
        )
        return str(response.completion_text or "")

    def _whitelist_denial(self, event: AstrMessageEvent) -> str | None:
        try:
            sender_id = str(event.get_sender_id() or "").strip()
        except Exception:
            sender_id = ""
        if sender_id and sender_id in self.settings.qq_whitelist:
            return None
        return "你没有权限使用此功能。"

    async def _report_result(self, event: AstrMessageEvent, text: str) -> Any:
        """Return an image report when enabled, with a reliable text fallback."""

        if not self.settings.render_reports_as_image:
            return event.plain_result(text)
        try:
            image_path = await self.html_render(
                _report_html(text),
                {},
                False,
                {"full_page": True, "type": "png", "timeout": 45_000},
            )
            if not isinstance(image_path, str) or not image_path.strip():
                raise ValueError("图片渲染没有返回有效路径")
            return event.image_result(image_path)
        except Exception as exc:
            self._log_warning("图片报告渲染失败，已改发文字：%s", exc)
            return event.plain_result(text)

    async def _structured_report_result(
        self,
        event: AstrMessageEvent,
        *,
        html_text: str,
        fallback_text: str,
    ) -> Any:
        """Render a deterministic report template with a plain-text fallback."""

        if not self.settings.render_reports_as_image:
            return event.plain_result(fallback_text)
        try:
            image_path = await self.html_render(
                html_text,
                {},
                False,
                {"full_page": True, "type": "png", "timeout": 45_000},
            )
            if not isinstance(image_path, str) or not image_path.strip():
                raise ValueError("图片渲染没有返回有效路径")
            return event.image_result(image_path)
        except Exception as exc:
            self._log_warning("结构化图片报告渲染失败，已改发文字：%s", exc)
            return event.plain_result(fallback_text)

    @staticmethod
    def _event_sender_id(event: AstrMessageEvent) -> str:
        try:
            return str(event.get_sender_id() or "").strip()
        except Exception:
            return ""

    def _request_timeout(self) -> float:
        return float(self.settings.request_timeout_seconds)

    def _llm_timeout(self) -> float:
        return float(self.settings.llm_timeout_seconds)

    def _analysis_total_timeout(self) -> float:
        """Bound the complete model-and-recommendation workflow."""

        return max(
            30.0,
            min(240.0, self._llm_timeout() * 3 + self._request_timeout()),
        )

    def _cache_put(self, cache: OrderedDict, key: Any, value: Any) -> None:
        cache[key] = (time.monotonic(), value)
        cache.move_to_end(key)
        while len(cache) > self.settings.max_runtime_cache_entries:
            cache.popitem(last=False)

    def _cache_get(self, cache: OrderedDict, key: Any) -> Any | None:
        entry = cache.get(key)
        if entry is None:
            return None
        created_at, value = entry
        if time.monotonic() - created_at > self.settings.cache_ttl_minutes * 60:
            cache.pop(key, None)
            return None
        cache.move_to_end(key)
        return value

    def _load_best_index(self) -> dict[str, Any]:
        candidates = [
            self.data_dir / "resource_profiles.json",
            self.root / "data" / "source_resource_index.json",
            self.root / "data" / "resource_profiles.json",
        ]
        errors = []
        ranked_candidates: list[tuple[datetime, Path]] = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                ranked_candidates.append((read_index_generated_at(path), path))
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}")
        for _generated_at, path in sorted(
            ranked_candidates, key=lambda item: item[0], reverse=True
        ):
            try:
                loaded = load_index(path)
                validate_index_semantics(loaded)
                if errors:
                    self._log_warning("资源索引加载失败：%s", "; ".join(errors))
                return loaded
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}")
        if errors:
            self._log_warning("资源索引加载失败：%s", "; ".join(errors))
        return {"$meta": {"schema_version": 1}, "profiles": {}}

    async def _ensure_market(self, *, force: bool = False) -> None:
        async with self._market_lock:
            cache_fresh = (
                self.records
                and time.monotonic() - self._market_loaded_at
                <= self.settings.cache_ttl_minutes * 60
            )
            if cache_fresh and not force:
                return
            market_url = self.settings.market_url or DEFAULT_MARKET_URL
            request_timeout = self._request_timeout()
            deadline = request_timeout * 2 + 10
            try:
                running = self._market_inflight_task
                if running is not None and running.done():
                    self._market_inflight_task = None
                    running = None
                if running is None:
                    running = asyncio.create_task(
                        asyncio.to_thread(
                            load_market,
                            market_url,
                            timeout=request_timeout,
                            max_retries=self.settings.network_retries,
                            deadline_seconds=deadline,
                        )
                    )
                    self._market_inflight_task = running
                _, records = await asyncio.wait_for(
                    asyncio.shield(running),
                    timeout=deadline + 2,
                )
                self._market_inflight_task = None
                self._set_records(records)
                self._market_loaded_at = time.monotonic()
                if force:
                    self._fallback_cache.clear()
                cache = {
                    "$meta": {"schema_version": 1},
                    "plugins": {item.plugin_id: item.to_dict() for item in records},
                }
                atomic_write_json(self.data_dir / "market_cache.json", cache)
                return
            except Exception as exc:
                self._log_warning("插件市场请求失败，尝试本地缓存：%s", exc)
            cache_paths = [
                self.data_dir / "market_cache.json",
                self.root / "data" / "market_snapshot.json",
            ]
            for path in cache_paths:
                try:
                    if path.stat().st_size > 16 * 1024 * 1024:
                        continue
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    plugins = raw.get("plugins", {})
                    if not isinstance(plugins, dict) or len(plugins) > MAX_MARKET_PLUGINS:
                        continue
                    records = [
                        PluginRecord.from_market(str(key), value)
                        for key, value in plugins.items()
                        if isinstance(value, dict)
                    ]
                    if records:
                        self._set_records(records)
                        self._market_loaded_at = time.monotonic()
                        return
                except Exception:
                    continue
            raise RuntimeError("无法访问插件市场且没有可用缓存")

    def _set_records(self, records: list[PluginRecord]) -> None:
        self.records = records
        self.record_by_id = {item.plugin_id: item for item in records}
        self.classifications = self.taxonomy.classify_market(records)
        self._market_loaded_at = time.monotonic()

    def _server(self, event: AstrMessageEvent):
        return probe_server(platform=event.get_platform_name(), astrbot_version=ASTRBOT_VERSION)

    async def _profile_for(self, event: AstrMessageEvent, record: PluginRecord):
        profile = get_profile(self.index, record.plugin_id)
        if profile and profile_is_current(
            profile,
            version=record.version,
            record_updated_at=record.updated_at,
        ):
            return profile
        cache_key = (record.plugin_id, record.version, record.repo)
        cached = self._cache_get(self._fallback_cache, cache_key)
        if cached is not None:
            return cached
        async with self._fallback_lock:
            cached = self._cache_get(self._fallback_cache, cache_key)
            if cached is not None:
                return cached
            observation = None
            if self.settings.enable_github_fallback:
                observation = await self._github_observation(record, cache_key)
            profile = build_resource_profile(record, self.rules, observation)
            if self.settings.enable_llm_fallback and needs_llm_fallback(profile):
                profile = await self._augment_with_llm(event, record, profile, observation)
            self._cache_put(self._fallback_cache, cache_key, profile)
            return profile

    async def _github_observation(self, record: PluginRecord, cache_key: tuple[str, str, str]):
        running = self._github_inflight_task
        if running is not None and running.done():
            completed_key = self._github_inflight_key
            try:
                completed = running.result()
            except Exception as exc:
                self._log_warning("后台 GitHub 分析失败：%s", exc)
                completed = None
            self._github_inflight_task = None
            self._github_inflight_key = None
            running = None
            if completed_key == cache_key:
                return completed

        if running is not None and self._github_inflight_key != cache_key:
            self._log_warning("已有 GitHub 静态分析仍在运行，本次使用保守市场画像")
            return None
        if running is None:
            request_timeout = self._request_timeout()
            client = GitHubClient(
                timeout=request_timeout,
                max_retries=self.settings.network_retries,
                min_interval=self.settings.github_min_interval_ms / 1000.0,
            )
            deadline = request_timeout * 3 + 10
            running = asyncio.create_task(
                asyncio.to_thread(
                    client.observe,
                    record.repo,
                    include_sbom=self.settings.enable_github_sbom,
                    deadline_seconds=deadline,
                )
            )
            self._github_inflight_task = running
            self._github_inflight_key = cache_key
        try:
            observation = await asyncio.wait_for(
                asyncio.shield(running),
                timeout=self._request_timeout() * 3 + 12,
            )
            self._github_inflight_task = None
            self._github_inflight_key = None
            return observation
        except TimeoutError:
            self._log_warning("GitHub 静态分析达到绝对期限，暂用保守画像")
            return None
        except Exception as exc:
            self._github_inflight_task = None
            self._github_inflight_key = None
            self._log_warning("GitHub 回退分析失败 %s: %s", record.plugin_id, exc)
            return None

    async def _augment_with_llm(self, event, record, profile, observation):
        facts = {
            "plugin": {
                "plugin_id": record.plugin_id,
                "version": record.version,
                "description": record.desc[:2000],
                "tags": record.tags[:30],
                "category": record.category,
            },
            "static_assessment": profile.to_dict(),
            "tree_paths": [
                item.get("path") for item in (observation.tree[:500] if observation else [])
            ],
            "sbom_packages": observation.packages[:500] if observation else [],
        }
        try:
            provider_id = self.settings.provider_id
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            if not provider_id:
                return profile
            system, prompt = build_assessment_prompt(facts)
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    system_prompt=system,
                    prompt=prompt,
                    temperature=0,
                ),
                timeout=self._llm_timeout(),
            )
            assessment = parse_assessment(response.completion_text)
            return merge_assessment(profile, assessment)
        except Exception as exc:
            self._log_warning("模型辅助分析失败 %s: %s", record.plugin_id, exc)
            return profile

    def _find_records(self, query: str) -> list[PluginRecord]:
        query = query.strip().lower()
        if not query:
            return self.records
        exact = next(
            (item for key, item in self.record_by_id.items() if key.lower() == query),
            None,
        )
        if exact:
            return [exact]
        return [
            item
            for item in self.records
            if query
            in " ".join(
                [
                    item.plugin_id,
                    item.display_name,
                    item.desc,
                    item.short_desc,
                    *item.tags,
                ]
            ).lower()
        ]

    async def _fetch_group_history(
        self,
        event: AstrMessageEvent,
        *,
        group_id: str,
        limit: int | None = None,
    ) -> HistoryFetchResult:
        provider = provider_for_event(
            event,
            timeout_seconds=self.settings.history_request_timeout_seconds,
            page_size=self.settings.history_page_size,
            gate=self._history_fetch_gate,
        )
        return await provider.fetch_group_history(
            group_id=group_id,
            limit=limit or self.settings.history_message_limit,
        )

    @staticmethod
    def _onebot_action_data(response: Any) -> dict[str, Any] | None:
        value = response
        if not isinstance(value, dict) and hasattr(value, "data"):
            value = value.data
        if isinstance(value, dict) and isinstance(value.get("data"), dict):
            value = value["data"]
        return value if isinstance(value, dict) else None

    async def _private_group_role(
        self,
        event: AstrMessageEvent,
        *,
        group_id: str,
    ) -> str | None:
        """Verify both bot membership and operator membership through OneBot."""

        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            return None
        operator_id = self._event_sender_id(event)
        try:
            bot_id = str(event.get_self_id() or "").strip()
        except Exception:
            bot_id = ""
        if not operator_id or not bot_id:
            return None

        async def member_info(user_id: str) -> dict[str, Any] | None:
            response = await asyncio.wait_for(
                call_action(
                    "get_group_member_info",
                    group_id=str(group_id),
                    user_id=str(user_id),
                    no_cache=True,
                ),
                timeout=min(10.0, self.settings.history_request_timeout_seconds),
            )
            data = self._onebot_action_data(response)
            if data is None:
                return None
            returned_group = str(data.get("group_id") or group_id)
            returned_user = str(data.get("user_id") or user_id)
            if returned_group != str(group_id) or returned_user != str(user_id):
                return None
            return data

        try:
            bot_membership = await member_info(bot_id)
            operator_membership = await member_info(operator_id)
        except Exception as exc:
            self._log_warning("私聊目标群权限校验失败（%s）", type(exc).__name__)
            return None
        if bot_membership is None or operator_membership is None:
            return None
        role = str(operator_membership.get("role") or "member").strip().casefold()
        return role if role in {"owner", "admin", "member"} else "member"

    async def _private_group_access_allowed(
        self,
        event: AstrMessageEvent,
        *,
        group_id: str,
        require_admin: bool,
    ) -> bool:
        if not event.is_private_chat():
            return True
        if require_admin:
            if not self.settings.require_private_export_admin:
                return True
        elif not self.settings.require_private_group_membership:
            return True
        role = await self._private_group_role(event, group_id=group_id)
        if role is None:
            return False
        return not require_admin or role in {"owner", "admin"}

    async def _backfill_group_history(
        self,
        event: AstrMessageEvent,
        *,
        platform: str,
        group_id: str,
    ) -> HistoryImportSummary:
        result = await self._fetch_group_history(
            event,
            group_id=group_id,
            limit=self.settings.history_message_limit,
        )
        try:
            self_id = str(event.get_self_id() or "")
        except Exception:
            self_id = ""
        imported = 0
        skipped_seen = 0
        skipped_self = 0
        for message in result.messages:
            if self.history_import_state.contains(
                platform=platform,
                group_id=group_id,
                stable_key=message.stable_key,
            ):
                skipped_seen += 1
                continue
            if self_id and message.sender_id == self_id:
                skipped_self += 1
            else:
                self.stats.observe(
                    platform=platform,
                    group_id=group_id,
                    text=message.semantic_text,
                    component_types=list(message.component_types),
                    occurred_at=message.occurred_at,
                )
                imported += 1
            self.history_import_state.mark(
                platform=platform,
                group_id=group_id,
                stable_key=message.stable_key,
            )
        if imported:
            self.stats.save()
            self._stats_dirty = 0
        self.history_import_state.save()
        return HistoryImportSummary(
            provider=result.provider,
            fetched=len(result.messages),
            imported=imported,
            skipped_seen=skipped_seen,
            skipped_self=skipped_self,
            warning=result.warning,
        )

    def _remember_live_message(
        self,
        *,
        platform: str,
        group_id: str,
        message: HistoryMessage,
    ) -> None:
        key = f"{platform}\0{group_id}"
        bucket = self._live_history.get(key)
        if bucket is None:
            bucket = deque(maxlen=self.settings.history_message_limit)
            self._live_history[key] = bucket
        elif any(item.stable_key == message.stable_key for item in bucket):
            return
        if len(bucket) == bucket.maxlen:
            removed = bucket.popleft()
            self._live_history_message_count -= 1
            self._live_history_char_count -= self._history_message_char_cost(removed)
        bucket.append(message)
        self._live_history_message_count += 1
        self._live_history_char_count += self._history_message_char_cost(message)
        self._live_history.move_to_end(key)
        while len(self._live_history) > self.settings.max_group_buckets:
            _removed_key, removed_bucket = self._live_history.popitem(last=False)
            self._live_history_message_count -= len(removed_bucket)
            self._live_history_char_count -= sum(
                self._history_message_char_cost(item) for item in removed_bucket
            )
        while (
            self._live_history_message_count > self._live_history_message_budget
            or self._live_history_char_count > self._live_history_char_budget
        ):
            oldest_key = next(iter(self._live_history), None)
            if oldest_key is None:
                break
            oldest_bucket = self._live_history[oldest_key]
            removed = oldest_bucket.popleft()
            self._live_history_message_count -= 1
            self._live_history_char_count -= self._history_message_char_cost(removed)
            if not oldest_bucket:
                self._live_history.pop(oldest_key, None)

    @staticmethod
    def _history_message_char_cost(message: HistoryMessage) -> int:
        cost = len(message.text)
        for segment in message.segments:
            data = segment.get("data")
            if isinstance(data, dict):
                cost += sum(len(str(value)) for value in data.values())
        return max(1, cost)

    def _live_messages_for(self, *, platform: str, group_id: str) -> list[HistoryMessage]:
        key = f"{platform}\0{group_id}"
        bucket = self._live_history.get(key)
        if bucket is None:
            return []
        self._live_history.move_to_end(key)
        return list(bucket)

    def _known_analysis_phrases(self) -> tuple[str, ...]:
        values: list[str] = []
        for rule in self.settings.topic_rules:
            if rule.enabled:
                values.extend((rule.display_name, *rule.keywords))
        return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))[:2_000]

    async def _analysis_history(
        self,
        event: AstrMessageEvent,
        *,
        platform: str,
        group_id: str,
    ) -> tuple[list[HistoryMessage], str, str]:
        """Read raw history for a short-lived draft and update only safe statistics."""

        provider_name = "插件启用后收到的消息"
        warning = ""
        messages: list[HistoryMessage] = []
        try:
            result = await self._fetch_group_history(
                event,
                group_id=group_id,
                limit=self.settings.history_message_limit,
            )
            provider_name = result.provider
            warning = result.warning
            messages = list(result.messages)
        except HistoryUnavailableError:
            warning = "当前平台无法补读较早消息，已使用插件收到的消息"
        except HistoryFetchError as exc:
            warning = "历史消息读取失败，已使用当前可用内容"
            self._log_warning("群历史读取失败（%s）", type(exc).__name__)
        if not messages:
            messages = self._live_messages_for(platform=platform, group_id=group_id)

        try:
            self_id = str(event.get_self_id() or "")
        except Exception:
            self_id = ""
        filtered: list[HistoryMessage] = []
        imported = 0
        seen_in_result: set[str] = set()
        for message in messages:
            if message.stable_key in seen_in_result:
                continue
            seen_in_result.add(message.stable_key)
            if self_id and message.sender_id == self_id:
                continue
            filtered.append(message)
            if self.history_import_state.contains(
                platform=platform,
                group_id=group_id,
                stable_key=message.stable_key,
            ):
                continue
            self.stats.observe(
                platform=platform,
                group_id=group_id,
                text=message.semantic_text,
                component_types=list(message.component_types),
                occurred_at=message.occurred_at,
            )
            self.history_import_state.mark(
                platform=platform,
                group_id=group_id,
                stable_key=message.stable_key,
            )
            imported += 1
        if imported:
            self.stats.save()
            self.history_import_state.save()
            self._stats_dirty = 0
        return filtered, provider_name, warning

    def _phrase_report_data(
        self,
        draft: AnalysisDraft,
        *,
        page: int = 1,
        show_all: bool = False,
    ) -> PhraseReportData:
        page_size = 50 if show_all else self.settings.phrase_preview_limit
        rows, pages = draft.visible_phrases(page=page, page_size=page_size)
        return PhraseReportData(
            group_label=draft.group_id,
            effective_messages=len(draft.messages),
            total_phrases=len(draft.active_phrases()),
            rows=tuple(
                PhraseReportRow(
                    index=item.index,
                    phrase=item.text,
                    count=item.count,
                    kind=item.kind,
                    edited=item.edited,
                )
                for item in rows
            ),
            page=max(1, min(pages, int(page))),
            total_pages=pages,
            preview_limit=self.settings.phrase_preview_limit,
            expires_minutes=draft.expires_in_minutes,
            filtered_messages=draft.filtered_message_count,
            history_provider=draft.history_provider,
            history_warning=draft.history_warning,
        )

    async def _phrase_report_result(
        self,
        event: AstrMessageEvent,
        draft: AnalysisDraft,
        *,
        page: int = 1,
        show_all: bool = False,
    ) -> Any:
        data = self._phrase_report_data(draft, page=page, show_all=show_all)
        return await self._structured_report_result(
            event,
            html_text=render_phrase_confirmation_html(data),
            fallback_text=phrase_confirmation_text(data),
        )

    @staticmethod
    def _normalize_repo_identity(value: object) -> str:
        repo = str(value or "").strip().casefold().replace("\\", "/")
        if not repo:
            return ""
        repo = repo.replace("git@github.com:", "https://github.com/")
        repo = repo.replace("ssh://git@github.com/", "https://github.com/")
        if "://" not in repo:
            repo = "https://" + repo.lstrip("/")
        parsed = urlsplit(repo)
        host = parsed.hostname or ""
        host = host.removeprefix("www.")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return ""
        owner, name = parts[:2]
        name = name.removesuffix(".git")
        return f"{host}/{owner}/{name}" if host and owner and name else ""

    def _installed_identities(self) -> tuple[set[str], set[str], set[str]]:
        plugin_ids: set[str] = set()
        names: set[str] = set()
        repos: set[str] = set()
        try:
            metadata_items = list(self.context.get_all_stars())
        except Exception as exc:
            metadata_items = []
            self._log_warning("读取已安装插件清单失败，使用目录回退：%s", exc)
        for metadata in metadata_items:
            plugin_id = str(getattr(metadata, "plugin_id", "") or "").strip().casefold()
            if plugin_id:
                plugin_ids.add(plugin_id)
                names.add(plugin_id.rsplit("/", 1)[-1])
            for field in ("name", "root_dir_name"):
                value = str(getattr(metadata, field, "") or "").strip().casefold()
                if value:
                    names.add(value)
            repo = self._normalize_repo_identity(getattr(metadata, "repo", ""))
            if repo:
                repos.add(repo)
        for metadata_path in self.root.parent.glob("*/metadata.yaml"):
            names.add(metadata_path.parent.name.casefold())
            try:
                for line in metadata_path.read_text(encoding="utf-8").splitlines():
                    key, separator, raw_value = line.partition(":")
                    if not separator:
                        continue
                    value = raw_value.strip().strip("'\"")
                    if key.strip().casefold() in {"name", "plugin_id"} and value:
                        names.add(value.casefold())
                    elif key.strip().casefold() in {"repo", "repository"} and value:
                        repos.add(self._normalize_repo_identity(value))
            except OSError:
                continue
        return plugin_ids, names, repos

    def _installed_prompt_context(self) -> list[dict[str, str]]:
        """Return a bounded, credential-free installed-plugin summary for review."""

        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        try:
            metadata_items = list(self.context.get_all_stars())
        except Exception:
            metadata_items = []
        for metadata in metadata_items[:100]:
            plugin_id = str(getattr(metadata, "plugin_id", "") or "").strip()[:300]
            name = str(
                getattr(metadata, "name", "")
                or getattr(metadata, "root_dir_name", "")
                or plugin_id.rsplit("/", 1)[-1]
            ).strip()[:200]
            key = (plugin_id or name).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "plugin_id": plugin_id,
                    "name": name,
                    "description": str(
                        getattr(metadata, "desc", "") or getattr(metadata, "description", "") or ""
                    ).strip()[:500],
                }
            )
        if rows:
            return rows
        for metadata_path in sorted(self.root.parent.glob("*/metadata.yaml"))[:100]:
            name = metadata_path.parent.name[:200]
            key = name.casefold()
            if key and key not in seen:
                seen.add(key)
                rows.append({"plugin_id": "", "name": name, "description": ""})
        return rows

    def _record_is_installed(
        self,
        record: PluginRecord,
        identities: tuple[set[str], set[str], set[str]],
    ) -> bool:
        plugin_ids, names, repos = identities
        record_id = record.plugin_id.casefold()
        record_names = {
            record.name.casefold(),
            record.display_name.casefold(),
            record_id.rsplit("/", 1)[-1],
        }
        repo = self._normalize_repo_identity(record.repo)
        return bool(
            record_id in plugin_ids or record_names.intersection(names) or (repo and repo in repos)
        )

    def _context_need_map(
        self, model_result: dict[str, Any] | None
    ) -> dict[str, tuple[float, list[str]]]:
        if not model_result:
            return {}
        confidence = max(0.0, min(1.0, float(model_result.get("confidence", 0.0))))
        needs = list(model_result.get("needs") or [])[:3]
        global_terms = list(model_result.get("search_terms") or [])[:12]
        result: dict[str, tuple[float, list[str]]] = {}
        for record in self.records:
            market_searchable = " ".join(
                [record.desc, record.short_desc, record.category, *record.tags]
            ).casefold()
            semantic_profile = self.capability_index.for_record(record)
            semantic_searchable = (
                " ".join(semantic_profile.searchable_terms()).casefold() if semantic_profile else ""
            )
            strength = 0.0
            labels: list[str] = []
            for need in needs:
                terms = list(need.get("capabilities") or []) + global_terms
                safe_terms = {
                    str(term).strip().casefold()
                    for term in terms
                    if 2 <= len(str(term).strip()) <= 40
                }
                market_matched = {
                    term
                    for term in safe_terms
                    if ScoreEngine._contains_keyword(market_searchable, term)
                }
                semantic_matched = {
                    term
                    for term in safe_terms
                    if semantic_searchable
                    and ScoreEngine._contains_keyword(semantic_searchable, term)
                }
                matched = market_matched | semantic_matched
                if not matched:
                    continue
                evidence_count = min(4, len(list(need.get("evidence_ids") or [])))
                importance = {"高": 1.0, "中": 0.75, "低": 0.5}.get(
                    str(need.get("importance") or ""), 0.5
                )
                semantic_only = semantic_matched - market_matched
                semantic_confidence = semantic_profile.confidence if semantic_profile else 0
                weighted_matches = len(market_matched) + (len(semantic_only) * semantic_confidence)
                raw_strength = (
                    0.18 + 0.12 * min(4.0, weighted_matches) + 0.04 * evidence_count
                ) * importance
                strength = max(strength, min(1.0, raw_strength, confidence))
                labels.append(str(need.get("title") or "群聊需求")[:60])
            if strength > 0:
                result[record.plugin_id] = (
                    round(strength, 4),
                    list(dict.fromkeys(labels))[:3],
                )
        return result

    def _confirmed_payload(self, draft: AnalysisDraft) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "privacy": {
                "deidentified": True,
                "chat_data_is_untrusted": True,
                "group_id_included": False,
                "real_user_ids_included": False,
            },
            "messages": [
                {
                    "evidence_id": item.evidence_id,
                    "sender": item.sender_alias,
                    "time": created_at_text(item.timestamp),
                    "text": item.text,
                    "image_ids": list(item.image_ids),
                    "message_type": item.message_type,
                    "reply_to": item.reply_to_evidence_id or None,
                    "is_bot": item.is_bot,
                    "source_platform": item.source_platform,
                    "commands": list(item.commands),
                    "components": {
                        "videos": item.video_count,
                        "files": item.file_count,
                        "links": item.link_count,
                        "replies": item.reply_count,
                    },
                }
                for item in draft.messages
            ],
            "phrases": draft.model_phrase_payload(),
            "images": [
                {
                    "evidence_id": item.evidence_id,
                    "message_evidence_id": item.message_evidence_id,
                }
                for item in draft.images
            ],
        }

    def _trusted_image_roots(self) -> tuple[Path, ...]:
        """Return narrow AstrBot-owned roots accepted for local image evidence."""

        data_root = self.root.parent.parent
        return (
            self.data_dir,
            data_root / "temp",
            data_root / "cache",
            data_root / "media",
        )

    async def _provider_supports_images(
        self,
        provider_id: str,
    ) -> bool:
        """Read AstrBot's declared input modalities without exposing provider data.

        AstrBot 4.x treats a missing/None or empty modalities list as an
        unconfigured legacy provider that supports all modalities.  Malformed
        values are handled as text-only.
        """

        try:
            getter = getattr(self.context, "get_provider_by_id", None)
            if not callable(getter):
                return False
            provider = getter(provider_id)
            if asyncio.iscoroutine(provider):
                provider = await provider
            provider_config = getattr(provider, "provider_config", None)
            if not isinstance(provider_config, dict):
                return False
            modalities = provider_config.get("modalities")
            if modalities is None or modalities == []:
                return True
            return isinstance(modalities, list) and any(
                str(item).strip().lower() == "image" for item in modalities
            )
        except Exception as error:
            self._log_warning("读取模型图片能力失败（%s）", type(error).__name__)
            return False

    async def _run_confirmed_model(
        self,
        event: AstrMessageEvent,
        draft: AnalysisDraft,
        *,
        run_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str, int, int, int, str]:
        standalone_audit = run_state is None
        if run_state is None:
            run_state = {}
        standalone_started_monotonic = time.monotonic()
        standalone_started_at = utc_now_text()
        run_state.setdefault("phase", "provider_selection")
        run_state.setdefault("model_called", False)
        run_state.setdefault("retried", False)
        run_state.setdefault("attempted_images", 0)
        run_state.setdefault("repair_used", False)
        run_state.setdefault("audit_finished", False)
        run_state.setdefault("llm_calls", 0)
        run_state.setdefault("prompt_tokens", 0)
        run_state.setdefault("completion_tokens", 0)
        run_state.setdefault("total_tokens", 0)
        run_state.setdefault("schema_fallbacks", 0)
        run_state.setdefault("stage_durations_ms", {})
        attempted_images = 0
        prepared_images = None

        def finish(
            result: dict[str, Any] | None,
            mode: str,
            sent_images: int,
            limitation: str,
            status: str,
        ) -> tuple[dict[str, Any] | None, str, int, int, int, str]:
            cleanup_prepared_images(prepared_images)
            run_state["model_status"] = status
            if not status.startswith("success"):
                run_state.setdefault(
                    "failure_phase", str(run_state.get("phase") or "unknown")[:48]
                )
            if standalone_audit:
                self._append_analysis_audit(
                    draft=draft,
                    run_state=run_state,
                    started_at=standalone_started_at,
                    started_monotonic=standalone_started_monotonic,
                    status=status,
                    result=result,
                )
            selected_images = len(prepared_images.images) if prepared_images is not None else 0
            skipped_images = max(0, len(draft.images) - sent_images)
            return (
                result,
                mode,
                selected_images,
                sent_images,
                skipped_images,
                limitation,
            )

        provider_id = self.settings.provider_id
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            except Exception:
                provider_id = ""
        if not provider_id:
            return finish(None, "文字分析", 0, "没有可用的需求分析模型", "no_provider")
        payload = self._confirmed_payload(draft)
        with self._analysis_phase(run_state, "context_windowing"):
            windows = await asyncio.to_thread(build_context_analysis_windows, payload)
        message_evidence_text = {item.evidence_id: item.text for item in draft.messages}
        confirmed_phrases = draft.model_phrase_payload()
        grounded_image_ids: set[str] = set()
        limitations: list[str] = []
        provider_supports_images = False
        if self.settings.enable_image_analysis and draft.images:
            provider_supports_images = await self._provider_supports_images(provider_id)
            if not provider_supports_images:
                limitations.append("当前模型无法分析图片内容，已完成文字分析")
        if self.settings.enable_image_analysis and provider_supports_images:
            preliminary_images = await asyncio.to_thread(
                prepare_images,
                draft.images,
                maximum=min(20, self.settings.max_images_for_analysis * 3),
                trusted_local_roots=self._trusted_image_roots(),
            )
            prepared_images = await validate_remote_images(
                preliminary_images,
                maximum=self.settings.max_images_for_analysis,
                timeout_seconds=min(10.0, self._request_timeout()),
                trusted_local_roots=self._trusted_image_roots(),
            )
        selected_images = list(prepared_images.images) if prepared_images else []
        selected_by_id = {item.evidence_id: item for item in selected_images}
        if draft.images and not self.settings.enable_image_analysis:
            limitations.append("图片内容分析已关闭")
        if prepared_images and prepared_images.invalid_count:
            limitations.append(f"有 {prepared_images.invalid_count} 张图片引用无效，已跳过")
        if prepared_images and prepared_images.duplicate_count:
            limitations.append(f"已去除 {prepared_images.duplicate_count} 张重复图片")
        limitation = "；".join(limitations)
        image_mode_available = provider_supports_images
        image_fallback = False
        analyzed_images = 0
        window_results: list[dict[str, Any]] = []

        async def invoke(
            system: str,
            prompt: str,
            *,
            local_allowed_ids: set[str],
            image_urls: list[str] | None = None,
            grounding_text: dict[str, str] | None = None,
            grounding_phrases: list[dict[str, Any]] | None = None,
            analyzed_image_ids: set[str] | None = None,
        ) -> dict[str, Any]:
            try:
                response = await self._llm_generate_analysis(
                    provider_id=provider_id,
                    system_prompt=system,
                    prompt=prompt,
                    contract_kind="context_analysis",
                    phase=str(run_state.get("phase") or "context_analysis"),
                    run_state=run_state,
                    image_urls=image_urls,
                )
                try:
                    return parse_context_analysis(
                        response.completion_text,
                        allowed_evidence_ids=local_allowed_ids,
                        evidence_text_by_id=grounding_text,
                        confirmed_phrases=grounding_phrases,
                        analyzed_image_ids=analyzed_image_ids,
                    )
                except Exception as parse_error:
                    if run_state["repair_used"] or not is_repairable_contract_error(parse_error):
                        raise
                    run_state["retried"] = True
                    repaired = await self._repair_contract_completion(
                        provider_id=provider_id,
                        contract_kind="context_analysis",
                        invalid_output=response.completion_text,
                        run_state=run_state,
                    )
                    return parse_context_analysis(
                        repaired,
                        allowed_evidence_ids=local_allowed_ids,
                        evidence_text_by_id=grounding_text,
                        confirmed_phrases=grounding_phrases,
                        analyzed_image_ids=analyzed_image_ids,
                    )
            except BaseException:
                cleanup_prepared_images(prepared_images)
                raise

        for window in windows:
            run_state["phase"] = "context_analysis"
            local_message_ids = {
                str(item.get("evidence_id") or "")
                for item in list(window.get("messages") or [])
                if str(item.get("evidence_id") or "")
            }
            local_image_ids = [
                str(item.get("evidence_id") or "")
                for item in list(window.get("images") or [])
                if str(item.get("evidence_id") or "")
            ]
            sent_image_ids = [
                image_id
                for image_id in local_image_ids
                if image_mode_available and image_id in selected_by_id
            ]
            image_urls = [selected_by_id[image_id].reference for image_id in sent_image_ids]
            system, prompt = build_context_analysis_prompt(
                window,
                attached_image_ids=sent_image_ids,
            )
            window_messages = list(window.get("messages") or [])
            local_grounding_text = {
                str(item.get("evidence_id") or ""): str(item.get("text") or "")
                for item in window_messages
                if str(item.get("evidence_id") or "")
            }
            local_phrases = [
                dict(item) for item in list(window.get("phrases") or []) if isinstance(item, dict)
            ]
            try:
                if image_urls:
                    attempted_images += len(image_urls)
                    run_state["attempted_images"] = attempted_images
                result = await invoke(
                    system,
                    prompt,
                    local_allowed_ids=local_message_ids | set(sent_image_ids),
                    image_urls=image_urls,
                    grounding_text=local_grounding_text,
                    grounding_phrases=local_phrases,
                    analyzed_image_ids=set(sent_image_ids),
                )
                analyzed_images += len(image_urls)
                grounded_image_ids.update(sent_image_ids)
                window_results.append(result)
                continue
            except Exception as first_error:
                if not image_urls:
                    self._log_warning(
                        "需求模型分段分析失败（%s）",
                        self._safe_analysis_error(first_error),
                    )
                    return finish(
                        None,
                        "文字分析",
                        analyzed_images,
                        "需求分析暂时未完成，请稍后重试",
                        "failed",
                    )
                limitations.append("当前分析方式无法查看图片，已改用文字分析")
                limitation = "；".join(dict.fromkeys(limitations))
                run_state["retried"] = True
                image_mode_available = False
                image_fallback = True
                self._log_warning(
                    "需求模型图文分段失败，改用文字分析（%s）",
                    self._safe_analysis_error(first_error),
                )
                try:
                    text_system, text_prompt = build_context_analysis_prompt(
                        window,
                        attached_image_ids=[],
                    )
                    window_results.append(
                        await invoke(
                            text_system,
                            text_prompt,
                            local_allowed_ids=local_message_ids,
                            grounding_text=local_grounding_text,
                            grounding_phrases=local_phrases,
                            analyzed_image_ids=set(),
                        )
                    )
                except Exception as second_error:
                    self._log_warning(
                        "需求模型文字降级分析失败（%s）",
                        self._safe_analysis_error(second_error),
                    )
                    return finish(
                        None,
                        "文字分析",
                        analyzed_images,
                        "需求分析暂时未完成，请稍后重试",
                        "failed_after_retry",
                    )

        while len(window_results) > 1:
            run_state["phase"] = "context_synthesis"
            merged_results: list[dict[str, Any]] = []
            for start in range(0, len(window_results), 20):
                batch = window_results[start : start + 20]
                if len(batch) == 1:
                    merged_results.append(batch[0])
                    continue
                system, prompt = build_context_synthesis_prompt(batch)
                try:
                    merged_results.append(
                        await invoke(
                            system,
                            prompt,
                            local_allowed_ids=set(message_evidence_text) | grounded_image_ids,
                            grounding_text=message_evidence_text,
                            grounding_phrases=confirmed_phrases,
                            analyzed_image_ids=grounded_image_ids,
                        )
                    )
                except Exception as error:
                    self._log_warning(
                        "需求模型综合结果校验失败，已使用通过校验的分段结果本地合并（%s）",
                        self._safe_analysis_error(error),
                    )
                    merged_results.append(
                        merge_validated_context_results(batch)
                    )
                    run_state["synthesis_fallbacks"] = int(
                        run_state.get("synthesis_fallbacks") or 0
                    ) + 1
                    limitations.append(
                        "模型综合格式异常，已使用通过校验的分段结果本地合并"
                    )
                    limitation = "；".join(dict.fromkeys(limitations))
            window_results = merged_results

        result = window_results[0] if window_results else None
        mode = "图文分析" if analyzed_images else "文字分析"
        status = "success_text_fallback" if image_fallback else "success"
        return finish(result, mode, analyzed_images, limitation, status)

    @staticmethod
    def _resource_level_text(profile: Any) -> str:
        peak = max((int(value) for value in profile.scores.values()), default=0)
        return {0: "轻量", 1: "轻量", 2: "一般", 3: "较高", 4: "重型"}.get(peak, "未知")

    @staticmethod
    def _resource_basis_text(profile: Any, profile_source: str) -> str:
        if "临时" in profile_source:
            return "临时静态估计"
        evidence_level = str(getattr(profile, "evidence_level", "")).casefold()
        if "local_source" in evidence_level or "sbom" in evidence_level:
            return "源码静态评估"
        if "github" in evidence_level or "tree" in evidence_level:
            return "仓库静态评估"
        if "market" in evidence_level:
            return "市场信息估计"
        return "静态评估"

    async def _recommend_for_confirmed_analysis(
        self,
        event: AstrMessageEvent,
        draft: AnalysisDraft,
        model_result: dict[str, Any] | None,
        *,
        run_state: dict[str, Any] | None = None,
    ) -> tuple[tuple[RecommendationCard, ...], int, tuple[str, ...]]:
        if run_state is None:
            run_state = {
                "phase": "candidate_retrieval",
                "model_called": False,
                "retried": False,
                "attempted_images": 0,
                "repair_used": False,
                "audit_finished": False,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "schema_fallbacks": 0,
                "stage_durations_ms": {},
            }
        if not model_result:
            run_state["candidate_review_status"] = "skipped_no_analysis"
            return (), 0, ()
        with self._analysis_phase(run_state, "candidate_retrieval"):
            await self._ensure_market()
        matched = self._context_need_map(model_result)
        identities = self._installed_identities()
        excluded = 0
        covered_need_names: set[str] = set()
        for plugin_id, (_strength, names) in matched.items():
            installed_record = self.record_by_id.get(plugin_id)
            if installed_record is not None and self._record_is_installed(
                installed_record, identities
            ):
                excluded += 1
                covered_need_names.update(names)
        candidates: list[tuple[float, PluginRecord, list[str]]] = []
        for plugin_id, (strength, names) in matched.items():
            record = self.record_by_id.get(plugin_id)
            if record is None:
                continue
            if self._record_is_installed(record, identities):
                continue
            uncovered_names = [name for name in names if name not in covered_need_names]
            if covered_need_names and names and not uncovered_names:
                continue
            candidates.append((strength, record, uncovered_names or names))
        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1].download_count,
                -item[1].stars,
                item[1].plugin_id.casefold(),
            )
        )
        server = self._server(event)
        installed_profiles, unresolved_installed = self._installed_profile_state()
        engine = ScoreEngine(self.records)
        demand = self.stats.demand_for(platform=draft.platform, group_id=draft.group_id)
        prepared: list[tuple[PluginRecord, Any, list[str], list[str], str]] = []
        scan_limit = max(self.settings.recommendation_limit * 4, 20)
        for strength, record, names in candidates[:scan_limit]:
            profile = get_profile(self.index, record.plugin_id)
            profile_source = "内置静态画像"
            if profile is None or not profile_is_current(
                profile,
                version=record.version,
                record_updated_at=record.updated_at,
            ):
                profile = build_resource_profile(record, self.rules)
                profile_source = "临时静态估计"
            conflicts = detect_capacity_conflicts(profile, installed_profiles, server)
            if unresolved_installed:
                conflicts.append("部分已安装插件缺少资源画像，冲突判断可能不完整")
            prepared.append((record, profile, conflicts, names, profile_source))

        if not prepared:
            run_state["candidate_review_status"] = "skipped_no_candidates"
            return (), excluded, tuple(sorted(covered_need_names))

        need_evidence = {
            str(need.get("title") or ""): {
                str(value) for value in list(need.get("evidence_ids") or []) if str(value)
            }
            for need in list(model_result.get("needs") or [])[:3]
            if str(need.get("title") or "")
        }
        review_payload = {
            "confirmed_needs": list(model_result.get("needs") or [])[:3],
            "server": server.to_dict(),
            "installed_plugins": self._installed_prompt_context(),
            "scoring_rules": {
                "total": 100,
                "demand_match": 30,
                "market_usage": {"total": 20, "downloads": 12, "stars": 8},
                "compatibility": 20,
                "resource_fit": 15,
                "maintenance": 10,
                "deployment": 5,
                "final_score_is_local": True,
            },
            "candidates": [
                {
                    "plugin_id": record.plugin_id,
                    "name": record.display_name or record.name,
                    "description": (record.short_desc or record.desc)[:1_000],
                    "category": record.category,
                    "tags": record.tags[:12],
                    "semantic_profile": self.capability_index.prompt_context(record),
                    "version": record.version,
                    "astrbot_version": record.astrbot_version,
                    "support_platforms": record.support_platforms[:12],
                    "market": {
                        "downloads": record.download_count,
                        "stars": record.stars,
                        "updated_at": record.updated_at,
                    },
                    "resource": {
                        "overall_level": self._resource_level_text(profile),
                        "dimensions": profile.levels,
                        "confidence": round(float(profile.confidence), 3),
                        "source": profile_source,
                        "external_processes": profile.external_processes[:8],
                        "background_tasks": profile.background_tasks,
                    },
                }
                for record, profile, _conflicts, _names, profile_source in prepared
            ],
        }
        provider_id = self.settings.provider_id
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            except Exception:
                provider_id = ""
        review_result: dict[str, Any] | None = None
        if provider_id:
            try:
                review_system, review_prompt = build_candidate_review_prompt(review_payload)
                response = await self._llm_generate_analysis(
                    provider_id=provider_id,
                    system_prompt=review_system,
                    prompt=review_prompt,
                    contract_kind="candidate_review",
                    phase="candidate_review",
                    run_state=run_state,
                )
                allowed_plugin_ids = {record.plugin_id for record, *_rest in prepared}
                try:
                    review_result = parse_candidate_review(
                        response.completion_text,
                        allowed_plugin_ids=allowed_plugin_ids,
                        need_evidence=need_evidence,
                    )
                except Exception as parse_error:
                    if run_state["repair_used"] or not is_repairable_contract_error(parse_error):
                        raise
                    repaired = await self._repair_contract_completion(
                        provider_id=provider_id,
                        contract_kind="candidate_review",
                        invalid_output=response.completion_text,
                        run_state=run_state,
                    )
                    review_result = parse_candidate_review(
                        repaired,
                        allowed_plugin_ids=allowed_plugin_ids,
                        need_evidence=need_evidence,
                    )
                run_state["candidate_review_status"] = "success"
            except Exception as exc:
                run_state["candidate_review_status"] = "failed"
                self._log_warning(
                    "候选插件需求复核失败（%s）",
                    self._safe_analysis_error(exc),
                )
        if review_result is None:
            run_state.setdefault("candidate_review_status", "no_provider")
            uncertainty = "候选插件复核未完成，未输出未经模型复核的建议"
            uncertainties = list(model_result.get("uncertainties") or [])
            if uncertainty not in uncertainties and len(uncertainties) < 10:
                uncertainties.append(uncertainty)
                model_result["uncertainties"] = uncertainties
            return (), excluded, tuple(sorted(covered_need_names))

        review_by_id = {
            item["plugin_id"]: item for item in list(review_result.get("assessments") or [])
        }
        review_uncertainties = [
            str(value) for value in list(review_result.get("uncertainties") or []) if str(value)
        ]
        if review_uncertainties:
            uncertainties = list(model_result.get("uncertainties") or [])
            for value in review_uncertainties:
                if value not in uncertainties and len(uncertainties) < 10:
                    uncertainties.append(value)
            model_result["uncertainties"] = uncertainties

        scored: list[tuple[RecommendationScore, PluginRecord, Any, list[str], str]] = []
        confidence = max(0.0, min(1.0, float(model_result.get("confidence", 0.0))))
        for record, profile, conflicts, _names, profile_source in prepared:
            review = review_by_id.get(record.plugin_id)
            if review is None:
                continue
            names = list(review["matched_need_titles"])
            review_fit = float(review["functional_fit"])
            fit = review_fit * (0.70 + 0.30 * confidence)
            review_warnings = [f"需求复核：{value}" for value in review["risks"]]
            score = engine.score(
                record,
                profile,
                server,
                demand,
                conflict_warnings=[*conflicts, *review_warnings],
                topic_match_strength=fit,
                matched_topics=names,
            )
            score.reasons.insert(0, f"需求复核：{review['reason']}")
            if score.total >= self.settings.minimum_recommendation_score:
                scored.append((score, record, profile, names, profile_source))
        scored.sort(key=lambda item: (-item[0].total, item[1].plugin_id.casefold()))
        cards: list[RecommendationCard] = []
        evidence_level = "较充分" if confidence >= 0.75 else "一般" if confidence >= 0.5 else "有限"
        detail_limit = {
            "compact": 1,
            "standard": 2,
            "detailed": max(3, self.settings.report_evidence_limit),
        }.get(self.settings.report_detail, 2)
        for rank, (score, record, profile, names, profile_source) in enumerate(
            scored[: self.settings.recommendation_limit], start=1
        ):
            reasons = list(score.reasons)
            reason = "；".join(reasons[:detail_limit]) or "与已确认的群聊需求相符"
            risk = "；".join(score.warnings[:detail_limit]) or "未发现明显风险"
            profile_confidence = max(0.0, min(1.0, float(getattr(profile, "confidence", 0.0))))
            resource_basis = self._resource_basis_text(profile, profile_source)
            if "临时" in profile_source or profile_confidence < 0.5:
                estimate_warning = "资源占用为低置信度静态估计，建议先隔离测试"
                risk = (
                    estimate_warning
                    if risk == "未发现明显风险"
                    else "；".join(dict.fromkeys((risk, estimate_warning)))
                )
            cards.append(
                RecommendationCard(
                    rank=rank,
                    name=record.display_name or record.name,
                    score=score.total,
                    resource_level=self._resource_level_text(profile),
                    reason=reason,
                    matched_need="、".join(names[:2]),
                    evidence_level=evidence_level,
                    resource_basis=resource_basis,
                    resource_confidence=profile_confidence,
                    risk=f"主要风险：{risk}",
                    external_service=(
                        "需要外部服务" if profile.external_processes else "无需额外服务"
                    ),
                )
            )
        return tuple(cards), excluded, tuple(sorted(covered_need_names))

    async def _confirmed_analysis_result(
        self,
        event: AstrMessageEvent,
        draft: AnalysisDraft,
    ) -> Any:
        workflow_started_at = utc_now_text()
        workflow_started_monotonic = time.monotonic()
        run_state: dict[str, Any] = {
            "phase": "analysis_queue",
            "model_called": False,
            "retried": False,
            "attempted_images": 0,
            "repair_used": False,
            "audit_finished": False,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "schema_fallbacks": 0,
            "stage_durations_ms": {},
        }
        model_result = None
        mode = "文字分析"
        selected_images = 0
        analyzed_images = 0
        skipped_images = len(draft.images)
        limitation = ""
        recommendations: tuple[RecommendationCard, ...] = ()
        excluded = 0
        covered_capabilities: tuple[str, ...] = ()
        workflow_status = "failed"
        try:
            async with asyncio.timeout(self._analysis_total_timeout()):
                async with self._analysis_gate:
                    (
                        model_result,
                        mode,
                        selected_images,
                        analyzed_images,
                        skipped_images,
                        limitation,
                    ) = await self._run_confirmed_model(
                        event,
                        draft,
                        run_state=run_state,
                    )
                    (
                        recommendations,
                        excluded,
                        covered_capabilities,
                    ) = await self._recommend_for_confirmed_analysis(
                        event,
                        draft,
                        model_result,
                        run_state=run_state,
                    )
            workflow_status = str(run_state.get("model_status") or "success")
            if model_result is not None and run_state.get("candidate_review_status") == "failed":
                workflow_status = "candidate_review_failed"
        except TimeoutError:
            limitation = "需求分析超过总处理时限，已停止后续模型调用，请稍后重试"
            phase = str(run_state.get("phase") or "unknown")[:48]
            run_state["failure_phase"] = phase
            workflow_status = "total_timeout"
            self._log_warning(
                "需求分析达到总处理时限：phase=%s（total_timeout）",
                phase,
            )
        if model_result and excluded and not recommendations:
            coverage_note = "当前已安装插件已经基本覆盖本次匹配到的主要能力，无需重复安装"
            limitation = "；".join(item for item in (limitation, coverage_note) if item)
        needs_list: list[NeedCard] = []
        for item in list((model_result or {}).get("needs") or [])[:3]:
            evidence_summary = str(item.get("evidence_summary") or "").strip()
            evidence_ids = [
                str(value) for value in list(item.get("evidence_ids") or []) if str(value)
            ]
            if self.settings.report_detail == "compact":
                evidence = evidence_summary
            else:
                id_limit = 2 if self.settings.report_detail == "standard" else len(evidence_ids)
                selected_ids = "、".join(evidence_ids[:id_limit])
                evidence = " · ".join(value for value in (evidence_summary, selected_ids) if value)
            needs_list.append(
                NeedCard(
                    title=str(item.get("title") or ""),
                    priority=str(item.get("importance") or ""),
                    evidence=evidence,
                )
            )
        needs = tuple(needs_list)
        if model_result:
            conclusion = str(model_result.get("group_profile") or "已完成需求分析")
            confidence = float(model_result.get("confidence", 0.0))
        else:
            conclusion = "本次需求分析未完成，未生成未经模型确认的插件建议"
            confidence = 0.0
        if model_result and not needs:
            uncertainties = [
                str(value).strip()
                for value in list(model_result.get("uncertainties") or [])
                if str(value).strip()
            ]
            zero_need_note = (
                "当前证据只支持聊天主题，尚未形成希望机器人完成的明确任务；"
                "可在词组确认页保留或改写能表达具体任务的词组后重新分析"
            )
            if uncertainties:
                zero_need_note += "；证据缺口：" + uncertainties[0]
            limitation = "；".join(value for value in (limitation, zero_need_note) if value)
        if self.settings.report_detail == "detailed" and model_result and needs:
            uncertainties = [
                str(value).strip()
                for value in list(model_result.get("uncertainties") or [])
                if str(value).strip()
            ][: self.settings.report_evidence_limit]
            if uncertainties:
                uncertainty_note = "仍需留意：" + "；".join(uncertainties)
                limitation = "；".join(value for value in (limitation, uncertainty_note) if value)
        data = AnalysisReportData(
            group_label=draft.group_id,
            generated_at=datetime.now(UTC),
            conclusion=conclusion,
            analysis_mode=mode,
            confidence=confidence,
            needs=needs,
            recommendations=recommendations,
            effective_messages=len(draft.messages),
            detected_images=len(draft.images),
            selected_images=selected_images,
            analyzed_images=analyzed_images,
            skipped_images=skipped_images,
            excluded_installed=excluded,
            covered_capabilities=covered_capabilities,
            limitation=limitation,
        )
        checkpoint_payload = report_to_payload(data)
        if model_result is not None:
            try:
                with self._analysis_phase(run_state, "checkpoint_save"):
                    self.analysis_checkpoints.put(
                        platform=draft.platform,
                        group_id=draft.group_id,
                        report=data,
                        result_hash=result_digest(model_result),
                    )
                run_state["checkpoint_saved"] = True
            except Exception as error:
                run_state["checkpoint_saved"] = False
                self._log_warning("安全检查点保存失败（%s）", error)

        with self._analysis_phase(run_state, "report_render"):
            report_result = await self._structured_report_result(
                event,
                html_text=render_analysis_report_html(data),
                fallback_text=analysis_report_text(data),
            )
        if workflow_status.startswith("success"):
            run_state["phase"] = "complete"
        else:
            run_state["phase"] = str(
                run_state.get("failure_phase") or workflow_status or "unknown"
            )[:48]
        self._append_analysis_audit(
            draft=draft,
            run_state=run_state,
            started_at=workflow_started_at,
            started_monotonic=workflow_started_monotonic,
            status=workflow_status,
            result=checkpoint_payload,
        )
        return report_result

    @filter.command("插件体检")
    @_qq_whitelist_required
    async def health(self, event: AstrMessageEvent):
        """检查服务器资源与资源画像状态。"""
        server = self._server(event)
        meta = self.index.get("$meta", {})
        yield await self._report_result(
            event,
            "插件顾问体检\n"
            f"内存：总计 {server.total_memory_mb} MiB，可用 {server.available_memory_mb} MiB\n"
            f"Swap：总计 {server.swap_total_mb} MiB，可用 {server.swap_free_mb} MiB\n"
            f"CPU：{server.cpu_cores:g} 核｜磁盘可用 {server.disk_free_mb} MiB\n"
            f"资源画像：{len(self.index.get('profiles', {}))} 条｜生成时间 {meta.get('generated_at', '未知')}\n"
            "说明：画像是静态风险估计，不是精确运行占用。",
        )

    @filter.command("资源画像")
    @_qq_whitelist_required
    async def resource_profile(self, event: AstrMessageEvent, query: GreedyStr):
        """查看插件功能说明与静态资源占用画像。"""
        value = str(query).strip()
        if not value:
            yield event.plain_result("请填写插件名称或 plugin_id。")
            return
        await self._ensure_market()
        matches = self._find_records(value)
        if not matches:
            yield event.plain_result("没有找到该插件。")
            return
        record = matches[0]
        semantic = self.capability_index.for_record(record)
        summary = (
            semantic.summary
            if semantic is not None
            else (record.short_desc or record.desc or "暂无功能说明")
        )
        capabilities = (
            list(semantic.capabilities[:8])
            if semantic is not None
            else [*record.tags[:6], record.category]
        )
        use_cases = list(semantic.use_cases[:3]) if semantic is not None else []
        limitations = list(semantic.limitations[:3]) if semantic is not None else []
        profile = await self._profile_for(event, record)
        levels = profile.levels
        yield await self._report_result(
            event,
            f"{record.display_name or record.name}\n"
            f"插件：{record.plugin_id}｜版本 {record.version or '未知'}\n\n"
            "功能说明\n"
            f"{summary}\n"
            f"主要能力：{'、'.join(item for item in capabilities if item) or '暂无结构化能力'}\n"
            f"适用场景：{'；'.join(use_cases) or '暂无结构化场景'}\n"
            f"功能限制：{'；'.join(limitations) or '未发现明确限制'}\n\n"
            "性能与资源占用（静态估计）\n"
            f"内存：空闲 {levels['idle_memory']} / 峰值 {levels['peak_memory']}\n"
            f"CPU：空闲 {levels['idle_cpu']} / 峰值 {levels['peak_cpu']}\n"
            f"磁盘 {levels['disk']}｜网络 {levels['network']}\n"
            f"外部进程：{', '.join(profile.external_processes) or '未发现'}\n"
            f"后台任务：{profile.background_tasks}\n"
            f"画像置信度：{profile.confidence:.0%}（{profile.evidence_level}）\n"
            f"资源依据：{'；'.join(profile.evidence[: self.settings.report_evidence_limit]) or '没有命中已知特征'}\n"
            f"未知项：{'；'.join(profile.unknowns[: self.settings.report_unknown_limit]) or '无'}",
        )

    async def _checkpoint_target(
        self, event: AstrMessageEvent, target_group: str
    ) -> tuple[str, str, str]:
        """Resolve and authorize a group used by recent-report commands."""

        platform = str(event.get_platform_name() or "")
        requested = str(target_group or "").strip()
        if event.is_private_chat():
            if not requested:
                return "", "", "请在命令后填写QQ群号。"
            if not requested.isdigit() or not 5 <= len(requested) <= 20:
                return "", "", "群号格式不正确，请填写5到20位数字的QQ群号。"
            if not await self._private_group_access_allowed(
                event, group_id=requested, require_admin=False
            ):
                return "", "", "你没有权限使用此功能。"
            return platform, requested, ""
        current_group = str(event.get_group_id() or "")
        if not current_group:
            return "", "", "当前会话不是群聊。"
        if requested and requested != current_group:
            return "", "", "群聊中只能查询或重发当前群的最近报告。"
        return platform, current_group, ""

    @filter.command("最近需求分析")
    @_qq_whitelist_required
    async def recent_group_analysis(self, event: AstrMessageEvent, target_group: str = ""):
        """Show bounded metadata for the latest unexpired analysis report."""

        platform, group_id, error = await self._checkpoint_target(event, target_group)
        if error:
            yield event.plain_result(error)
            return
        checkpoint = self.analysis_checkpoints.get(platform=platform, group_id=group_id)
        if checkpoint is None:
            yield event.plain_result("没有找到24小时内可用的需求分析报告，请先完成一次 /需求分析。")
            return
        data = checkpoint.to_report_data(group_label=group_id)
        remaining_minutes = max(0, int((checkpoint.expires_at - time.time()) / 60))
        need_names = "、".join(item.title for item in data.needs) or "未形成可靠需求"
        recommendation_names = "、".join(item.name for item in data.recommendations[:5]) or "无"
        yield event.plain_result(
            "最近需求分析\n"
            f"生成时间：{data.generated_at.astimezone().strftime('%Y-%m-%d %H:%M')}\n"
            f"分析方式：{data.analysis_mode}｜可信度 {data.confidence:.0%}\n"
            f"主要需求：{need_names}\n"
            f"推荐插件：{recommendation_names}\n"
            f"剩余有效期：约 {remaining_minutes} 分钟\n"
            f"结果校验：{checkpoint.result_hash[:12] or '无'}\n"
            "发送 /重发需求分析 可免 Token 重新生成并发送本报告。"
        )

    @filter.command("重发需求分析")
    @_qq_whitelist_required
    async def resend_group_analysis(self, event: AstrMessageEvent, target_group: str = ""):
        """Re-render the latest report checkpoint without calling a model."""

        platform, group_id, error = await self._checkpoint_target(event, target_group)
        if error:
            yield event.plain_result(error)
            return
        checkpoint = self.analysis_checkpoints.get(platform=platform, group_id=group_id)
        if checkpoint is None:
            yield event.plain_result(
                "没有找到24小时内可重发的需求分析报告，请先完成一次 /需求分析。"
            )
            return
        data = checkpoint.to_report_data(group_label=group_id)
        self._log_info("使用安全检查点免 Token 重发需求分析报告")
        yield await self._structured_report_result(
            event,
            html_text=render_analysis_report_html(data),
            fallback_text=analysis_report_text(data),
        )

    @filter.command("需求分析")
    @_qq_whitelist_required
    async def group_analysis(
        self,
        event: AstrMessageEvent,
        target_or_confirmation: str = "",
        confirmation: str = "",
    ):
        """Analyze the current group or a numeric group selected in private chat."""
        raw_arguments = " ".join(
            value
            for value in (
                str(target_or_confirmation).strip(),
                str(confirmation).strip(),
            )
            if value
        )
        confirmation_words = {"确认", "重新分析", "是", "yes"}
        platform = event.get_platform_name()
        is_private = event.is_private_chat()
        if is_private:
            parts = raw_arguments.split()
            if not parts:
                yield event.plain_result(
                    "请指定需要分析的QQ群号。\n发送 /需求分析 群号，确认后再开始分析。"
                )
                return
            target_group_id = parts[0]
            if not target_group_id.isdigit() or not 5 <= len(target_group_id) <= 20:
                yield event.plain_result(
                    "群号格式不正确，请在 /需求分析 后填写5到20位数字的QQ群号。"
                )
                return
            if not await self._private_group_access_allowed(
                event,
                group_id=target_group_id,
                require_admin=False,
            ):
                yield event.plain_result("你没有权限使用此功能。")
                return
            trailing = " ".join(parts[1:]).strip().casefold()
            if trailing and trailing not in confirmation_words:
                yield event.plain_result(
                    "参数格式不正确。\n私聊：/需求分析 群号"
                )
                return
        else:
            target_group_id = str(event.get_group_id() or "")
            if raw_arguments and raw_arguments.casefold() not in confirmation_words:
                yield event.plain_result("参数格式不正确。\n群聊：/需求分析")
                return

        messages, history_provider, history_warning = await self._analysis_history(
            event,
            platform=platform,
            group_id=target_group_id,
        )
        if len(messages) < self.settings.minimum_messages_for_analysis:
            detail = f"\n历史读取提示：{history_warning}" if history_warning else ""
            yield event.plain_result(
                f"当前只有 {len(messages)} 条可分析消息，至少需要 "
                f"{self.settings.minimum_messages_for_analysis} 条。\n"
                "请确认机器人已经加入目标群并能读取群历史。"
                f"{detail}"
            )
            return
        sources = phrase_sources(messages, max_message_chars=self.settings.max_message_chars)
        phrases = await asyncio.to_thread(
            extract_phrases,
            sources,
            known_phrases=self._known_analysis_phrases(),
            blacklist_words=self.settings.blacklist_words,
            blacklist_regexes=self.settings.blacklist_regexes,
            stop_words=self.settings.stop_words,
            minimum_count=1,
        )
        draft = self.analysis_drafts.create(
            owner_id=self._event_sender_id(event),
            platform=platform,
            group_id=target_group_id,
            messages=messages,
            phrases=phrases,
            history_provider=history_provider,
            history_warning=history_warning,
        )
        yield await self._phrase_report_result(event, draft)

    def _active_draft_for_event(self, event: AstrMessageEvent) -> AnalysisDraft | None:
        owner_id = self._event_sender_id(event)
        try:
            is_private = event.is_private_chat()
        except Exception:
            return None
        if is_private:
            draft = self.analysis_drafts.get(owner_id)
        else:
            try:
                draft = self.analysis_drafts.get(
                    owner_id,
                    platform=event.get_platform_name(),
                    group_id=str(event.get_group_id() or ""),
                )
            except Exception:
                return None
        if draft is None:
            return None
        return draft

    @filter.command("显示全部分词")
    @_qq_whitelist_required
    async def show_all_phrases(self, event: AstrMessageEvent, page: int = 1):
        """分页显示当前分析草稿中的全部词组。"""

        draft = self._active_draft_for_event(event)
        if draft is None:
            yield event.plain_result("当前没有可用的分析草稿，请先使用 /需求分析。")
            return
        yield await self._phrase_report_result(
            event,
            draft,
            page=max(1, int(page)),
            show_all=True,
        )

    @filter.command("修改分词")
    @_qq_whitelist_required
    async def modify_phrase(
        self,
        event: AstrMessageEvent,
        index: int,
        new_phrase: GreedyStr,
    ):
        """修改当前草稿中一个稳定编号对应的词组。"""

        draft = self._active_draft_for_event(event)
        if draft is None:
            yield event.plain_result("当前没有可用的分析草稿，请先使用 /需求分析。")
            return
        try:
            draft.modify_phrase(index, str(new_phrase))
        except KeyError:
            yield event.plain_result("没有找到该序号，或该词组已经删除。")
            return
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        yield await self._phrase_report_result(event, draft)

    @filter.command("删除分词")
    @_qq_whitelist_required
    async def delete_phrase(self, event: AstrMessageEvent, index: int):
        """删除当前草稿中一个稳定编号对应的词组。"""

        draft = self._active_draft_for_event(event)
        if draft is None:
            yield event.plain_result("当前没有可用的分析草稿，请先使用 /需求分析。")
            return
        try:
            draft.delete_phrase(index)
        except KeyError:
            yield event.plain_result("没有找到该序号，或该词组已经删除。")
            return
        yield await self._phrase_report_result(event, draft)

    @filter.command("确认分词")
    @_qq_whitelist_required
    async def confirm_phrases(self, event: AstrMessageEvent):
        """确认词组并开始唯一一次真实需求模型分析。"""

        draft = self._active_draft_for_event(event)
        if draft is None:
            yield event.plain_result("当前没有可用的分析草稿，请先使用 /需求分析。")
            return
        try:
            yield await self._confirmed_analysis_result(event, draft)
        finally:
            self.analysis_drafts.pop(
                self._event_sender_id(event),
                platform=draft.platform,
                group_id=draft.group_id,
            )

    @filter.command("取消分析")
    @_qq_whitelist_required
    async def cancel_analysis(self, event: AstrMessageEvent):
        """删除当前用户的短期分析草稿。"""

        draft = self._active_draft_for_event(event)
        removed = (
            self.analysis_drafts.pop(
                self._event_sender_id(event),
                platform=draft.platform,
                group_id=draft.group_id,
            )
            if draft is not None
            else None
        )
        if removed is None:
            yield event.plain_result("当前没有可取消的分析草稿。")
            return
        yield event.plain_result("本次分析已取消，临时聊天上下文和词组草稿已清除。")

    @filter.command("导出聊天记录")
    @_qq_whitelist_required
    async def export_chat_history(
        self,
        event: AstrMessageEvent,
        arguments: GreedyStr = "",
    ):
        """Export recent QQ group history through a compatible OneBot action."""

        tokens = str(arguments).strip().split()
        if len(tokens) > 1:
            yield event.plain_result(
                "参数格式不正确，请使用 /导出聊天记录 [群号]。"
                "导出格式和时间范围请在插件高级设置中选择。"
            )
            return
        requested_group = tokens[0] if tokens else ""
        if requested_group and (
            not requested_group.isdigit() or not 5 <= len(requested_group) <= 20
        ):
            yield event.plain_result("群号格式不正确，请填写5到20位数字。")
            return

        is_private = event.is_private_chat()
        if is_private:
            if not requested_group:
                yield event.plain_result(
                    "请指定需要导出的QQ群号。\n示例：/导出聊天记录 123456789"
                )
                return
            group_id = requested_group
            if not await self._private_group_access_allowed(
                event,
                group_id=group_id,
                require_admin=True,
            ):
                yield event.plain_result("你没有权限使用此功能。")
                return
        else:
            group_id = str(event.get_group_id() or "")
            if not group_id:
                yield event.plain_result("当前会话不是可导出的群聊。")
                return
            if requested_group and requested_group != group_id:
                yield event.plain_result(
                    "群聊中只能导出当前群；如需导出其他群，请在私聊中指定群号。"
                )
                return

        safe_limit = self.settings.history_message_limit
        export_format = self.settings.history_export_format
        export_time_range = self.settings.history_export_time_range

        try:
            result = await self._fetch_group_history(
                event,
                group_id=group_id,
                limit=safe_limit,
            )
        except (HistoryUnavailableError, HistoryFetchError) as exc:
            self._log_warning("聊天记录导出读取失败（%s）", type(exc).__name__)
            yield event.plain_result("无法读取聊天记录，请确认平台连接后重试。")
            return
        range_hours = _HISTORY_EXPORT_RANGE_HOURS[export_time_range]
        if range_hours is not None:
            cutoff = int((datetime.now(UTC) - timedelta(hours=range_hours)).timestamp())
            filtered_messages = tuple(
                message
                for message in result.messages
                if message.timestamp is not None and message.timestamp >= cutoff
            )
            excluded_count = len(result.messages) - len(filtered_messages)
            warning_parts = [part for part in (result.warning,) if part]
            if excluded_count:
                warning_parts.append(
                    f"按{_HISTORY_EXPORT_RANGE_LABELS[export_time_range]}排除 "
                    f"{excluded_count} 条过期或时间未知的消息"
                )
            result = HistoryFetchResult(
                messages=filtered_messages,
                provider=result.provider,
                requested=result.requested,
                reached_limit=result.reached_limit,
                warning="；".join(warning_parts),
            )
        if not result.messages:
            yield event.plain_result(
                f"{_HISTORY_EXPORT_RANGE_LABELS[export_time_range]}内没有可导出的消息。"
            )
            return

        try:
            export_path = write_history_export(
                self.data_dir / "exports",
                group_id=group_id,
                result=result,
                export_format=export_format,
                export_time_range=export_time_range,
            )
        except (OSError, ValueError) as exc:
            self._log_warning("生成聊天记录文件失败（%s）", type(exc).__name__)
            yield event.plain_result("聊天记录已读取，但生成导出文件失败，请稍后重试。")
            return

        warning_note = f"；{result.warning}" if result.warning else ""
        yield event.chain_result(
            [
                Plain(
                    f"已从 {result.provider} 导出群 {group_id} 的 "
                    f"{len(result.messages)} 条消息（{export_format.upper()}，"
                    f"{_HISTORY_EXPORT_RANGE_LABELS[export_time_range]}）{warning_note}。\n"
                    "媒体文件不会下载，导出中只保留消息段和可用引用。"
                ),
                File(name=export_path.name, file=str(export_path.resolve())),
            ]
        )

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-1000)
    async def collect_group_stats(self, event: AstrMessageEvent):
        try:
            live_message = history_message_from_event(event)
            if live_message is None:
                return
            self.stats.observe(
                platform=event.get_platform_name(),
                group_id=event.get_group_id(),
                text=live_message.semantic_text,
                component_types=list(live_message.component_types),
                occurred_at=live_message.occurred_at,
            )
            platform = event.get_platform_name()
            group_id = str(event.get_group_id() or "")
            self._remember_live_message(
                platform=platform,
                group_id=group_id,
                message=live_message,
            )
            self.history_import_state.mark(
                platform=platform,
                group_id=group_id,
                stable_key=live_message.stable_key,
            )
            self._stats_dirty += 1
            if self._stats_dirty >= self.settings.stats_flush_interval_messages:
                self.stats.save()
                self.history_import_state.save()
                self._stats_dirty = 0
        except Exception as exc:
            self._log_warning("需求数据统计失败：%s", exc)

    def _installed_plugin_names(self) -> set[str]:
        names: set[str] = set()
        try:
            for metadata in self.context.get_all_stars():
                for value in (
                    getattr(metadata, "plugin_id", ""),
                    getattr(metadata, "name", ""),
                    getattr(metadata, "root_dir_name", ""),
                ):
                    if value:
                        names.add(str(value))
        except Exception as exc:
            self._log_warning("读取 AstrBot 已安装插件清单失败，使用目录回退：%s", exc)
        plugin_root = self.root.parent
        for metadata in plugin_root.glob("*/metadata.yaml"):
            try:
                for line in metadata.read_text(encoding="utf-8").splitlines():
                    if line.lower().startswith("name:"):
                        names.add(line.split(":", 1)[1].strip().strip("'\""))
                        break
            except OSError:
                continue
        return names

    def _installed_profile_state(self):
        profiles = []
        try:
            metadata_items = list(self.context.get_all_stars())
        except Exception:
            return profiles, 0
        by_plugin_id: dict[str, list[PluginRecord]] = {}
        by_repo: dict[str, list[PluginRecord]] = {}
        by_name: dict[str, list[PluginRecord]] = {}
        for record in self.records:
            by_plugin_id.setdefault(record.plugin_id.casefold(), []).append(record)
            by_repo.setdefault(record.repo.casefold(), []).append(record)
            by_name.setdefault(record.name.casefold(), []).append(record)
        unresolved = 0
        seen_profiles: set[str] = set()
        for metadata in metadata_items:
            plugin_id = str(getattr(metadata, "plugin_id", "") or "").strip().casefold()
            repo = str(getattr(metadata, "repo", "") or "").strip().casefold()
            names = {
                str(getattr(metadata, field, "") or "").strip().casefold()
                for field in ("name", "root_dir_name")
            }
            names.discard("")
            matched: list[PluginRecord] = []
            for mapping, keys in (
                (by_plugin_id, [plugin_id] if plugin_id else []),
                (by_repo, [repo] if repo else []),
                (by_name, sorted(names)),
            ):
                candidates = {item.plugin_id: item for key in keys for item in mapping.get(key, [])}
                if len(candidates) == 1:
                    matched = list(candidates.values())
                    break
            if not matched:
                unresolved += 1
                continue
            resolved_id = matched[0].plugin_id
            profile = get_profile(self.index, resolved_id)
            if profile is None:
                unresolved += 1
            elif resolved_id not in seen_profiles:
                seen_profiles.add(resolved_id)
                profiles.append(profile)
        return profiles, unresolved

    async def terminate(self):
        if self._github_inflight_task is not None:
            # Threads cannot be force-cancelled; the observation has an absolute
            # deadline, so stop awaiting it and let the bounded worker exit.
            self._github_inflight_task.cancel()
        if self._market_inflight_task is not None:
            self._market_inflight_task.cancel()
        try:
            self.stats.save()
            self.history_import_state.save()
        except Exception as exc:
            self._log_warning("保存需求数据失败：%s", exc)
