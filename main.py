from __future__ import annotations

import asyncio
import functools
import hashlib
import html
import json
import secrets
import time
from collections import Counter, OrderedDict, deque
from datetime import UTC, datetime
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
from .advisor.analysis_draft import (
    AnalysisDraft,
    AnalysisDraftStore,
    created_at_text,
    phrase_sources,
)
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
from .advisor.config import AdvisorConfig, parse_config
from .advisor.conflicts import detect_capacity_conflicts
from .advisor.image_evidence import (
    cleanup_prepared_images,
    prepare_images,
    validate_remote_images,
)
from .advisor.index import (
    atomic_write_json,
    get_profile,
    index_generated_at,
    load_index,
    profile_is_current,
    validate_index_semantics,
)
from .advisor.llm_fallback import (
    build_assessment_prompt,
    build_candidate_review_prompt,
    build_context_analysis_prompt,
    build_context_analysis_windows,
    build_context_synthesis_prompt,
    build_group_analysis_prompt,
    merge_assessment,
    needs_llm_fallback,
    parse_assessment,
    parse_candidate_review,
    parse_context_analysis,
    parse_group_analysis,
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
from .advisor.taxonomy import PluginTaxonomy, TopicMatch

PLUGIN_NAME = "astrbot_plugin_advisor"
MAX_GROUP_MODEL_PAYLOAD_BYTES = 20 * 1024


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


def _bounded_group_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-only group payload with a hard UTF-8 byte ceiling."""

    bounded = json.loads(json.dumps(payload, ensure_ascii=False))

    def payload_bytes() -> int:
        return len(
            json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )

    while payload_bytes() > MAX_GROUP_MODEL_PAYLOAD_BYTES:
        topic_features = bounded.get("topic_features")
        topic_rows = (
            topic_features.get("topics")
            if isinstance(topic_features, dict)
            and isinstance(topic_features.get("topics"), list)
            else []
        )
        if topic_rows:
            topic_rows.pop()
        elif (
            isinstance(bounded.get("cooccurrences"), list) and bounded["cooccurrences"]
        ):
            bounded["cooccurrences"].pop()
        elif isinstance(bounded.get("trends"), list) and bounded["trends"]:
            bounded["trends"].pop()
        elif isinstance(bounded.get("top_terms"), list) and bounded["top_terms"]:
            bounded["top_terms"].pop()
        elif isinstance(bounded.get("commands"), dict) and bounded["commands"]:
            bounded["commands"].pop(next(reversed(bounded["commands"])))
        elif (
            isinstance(bounded.get("intent_counts"), dict) and bounded["intent_counts"]
        ):
            bounded["intent_counts"].pop(next(reversed(bounded["intent_counts"])))
        else:
            sample = bounded.get("sample")
            bounded = {
                "schema_version": 2,
                "privacy": {
                    "aggregate_only": True,
                    "raw_messages_included": False,
                    "identity_fields_included": False,
                },
                "sample": sample if isinstance(sample, dict) else {},
            }
            if payload_bytes() > MAX_GROUP_MODEL_PAYLOAD_BYTES:
                bounded = {"schema_version": 2, "privacy": {"aggregate_only": True}}
            break
    return bounded


class PluginAdvisor(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.settings: AdvisorConfig = parse_config(config)
        self.root = Path(__file__).resolve().parent
        self.data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rules = load_rules(self.root / "data" / "resource_rules.json")
        self.taxonomy = PluginTaxonomy.from_file(
            self.root / "data" / "plugin_taxonomy.json"
        )
        self.index = self._load_best_index()
        self.records: list[PluginRecord] = []
        self.record_by_id: dict[str, PluginRecord] = {}
        self.classifications = {}
        self._market_lock = asyncio.Lock()
        self._market_inflight_task: asyncio.Task | None = None
        self._fallback_lock = asyncio.Lock()
        self._fallback_cache: OrderedDict[tuple[str, str, str], tuple[float, Any]] = (
            OrderedDict()
        )
        self._group_model_cache: OrderedDict[str, tuple[int, float, dict[str, Any]]] = (
            OrderedDict()
        )
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
        self.analysis_drafts = AnalysisDraftStore(
            ttl_seconds=self.settings.analysis_draft_ttl_minutes * 60,
            max_entries=self.settings.max_group_buckets,
        )
        self.analysis_audit = AnalysisAuditLog(
            self.data_dir / "analysis_audit.json",
            maximum_records=self.settings.max_runtime_cache_entries,
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
                self.settings.topic_rules
                if self.settings.enable_topic_classification
                else ()
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
            type(value).__name__ if isinstance(value, BaseException) else value
            for value in args
        )

    def _whitelist_denial(self, event: AstrMessageEvent) -> str | None:
        try:
            sender_id = str(event.get_sender_id() or "").strip()
        except Exception:
            sender_id = ""
        if sender_id and sender_id in self.settings.qq_whitelist:
            return None
        return "你没有权限使用此功能。"

    async def _report_result(
        self, event: AstrMessageEvent, text: str
    ) -> Any:
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
        loaded_candidates: list[dict[str, Any]] = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                loaded = load_index(path)
                validate_index_semantics(loaded)
                loaded_candidates.append(loaded)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        if errors:
            self._log_warning("资源索引加载失败：%s", "; ".join(errors))
        if loaded_candidates:
            return max(loaded_candidates, key=index_generated_at)
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
                    if (
                        not isinstance(plugins, dict)
                        or len(plugins) > MAX_MARKET_PLUGINS
                    ):
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
        return probe_server(
            platform=event.get_platform_name(), astrbot_version=ASTRBOT_VERSION
        )

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
                profile = await self._augment_with_llm(
                    event, record, profile, observation
                )
            self._cache_put(self._fallback_cache, cache_key, profile)
            return profile

    async def _github_observation(
        self, record: PluginRecord, cache_key: tuple[str, str, str]
    ):
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
                item.get("path")
                for item in (observation.tree[:500] if observation else [])
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

    def _topic_matches(
        self, *, platform: str, group_id: str
    ) -> tuple[dict[str, float], dict[str, int], list[TopicMatch]]:
        summary = self.stats.summary_for(platform=platform, group_id=group_id)
        if (
            int(summary.get("messages", 0))
            < self.settings.minimum_messages_for_analysis
        ):
            return {}, {}, []
        demand = self.stats.demand_for(platform=platform, group_id=group_id)
        keywords = self.stats.keyword_frequencies_for(
            platform=platform,
            group_id=group_id,
            top_n=self.settings.word_frequency_top_n,
            min_count=self.settings.word_min_count,
        )
        matches = self.taxonomy.infer_topics(
            keywords,
            demand,
            limit=self.settings.llm_max_topics,
        )
        if not self.settings.enable_topic_classification:
            return demand, keywords, []
        existing = {item.topic_id for item in matches}
        for rule in self.settings.topic_rules:
            if not rule.enabled or rule.topic_id in existing:
                continue
            hits = float(demand.get(f"topic:{rule.topic_id}", 0.0))
            if hits < self.settings.topic_match_min_score:
                continue
            strength = min(
                1.0, hits / max(1.0, self.settings.topic_match_min_score * 3)
            )
            matches.append(
                TopicMatch(
                    topic_id=rule.topic_id,
                    name=rule.display_name,
                    hit_count=max(1, int(round(hits))),
                    strength=round(strength, 4),
                    confidence=round(min(0.85, 0.45 + 0.08 * hits), 4),
                    categories=(),
                    evidence=(f"内置聚合规则加权命中 {hits:g} 次",),
                )
            )
        matches.sort(key=lambda item: (-item.strength, -item.hit_count, item.topic_id))
        return demand, keywords, matches[: self.settings.llm_max_topics]

    def _plugin_topic_map(
        self, topic_matches: list[TopicMatch]
    ) -> dict[str, tuple[float, list[str]]]:
        by_topic = {item.topic_id: item for item in topic_matches}
        result: dict[str, tuple[float, list[str]]] = {}
        for item in self.taxonomy.match_plugins(
            self.records, topic_matches, limit=max(1, len(self.records))
        ):
            names = [by_topic[topic].name for topic in item.matched_topics]
            result[item.plugin_id] = (item.match_strength, names)

        # Versioned built-in rules may define topics absent from the taxonomy file.
        active_rules = [
            rule
            for rule in self.settings.topic_rules
            if rule.enabled and rule.topic_id in by_topic and rule.plugin_keywords
        ]
        for record in self.records:
            text = " ".join(
                [
                    record.plugin_id,
                    record.display_name,
                    record.desc,
                    record.short_desc,
                    record.category,
                    *record.tags,
                ]
            ).casefold()
            custom = [
                rule
                for rule in active_rules
                if any(
                    ScoreEngine._contains_keyword(text, term)
                    for term in rule.plugin_keywords
                )
            ]
            if not custom:
                continue
            prior_strength, prior_names = result.get(record.plugin_id, (0.0, []))
            custom_strength = max(by_topic[rule.topic_id].strength for rule in custom)
            names = list(
                dict.fromkeys(
                    prior_names + [by_topic[rule.topic_id].name for rule in custom]
                )
            )
            result[record.plugin_id] = (max(prior_strength, custom_strength), names)
        return result

    def _model_need_map(
        self, model_result: dict[str, Any] | None
    ) -> dict[str, tuple[float, list[str]]]:
        """Map model-discovered capabilities to market metadata deterministically.

        The model cannot select plugin IDs or assign recommendation points.  It may
        only supply bounded query terms that cite aggregate feature IDs; model-only
        demand strength is capped at 0.45 before the fixed scoring engine sees it.
        """

        if not model_result:
            return {}
        confidence = max(0.0, min(0.70, float(model_result.get("confidence", 0.0))))
        needs = list(model_result.get("emerging_needs") or [])[:6]
        known_scores = dict(model_result.get("theme_scores") or {})
        topic_names = {topic.topic_id: topic.name for topic in self.taxonomy.topics}
        result: dict[str, tuple[float, list[str]]] = {}
        for record in self.records:
            # Model-only discovery deliberately searches descriptive metadata but
            # excludes identifiers and display names.  A model therefore cannot
            # target one plugin merely by echoing its plugin_id, slug or repo URL.
            text = " ".join(
                [
                    record.desc,
                    record.short_desc,
                    record.category,
                    *record.tags,
                ]
            ).casefold()
            matched_names: list[str] = []
            strength = 0.0
            for need in needs:
                terms = list(need.get("query_terms") or []) + list(
                    need.get("capabilities") or []
                )
                matched = {
                    str(term).casefold()
                    for term in terms[:16]
                    if 2 <= len(str(term).strip()) <= 40
                    and ScoreEngine._contains_keyword(text, str(term))
                }
                if not matched:
                    continue
                evidence_count = min(
                    3, len(list(need.get("evidence_feature_ids") or []))
                )
                raw_strength = (
                    0.10 + 0.07 * min(4, len(matched)) + 0.03 * evidence_count
                )
                strength = max(strength, min(0.45, raw_strength, confidence * 0.65))
                matched_names.append(str(need.get("label") or "模型发现需求")[:60])

            classification = self.classifications.get(record.plugin_id)
            if classification is not None:
                for topic_id in classification.topics:
                    score = max(0.0, min(1.0, float(known_scores.get(topic_id, 0.0))))
                    if score <= 0:
                        continue
                    strength = max(strength, min(0.45, score * confidence * 0.65))
                    matched_names.append(topic_names.get(topic_id, topic_id))
            if strength > 0:
                result[record.plugin_id] = (
                    round(strength, 4),
                    list(dict.fromkeys(matched_names))[:5],
                )
        return result

    @staticmethod
    def _merge_topic_maps(
        deterministic: dict[str, tuple[float, list[str]]],
        model: dict[str, tuple[float, list[str]]],
    ) -> dict[str, tuple[float, list[str]]]:
        merged = dict(deterministic)
        for plugin_id, (strength, names) in model.items():
            prior_strength, prior_names = merged.get(plugin_id, (0.0, []))
            merged[plugin_id] = (
                max(prior_strength, min(0.45, strength)),
                list(dict.fromkeys([*prior_names, *names]))[:5],
            )
        return merged

    async def _group_context(
        self,
        event: AstrMessageEvent,
        *,
        target_platform: str | None = None,
        target_group_id: str | None = None,
        force_model_refresh: bool = False,
    ) -> tuple[
        dict[str, float],
        dict[str, int],
        list[TopicMatch],
        dict[str, Any] | None,
        dict[str, tuple[float, list[str]]],
    ]:
        platform = target_platform or event.get_platform_name()
        group_id = (
            str(target_group_id)
            if target_group_id is not None
            else str(event.get_group_id() or "")
        )
        demand, keywords, topics = self._topic_matches(
            platform=platform, group_id=group_id
        )
        model_result = None
        summary = self.stats.summary_for(platform=platform, group_id=group_id)
        if (
            group_id
            and self.settings.enable_group_statistics
            and self.settings.enable_topic_classification
            and int(summary.get("messages", 0))
            >= self.settings.minimum_messages_for_analysis
        ):
            model_result = await self._llm_group_analysis(
                event,
                topics,
                target_platform=platform,
                target_group_id=group_id,
                force_refresh=force_model_refresh,
            )
        plugin_topics = self._merge_topic_maps(
            self._plugin_topic_map(topics), self._model_need_map(model_result)
        )
        return demand, keywords, topics, model_result, plugin_topics

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
        bucket.append(message)
        self._live_history.move_to_end(key)
        while len(self._live_history) > self.settings.max_group_buckets:
            self._live_history.popitem(last=False)

    def _live_messages_for(self, *, platform: str, group_id: str) -> list[HistoryMessage]:
        key = f"{platform}\0{group_id}"
        bucket = self._live_history.get(key)
        if bucket is None:
            return []
        self._live_history.move_to_end(key)
        return list(bucket)

    def _known_analysis_phrases(self) -> tuple[str, ...]:
        values: list[str] = []
        for topic in self.taxonomy.topics:
            values.extend((topic.name, *topic.aliases))
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
        if self.settings.enable_history_backfill:
            try:
                result = await self._fetch_group_history(
                    event,
                    group_id=group_id,
                    limit=self.settings.history_message_limit,
                )
                provider_name = result.provider
                warning = result.warning
                messages = list(result.messages)
            except HistoryUnavailableError as exc:
                warning = str(exc)
            except HistoryFetchError as exc:
                warning = f"历史读取失败：{exc}"
                self._log_warning("群历史读取失败：%s", exc)
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
                        getattr(metadata, "desc", "")
                        or getattr(metadata, "description", "")
                        or ""
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
            record_id in plugin_ids
            or record_names.intersection(names)
            or (repo and repo in repos)
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
            searchable = " ".join(
                [record.desc, record.short_desc, record.category, *record.tags]
            ).casefold()
            strength = 0.0
            labels: list[str] = []
            for need in needs:
                terms = list(need.get("capabilities") or []) + global_terms
                matched = {
                    str(term).strip().casefold()
                    for term in terms
                    if 2 <= len(str(term).strip()) <= 40
                    and ScoreEngine._contains_keyword(searchable, str(term))
                }
                if not matched:
                    continue
                evidence_count = min(4, len(list(need.get("evidence_ids") or [])))
                importance = {"高": 1.0, "中": 0.75, "低": 0.5}.get(
                    str(need.get("importance") or ""), 0.5
                )
                raw_strength = (
                    0.18 + 0.12 * min(4, len(matched)) + 0.04 * evidence_count
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

    async def _run_confirmed_model(
        self,
        event: AstrMessageEvent,
        draft: AnalysisDraft,
    ) -> tuple[dict[str, Any] | None, str, int, int, int, str]:
        started_monotonic = time.monotonic()
        started_at = utc_now_text()
        model_called = False
        retried = False
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
            if self.settings.enable_logging:
                finished_at = utc_now_text()
                self.analysis_audit.append(
                    AnalysisAuditRecord(
                        analysis_id=audit_id(
                            message_count=len(draft.messages),
                            phrase_count=len(draft.active_phrases()),
                            nonce=secrets.token_hex(8),
                        ),
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=max(
                            0, int((time.monotonic() - started_monotonic) * 1000)
                        ),
                        model_called=model_called,
                        cache_used=False,
                        retried=retried,
                        text_messages=len(draft.messages),
                        phrases=len(draft.active_phrases()),
                        detected_images=len(draft.images),
                        sent_images=attempted_images,
                        status=status,
                        result_hash=result_digest(result),
                    )
                )
            selected_images = (
                len(prepared_images.images) if prepared_images is not None else 0
            )
            skipped_images = max(0, len(draft.images) - sent_images)
            return (
                result,
                mode,
                selected_images,
                sent_images,
                skipped_images,
                limitation,
            )

        if not self.settings.enable_llm_group_summary:
            return finish(None, "文字分析", 0, "需求模型分析已关闭", "disabled")
        provider_id = self.settings.provider_id
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            except Exception:
                provider_id = ""
        if not provider_id:
            return finish(
                None, "文字分析", 0, "没有可用的需求分析模型", "no_provider"
            )
        payload = self._confirmed_payload(draft)
        windows = build_context_analysis_windows(payload)
        message_evidence_text = {
            item.evidence_id: item.text for item in draft.messages
        }
        confirmed_phrases = draft.model_phrase_payload()
        grounded_image_ids: set[str] = set()
        if self.settings.enable_image_analysis:
            preliminary_images = prepare_images(
                draft.images,
                maximum=min(20, self.settings.max_images_for_analysis * 3),
            )
            prepared_images = await validate_remote_images(
                preliminary_images,
                maximum=self.settings.max_images_for_analysis,
                timeout_seconds=min(10.0, self._request_timeout()),
            )
        selected_images = list(prepared_images.images) if prepared_images else []
        selected_by_id = {item.evidence_id: item for item in selected_images}
        limitations: list[str] = []
        if draft.images and not self.settings.enable_image_analysis:
            limitations.append("图片内容分析已关闭")
        if prepared_images and prepared_images.invalid_count:
            limitations.append(
                f"有 {prepared_images.invalid_count} 张图片引用无效，已跳过"
            )
        if prepared_images and prepared_images.duplicate_count:
            limitations.append(
                f"已去除 {prepared_images.duplicate_count} 张重复图片"
            )
        limitation = "；".join(limitations)
        image_mode_available = True
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
            nonlocal model_called
            kwargs: dict[str, Any] = {
                "chat_provider_id": provider_id,
                "system_prompt": system,
                "prompt": prompt,
                "temperature": 0,
            }
            if image_urls:
                kwargs["image_urls"] = image_urls
            model_called = True
            response = await asyncio.wait_for(
                self.context.llm_generate(**kwargs),
                timeout=self._llm_timeout(),
            )
            return parse_context_analysis(
                response.completion_text,
                allowed_evidence_ids=local_allowed_ids,
                evidence_text_by_id=grounding_text,
                confirmed_phrases=grounding_phrases,
                analyzed_image_ids=analyzed_image_ids,
            )

        for window in windows:
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
            image_urls = [
                selected_by_id[image_id].reference
                for image_id in sent_image_ids
            ]
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
                dict(item)
                for item in list(window.get("phrases") or [])
                if isinstance(item, dict)
            ]
            try:
                if image_urls:
                    attempted_images += len(image_urls)
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
                    self._log_warning("需求模型分段分析失败：%s", first_error)
                    return finish(
                        None,
                        "文字分析",
                        analyzed_images,
                        f"需求分析未完成：{first_error}",
                        "failed",
                    )
                limitations.append("当前分析方式无法查看图片，已改用文字分析")
                limitation = "；".join(dict.fromkeys(limitations))
                retried = True
                image_mode_available = False
                image_fallback = True
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
                    self._log_warning("需求模型文字降级分析失败：%s", second_error)
                    return finish(
                        None,
                        "文字分析",
                        analyzed_images,
                        f"需求分析未完成：{second_error}",
                        "failed_after_retry",
                    )

        while len(window_results) > 1:
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
                            local_allowed_ids=set(message_evidence_text)
                            | grounded_image_ids,
                            grounding_text=message_evidence_text,
                            grounding_phrases=confirmed_phrases,
                            analyzed_image_ids=grounded_image_ids,
                        )
                    )
                except Exception as error:
                    self._log_warning("需求模型综合分析失败：%s", error)
                    return finish(
                        None,
                        "文字分析",
                        analyzed_images,
                        f"需求综合分析未完成：{error}",
                        "failed",
                    )
            window_results = merged_results

        result = window_results[0] if window_results else None
        mode = "图文分析" if analyzed_images else "文字分析"
        status = "success_text_fallback" if image_fallback else "success"
        return finish(result, mode, analyzed_images, limitation, status)

    @staticmethod
    def _resource_level_text(profile: Any) -> str:
        peak = max((int(value) for value in profile.scores.values()), default=0)
        return {0: "轻量", 1: "轻量", 2: "一般", 3: "较高", 4: "重型"}.get(
            peak, "未知"
        )

    async def _recommend_for_confirmed_analysis(
        self,
        event: AstrMessageEvent,
        draft: AnalysisDraft,
        model_result: dict[str, Any] | None,
    ) -> tuple[tuple[RecommendationCard, ...], int, tuple[str, ...]]:
        if not model_result:
            return (), 0, ()
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
        prepared: list[
            tuple[PluginRecord, Any, list[str], list[str], str]
        ] = []
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
            return (), excluded, tuple(sorted(covered_need_names))

        need_evidence = {
            str(need.get("title") or ""): {
                str(value)
                for value in list(need.get("evidence_ids") or [])
                if str(value)
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
                review_system, review_prompt = build_candidate_review_prompt(
                    review_payload
                )
                response = await asyncio.wait_for(
                    self.context.llm_generate(
                        chat_provider_id=provider_id,
                        system_prompt=review_system,
                        prompt=review_prompt,
                        temperature=0,
                    ),
                    timeout=self._llm_timeout(),
                )
                review_result = parse_candidate_review(
                    response.completion_text,
                    allowed_plugin_ids={record.plugin_id for record, *_rest in prepared},
                    need_evidence=need_evidence,
                )
            except Exception as exc:
                self._log_warning(
                    "候选插件需求复核失败（%s）", type(exc).__name__
                )
        if review_result is None:
            uncertainty = "候选插件复核未完成，未输出未经模型复核的建议"
            uncertainties = list(model_result.get("uncertainties") or [])
            if uncertainty not in uncertainties and len(uncertainties) < 10:
                uncertainties.append(uncertainty)
                model_result["uncertainties"] = uncertainties
            return (), excluded, tuple(sorted(covered_need_names))

        review_by_id = {
            item["plugin_id"]: item
            for item in list(review_result.get("assessments") or [])
        }
        review_uncertainties = [
            str(value)
            for value in list(review_result.get("uncertainties") or [])
            if str(value)
        ]
        if review_uncertainties:
            uncertainties = list(model_result.get("uncertainties") or [])
            for value in review_uncertainties:
                if value not in uncertainties and len(uncertainties) < 10:
                    uncertainties.append(value)
            model_result["uncertainties"] = uncertainties

        scored: list[tuple[RecommendationScore, PluginRecord, Any, list[str]]] = []
        confidence = max(0.0, min(1.0, float(model_result.get("confidence", 0.0))))
        for record, profile, conflicts, _names, _profile_source in prepared:
            review = review_by_id.get(record.plugin_id)
            if review is None:
                continue
            names = list(review["matched_need_titles"])
            fit = min(float(review["functional_fit"]), confidence)
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
                scored.append((score, record, profile, names))
        scored.sort(key=lambda item: (-item[0].total, item[1].plugin_id.casefold()))
        cards: list[RecommendationCard] = []
        evidence_level = (
            "较充分" if confidence >= 0.75 else "一般" if confidence >= 0.5 else "有限"
        )
        detail_limit = {
            "compact": 1,
            "standard": 2,
            "detailed": max(3, self.settings.report_evidence_limit),
        }.get(self.settings.report_detail, 2)
        for rank, (score, record, profile, names) in enumerate(
            scored[: self.settings.recommendation_limit], start=1
        ):
            reasons = list(score.reasons)
            reason = (
                "；".join(reasons[:detail_limit])
                or "与已确认的群聊需求相符"
            )
            risk = "；".join(score.warnings[:detail_limit]) or "未发现明显风险"
            cards.append(
                RecommendationCard(
                    rank=rank,
                    name=record.display_name or record.name,
                    score=score.total,
                    resource_level=self._resource_level_text(profile),
                    reason=reason,
                    matched_need="、".join(names[:2]),
                    evidence_level=evidence_level,
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
        async with self._analysis_gate:
            (
                model_result,
                mode,
                selected_images,
                analyzed_images,
                skipped_images,
                limitation,
            ) = (
                await self._run_confirmed_model(event, draft)
            )
            (
                recommendations,
                excluded,
                covered_capabilities,
            ) = await self._recommend_for_confirmed_analysis(event, draft, model_result)
        if model_result and excluded and not recommendations:
            coverage_note = "当前已安装插件已经基本覆盖本次匹配到的主要能力，无需重复安装"
            limitation = "；".join(
                item for item in (limitation, coverage_note) if item
            )
        needs_list: list[NeedCard] = []
        for item in list((model_result or {}).get("needs") or [])[:3]:
            evidence_summary = str(item.get("evidence_summary") or "").strip()
            evidence_ids = [
                str(value)
                for value in list(item.get("evidence_ids") or [])
                if str(value)
            ]
            if self.settings.report_detail == "compact":
                evidence = evidence_summary
            else:
                id_limit = 2 if self.settings.report_detail == "standard" else len(evidence_ids)
                selected_ids = "、".join(evidence_ids[:id_limit])
                evidence = " · ".join(
                    value for value in (evidence_summary, selected_ids) if value
                )
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
        if self.settings.report_detail == "detailed" and model_result:
            uncertainties = [
                str(value).strip()
                for value in list(model_result.get("uncertainties") or [])
                if str(value).strip()
            ][: self.settings.report_evidence_limit]
            if uncertainties:
                uncertainty_note = "仍需留意：" + "；".join(uncertainties)
                limitation = "；".join(
                    value for value in (limitation, uncertainty_note) if value
                )
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
        return await self._structured_report_result(
            event,
            html_text=render_analysis_report_html(data),
            fallback_text=analysis_report_text(data),
        )

    async def _llm_group_analysis(
        self,
        event: AstrMessageEvent,
        topic_matches: list[TopicMatch],
        *,
        target_platform: str | None = None,
        target_group_id: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        if not self.settings.enable_llm_group_summary:
            return None
        allowed_themes = {topic.topic_id for topic in self.taxonomy.topics}
        allowed_themes.update(rule.topic_id for rule in self.settings.topic_rules)
        platform = target_platform or event.get_platform_name()
        group_id = (
            str(target_group_id)
            if target_group_id is not None
            else str(event.get_group_id() or "")
        )
        aggregate = self.stats.model_features_for(
            platform=platform, group_id=group_id
        )
        aggregate["topic_features"] = self.taxonomy.model_feature_payload(
            topic_matches, limit=self.settings.llm_max_topics
        )
        demand_counts = aggregate.pop("demand_counts", None)
        aggregate["intent_counts"] = {
            str(key): value
            for key, value in (
                demand_counts.items() if isinstance(demand_counts, dict) else ()
            )
            if not str(key).startswith("topic:")
        }
        aggregate = _bounded_group_model_payload(aggregate)
        allowed_feature_ids = {
            str(item.get("feature_id"))
            for collection_name in ("top_terms", "cooccurrences", "trends")
            for item in (
                aggregate.get(collection_name)
                if isinstance(aggregate.get(collection_name), list)
                else []
            )
            if isinstance(item, dict) and item.get("feature_id")
        }
        allowed_intents = set(aggregate["intent_counts"])
        messages = int(
            aggregate.get("sample", {}).get("eligible_messages", 0)
            if isinstance(aggregate.get("sample"), dict)
            else aggregate.get("messages", 0)
        )
        revision = max(1, messages // 25)
        raw_cache_key = (
            f"{self._stats_salt}\0{platform}\0{group_id}"
        ).encode("utf-8", errors="ignore")
        cache_key = hashlib.sha256(raw_cache_key).hexdigest()[:24]
        cached = self._group_model_cache.get(cache_key)
        if force_refresh:
            self._group_model_cache.pop(cache_key, None)
            cached = None
        if (
            cached is not None
            and cached[0] == revision
            and time.monotonic() - cached[1] < self.settings.cache_ttl_minutes * 60
        ):
            self._group_model_cache.move_to_end(cache_key)
            return cached[2]
        try:
            provider_id = self.settings.provider_id
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            if not provider_id:
                return None
            system, prompt = build_group_analysis_prompt(aggregate, allowed_themes)
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    system_prompt=system,
                    prompt=prompt,
                    temperature=0,
                ),
                timeout=self._llm_timeout(),
            )
            parsed = parse_group_analysis(
                response.completion_text,
                allowed_themes=allowed_themes,
                allowed_feature_ids=allowed_feature_ids,
                allowed_intents=allowed_intents,
            )
            self._group_model_cache[cache_key] = (
                revision,
                time.monotonic(),
                parsed,
            )
            self._group_model_cache.move_to_end(cache_key)
            while (
                len(self._group_model_cache) > self.settings.max_runtime_cache_entries
            ):
                self._group_model_cache.popitem(last=False)
            return parsed
        except Exception as exc:
            self._log_warning("需求模型分析失败：%s", exc)
            return None

    def _format_score(self, item: RecommendationScore, record: PluginRecord) -> str:
        name = record.display_name or record.name
        warning_limit = 1 if self.settings.report_detail == "compact" else 2
        if self.settings.report_detail == "detailed":
            warning_limit = self.settings.report_evidence_limit
        warning = "；".join(item.warnings[:warning_limit]) or "无明显风险"
        if self.settings.report_detail == "compact":
            return f"{name}（{record.plugin_id}） {item.total:.1f}/100｜{warning}"
        output = (
            f"{name}（{record.plugin_id}） {item.total:.1f}/100\n"
            f"需求 {item.demand:.1f}｜市场 {item.market_usage:.1f}｜兼容 {item.compatibility:.1f}｜"
            f"资源 {item.resource_fit:.1f}｜维护 {item.maintenance:.1f}｜部署 {item.deployment:.1f}\n"
            f"风险：{warning}｜画像可信度 {item.confidence:.0%}"
        )
        if self.settings.report_detail == "detailed":
            reasons = "；".join(item.reasons[: self.settings.report_evidence_limit])
            if reasons:
                output += f"\n依据：{reasons}"
        return output

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
            "说明：画像是静态风险估计，不是精确运行占用。"
        )

    @filter.command("插件推荐")
    @_qq_whitelist_required
    async def recommend(self, event: AstrMessageEvent, query: GreedyStr = ""):
        """根据服务器和群聊需求推荐插件。"""
        await self._ensure_market()
        candidates = self._find_records(str(query))
        if not candidates:
            yield event.plain_result("没有找到匹配的市场插件。")
            return
        (
            demand,
            _keywords,
            _topics,
            _model_result,
            plugin_topics,
        ) = await self._group_context(event)
        server = self._server(event)
        engine = ScoreEngine(self.records)
        results = []
        stale_profile_ids: set[str] = set()
        installed = {item.lower() for item in self._installed_plugin_names()}
        installed_profiles, unresolved_installed = self._installed_profile_state()
        for record in candidates:
            if (
                record.name.lower() in installed
                or record.plugin_id.lower() in installed
            ):
                continue
            profile = get_profile(self.index, record.plugin_id)
            if profile is None or not profile_is_current(
                profile,
                version=record.version,
                record_updated_at=record.updated_at,
            ):
                stale_profile_ids.add(record.plugin_id)
                profile = build_resource_profile(record, self.rules)
            conflicts = detect_capacity_conflicts(profile, installed_profiles, server)
            if unresolved_installed:
                conflicts.append(
                    f"{unresolved_installed} 个已安装插件缺少资源画像，容量冲突可能漏判"
                )
            topic_strength, topic_names = plugin_topics.get(record.plugin_id, (0.0, []))
            results.append(
                (
                    engine.score(
                        record,
                        profile,
                        server,
                        demand,
                        conflict_warnings=conflicts,
                        topic_match_strength=topic_strength,
                        matched_topics=topic_names,
                    ),
                    record,
                )
            )
        results.sort(key=lambda pair: pair[0].total, reverse=True)
        limit = self.settings.recommendation_limit
        fallback_limit = self.settings.recommendation_fallback_limit
        refreshed = 0
        replacement = {}
        for _score, record in results[: max(limit * 2, 10)]:
            if record.plugin_id not in stale_profile_ids or refreshed >= fallback_limit:
                continue
            profile = await self._profile_for(event, record)
            conflicts = detect_capacity_conflicts(profile, installed_profiles, server)
            if unresolved_installed:
                conflicts.append(
                    f"{unresolved_installed} 个已安装插件缺少资源画像，容量冲突可能漏判"
                )
            topic_strength, topic_names = plugin_topics.get(record.plugin_id, (0.0, []))
            replacement[record.plugin_id] = engine.score(
                record,
                profile,
                server,
                demand,
                conflict_warnings=conflicts,
                topic_match_strength=topic_strength,
                matched_topics=topic_names,
            )
            refreshed += 1
        if replacement:
            results = [
                (replacement.get(record.plugin_id, score), record)
                for score, record in results
            ]
            results.sort(key=lambda pair: pair[0].total, reverse=True)
        eligible_count = len(results)
        results = [
            item
            for item in results
            if item[0].total >= self.settings.minimum_recommendation_score
        ]
        body = [self._format_score(score, record) for score, record in results[:limit]]
        if not body:
            if eligible_count == 0:
                yield event.plain_result("匹配到的插件都已安装，没有新的候选项。")
            else:
                yield event.plain_result(
                    f"有 {eligible_count} 个未安装候选，但都低于最低推荐分 "
                    f"{self.settings.minimum_recommendation_score:g}。"
                )
            return
        yield await self._report_result(
            event, "插件推荐（高到低）\n\n" + "\n\n".join(body)
        )

    @filter.command("插件风险")
    @_qq_whitelist_required
    async def risk(self, event: AstrMessageEvent, query: GreedyStr):
        """查看一个插件的资源风险画像。"""
        await self._ensure_market()
        matches = self._find_records(str(query))
        if not matches:
            yield event.plain_result("没有找到该插件。")
            return
        record = matches[0]
        profile = await self._profile_for(event, record)
        levels = profile.levels
        yield await self._report_result(
            event,
            f"{record.display_name or record.name}\n"
            f"内存：空闲 {levels['idle_memory']} / 峰值 {levels['peak_memory']}\n"
            f"CPU：空闲 {levels['idle_cpu']} / 峰值 {levels['peak_cpu']}\n"
            f"磁盘 {levels['disk']}｜网络 {levels['network']}\n"
            f"外部进程：{', '.join(profile.external_processes) or '未发现'}\n"
            f"后台任务：{profile.background_tasks}\n"
            f"置信度：{profile.confidence:.0%}（{profile.evidence_level}）\n"
            f"依据：{'；'.join(profile.evidence[: self.settings.report_evidence_limit]) or '没有命中已知特征'}\n"
            f"未知：{'；'.join(profile.unknowns[: self.settings.report_unknown_limit]) or '无'}"
        )

    @filter.command("资源画像")
    @_qq_whitelist_required
    async def resource_profile(self, event: AstrMessageEvent, query: GreedyStr):
        """“插件风险”的同义命令，查看一个插件的资源风险画像。"""
        async for result in self.risk(event, query):
            yield result

    @filter.command("插件对比")
    @_qq_whitelist_required
    async def compare(self, event: AstrMessageEvent, first: str, second: str):
        """比较两个插件的推荐度。"""
        await self._ensure_market()
        left = self._find_records(first)
        right = self._find_records(second)
        if not left or not right:
            yield event.plain_result(
                "至少有一个插件未找到，请使用 plugin_id 或更准确的名称。"
            )
            return
        server = self._server(event)
        (
            demand,
            _keywords,
            _topics,
            _model_result,
            plugin_topics,
        ) = await self._group_context(event)
        engine = ScoreEngine(self.records)
        output = []
        for record in (left[0], right[0]):
            profile = await self._profile_for(event, record)
            topic_strength, topic_names = plugin_topics.get(record.plugin_id, (0.0, []))
            output.append(
                self._format_score(
                    engine.score(
                        record,
                        profile,
                        server,
                        demand,
                        topic_match_strength=topic_strength,
                        matched_topics=topic_names,
                    ),
                    record,
                )
            )
        yield await self._report_result(
            event, "插件对比\n\n" + "\n\n".join(output)
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
        if not self.settings.enable_group_statistics:
            yield event.plain_result("需求数据记录尚未启用，可在插件配置中开启。")
            return

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
                    "请指定需要分析的QQ群号。\n"
                    "发送 /需求分析 群号，确认后再开始分析。"
                )
                return
            target_group_id = parts[0]
            if not target_group_id.isdigit() or not 5 <= len(target_group_id) <= 20:
                yield event.plain_result(
                    "群号格式不正确，请在 /需求分析 后填写5到20位数字的QQ群号。"
                )
                return
            confirmed = " ".join(parts[1:]).strip().casefold() in confirmation_words
            if not confirmed:
                yield event.plain_result(
                    f"是否重新分析群 {target_group_id} 的最新需求？\n"
                    "这会忽略该群上一次模型分析结果并重新生成报告。\n"
                    f"发送 /需求分析 {target_group_id} 确认 开始。"
                )
                return
        else:
            target_group_id = str(event.get_group_id() or "")
            confirmed = raw_arguments.casefold() in confirmation_words
        if not is_private and not confirmed:
            yield event.plain_result(
                "是否重新分析当前群的最新需求？\n"
                "这会忽略上一次模型分析结果并重新生成报告。\n"
                "发送 /需求分析 确认 开始。"
            )
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
        phrases = extract_phrases(
            phrase_sources(messages),
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
        draft = self.analysis_drafts.get(self._event_sender_id(event))
        if draft is None:
            return None
        try:
            if not event.is_private_chat() and str(event.get_group_id() or "") != draft.group_id:
                return None
        except Exception:
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
            self.analysis_drafts.pop(self._event_sender_id(event))

    @filter.command("取消分析")
    @_qq_whitelist_required
    async def cancel_analysis(self, event: AstrMessageEvent):
        """删除当前用户的短期分析草稿。"""

        removed = self.analysis_drafts.pop(self._event_sender_id(event))
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
        format_aliases = {
            "json": "json",
            "jsonl": "jsonl",
            "txt": "txt",
            "文本": "txt",
        }
        export_format = "json"
        if tokens and tokens[-1].casefold() in format_aliases:
            export_format = format_aliases[tokens.pop().casefold()]

        is_private = event.is_private_chat()
        if is_private:
            if not tokens:
                yield event.plain_result(
                    "请指定需要导出的QQ群号。\n"
                    "示例：/导出聊天记录 123456789 1000 json"
                )
                return
            group_id = tokens.pop(0)
            if not group_id.isdigit() or not 5 <= len(group_id) <= 20:
                yield event.plain_result("群号格式不正确，请填写5到20位数字。")
                return
        else:
            group_id = str(event.get_group_id() or "")
            if not group_id:
                yield event.plain_result("当前会话不是可导出的群聊。")
                return

        requested_limit = self.settings.history_message_limit
        if tokens:
            if len(tokens) != 1 or not tokens[0].isdigit():
                yield event.plain_result(
                    "参数格式不正确。\n"
                    "群聊：/导出聊天记录 [数量] [json|jsonl|txt]\n"
                    "私聊：/导出聊天记录 群号 [数量] [json|jsonl|txt]"
                )
                return
            requested_limit = int(tokens[0])
        safe_limit = max(
            1,
            min(self.settings.history_message_limit, requested_limit),
        )

        try:
            result = await self._fetch_group_history(
                event,
                group_id=group_id,
                limit=safe_limit,
            )
        except (HistoryUnavailableError, HistoryFetchError) as exc:
            yield event.plain_result(f"无法导出聊天记录：{exc}")
            return
        if not result.messages:
            yield event.plain_result("历史接口没有返回可导出的消息。")
            return

        try:
            export_path = write_history_export(
                self.data_dir / "exports",
                group_id=group_id,
                result=result,
                export_format=export_format,
            )
        except (OSError, ValueError) as exc:
            self._log_warning("生成聊天记录文件失败：%s", exc)
            yield event.plain_result(f"聊天记录读取成功，但生成导出文件失败：{exc}")
            return

        limit_note = ""
        if requested_limit > safe_limit:
            limit_note = f"；受配置上限限制，本次最多读取 {safe_limit} 条"
        warning_note = f"；{result.warning}" if result.warning else ""
        yield event.chain_result(
            [
                Plain(
                    f"已从 {result.provider} 导出群 {group_id} 的 "
                    f"{len(result.messages)} 条消息{limit_note}{warning_note}。\n"
                    "媒体文件不会下载，导出中只保留消息段和可用引用。"
                ),
                File(name=export_path.name, file=str(export_path.resolve())),
            ]
        )

    @filter.command("插件分类")
    @_qq_whitelist_required
    async def plugin_categories(self, event: AstrMessageEvent, query: GreedyStr = ""):
        """查看市场分类统计，或查询某分类/插件。"""
        await self._ensure_market()
        value = str(query).strip().casefold()
        if not value:
            counts: Counter[str] = Counter()
            for item in self.classifications.values():
                counts.update(item.categories)
            body = [
                f"{self.taxonomy.categories.get(key, key)} [{key}]：{count}"
                for key, count in counts.most_common()
            ]
            yield await self._report_result(
                event, "插件类型总览（一个插件可属于多类）\n" + "\n".join(body)
            )
            return
        category_ids = {
            key
            for key, name in self.taxonomy.categories.items()
            if value == key.casefold() or value in name.casefold()
        }
        if category_ids:
            records = [
                self.record_by_id[plugin_id]
                for plugin_id, item in self.classifications.items()
                if set(item.categories) & category_ids
                and plugin_id in self.record_by_id
            ]
            records.sort(
                key=lambda item: (-item.download_count, -item.stars, item.plugin_id)
            )
            limit = self.settings.recommendation_limit
            body = [
                f"{item.display_name or item.name}（{item.plugin_id}）下载 {item.download_count}｜Star {item.stars}"
                for item in records[:limit]
            ]
            yield await self._report_result(
                event,
                f"分类 {', '.join(sorted(category_ids))}，共 {len(records)} 个；"
                f"显示前 {min(limit, len(records))} 个\n" + "\n".join(body),
            )
            return
        records = self._find_records(value)
        if not records:
            yield event.plain_result("没有找到该分类或插件。")
            return
        body = []
        for record in records[: self.settings.recommendation_limit]:
            item = self.classifications.get(record.plugin_id)
            if item is None:
                continue
            category_names = [
                self.taxonomy.categories.get(key, key) for key in item.categories
            ]
            body.append(
                f"{record.display_name or record.name}（{record.plugin_id}）\n"
                f"类型：{'、'.join(category_names)}｜主题：{'、'.join(item.topics) or '未识别'}\n"
                f"置信度 {item.confidence:.0%}｜依据：{'；'.join(item.evidence[:3]) or '市场分类'}"
            )
        yield await self._report_result(
            event, "插件分类结果\n\n" + "\n\n".join(body)
        )

    @filter.command("插件排行")
    @_qq_whitelist_required
    async def plugin_ranking(self, event: AstrMessageEvent, page: int = 1):
        """按当前服务器与群需求分页列出全部市场插件。"""
        await self._ensure_market()
        (
            demand,
            _keywords,
            _topics,
            _model_result,
            plugin_topics,
        ) = await self._group_context(event)
        server = self._server(event)
        engine = ScoreEngine(self.records)
        ranked = []
        installed = {value.casefold() for value in self._installed_plugin_names()}
        for record in self.records:
            profile = get_profile(self.index, record.plugin_id)
            if profile is None or not profile_is_current(
                profile,
                version=record.version,
                record_updated_at=record.updated_at,
            ):
                profile = build_resource_profile(record, self.rules)
            strength, names = plugin_topics.get(record.plugin_id, (0.0, []))
            score = engine.score(
                record,
                profile,
                server,
                demand,
                topic_match_strength=strength,
                matched_topics=names,
            )
            ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0].total, item[1].plugin_id.casefold()))
        page_size = self.settings.recommendation_limit
        total_pages = max(1, (len(ranked) + page_size - 1) // page_size)
        safe_page = max(1, min(total_pages, int(page)))
        start = (safe_page - 1) * page_size
        body = []
        for offset, (score, record) in enumerate(
            ranked[start : start + page_size], start=start + 1
        ):
            is_installed = (
                record.plugin_id.casefold() in installed
                or record.name.casefold() in installed
            )
            body.append(
                f"{offset}. {record.display_name or record.name}（{record.plugin_id}）"
                f" {score.total:.1f}/100｜下载 {record.download_count}｜Star {record.stars}"
                f"{'｜已安装' if is_installed else ''}"
            )
        yield await self._report_result(
            event,
            f"全部插件排行 第 {safe_page}/{total_pages} 页｜共 {len(ranked)} 个\n"
            + "\n".join(body)
            + f"\n发送 /插件排行 {min(total_pages, safe_page + 1)} 查看下一页。",
        )

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-1000)
    async def collect_group_stats(self, event: AstrMessageEvent):
        if not self.settings.enable_group_statistics:
            return
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
                candidates = {
                    item.plugin_id: item
                    for key in keys
                    for item in mapping.get(key, [])
                }
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
