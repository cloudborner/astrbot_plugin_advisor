from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from astrbot import __version__ as ASTRBOT_VERSION
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import GreedyStr

from .advisor.chat_stats import ChatStatsStore
from .advisor.config import AdvisorConfig, parse_config
from .advisor.conflicts import detect_capacity_conflicts
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
    build_group_analysis_prompt,
    merge_assessment,
    needs_llm_fallback,
    parse_assessment,
    parse_group_analysis,
)
from .advisor.market import DEFAULT_MARKET_URL, GitHubClient, load_market
from .advisor.models import MAX_MARKET_PLUGINS, PluginRecord
from .advisor.resource_rules import build_resource_profile, load_rules
from .advisor.scoring import RecommendationScore, ScoreEngine
from .advisor.system_probe import probe_server
from .advisor.taxonomy import PluginTaxonomy, TopicMatch

PLUGIN_NAME = "astrbot_plugin_advisor"
MAX_GROUP_MODEL_PAYLOAD_BYTES = 20 * 1024


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
        logger.info(
            "插件顾问已加载：资源画像 %d 条",
            len(self.index.get("profiles", {})),
        )

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
            logger.warning("资源索引加载失败：%s", "; ".join(errors))
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
                logger.warning("插件市场请求失败，尝试本地缓存：%s", exc)
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
                logger.warning("后台 GitHub 分析失败：%s", exc)
                completed = None
            self._github_inflight_task = None
            self._github_inflight_key = None
            running = None
            if completed_key == cache_key:
                return completed

        if running is not None and self._github_inflight_key != cache_key:
            logger.warning("已有 GitHub 静态分析仍在运行，本次使用保守市场画像")
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
            logger.warning("GitHub 静态分析达到绝对期限，暂用保守画像")
            return None
        except Exception as exc:
            self._github_inflight_task = None
            self._github_inflight_key = None
            logger.warning("GitHub 回退分析失败 %s: %s", record.plugin_id, exc)
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
            logger.warning("模型辅助分析失败 %s: %s", record.plugin_id, exc)
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
        self, event: AstrMessageEvent
    ) -> tuple[
        dict[str, float],
        dict[str, int],
        list[TopicMatch],
        dict[str, Any] | None,
        dict[str, tuple[float, list[str]]],
    ]:
        platform = event.get_platform_name()
        group_id = event.get_group_id()
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
            model_result = await self._llm_group_analysis(event, topics)
        plugin_topics = self._merge_topic_maps(
            self._plugin_topic_map(topics), self._model_need_map(model_result)
        )
        return demand, keywords, topics, model_result, plugin_topics

    async def _llm_group_analysis(
        self,
        event: AstrMessageEvent,
        topic_matches: list[TopicMatch],
    ) -> dict[str, Any] | None:
        if not self.settings.enable_llm_group_summary:
            return None
        allowed_themes = {topic.topic_id for topic in self.taxonomy.topics}
        allowed_themes.update(rule.topic_id for rule in self.settings.topic_rules)
        aggregate = self.stats.model_features_for(
            platform=event.get_platform_name(), group_id=event.get_group_id()
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
            f"{self._stats_salt}\0{event.get_platform_name()}\0{event.get_group_id()}"
        ).encode("utf-8", errors="ignore")
        cache_key = hashlib.sha256(raw_cache_key).hexdigest()[:24]
        cached = self._group_model_cache.get(cache_key)
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
            logger.warning("群需求模型分析失败：%s", exc)
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
    async def health(self, event: AstrMessageEvent):
        """检查服务器资源与资源画像状态。"""
        server = self._server(event)
        meta = self.index.get("$meta", {})
        yield event.plain_result(
            "插件顾问体检\n"
            f"内存：总计 {server.total_memory_mb} MiB，可用 {server.available_memory_mb} MiB\n"
            f"Swap：总计 {server.swap_total_mb} MiB，可用 {server.swap_free_mb} MiB\n"
            f"CPU：{server.cpu_cores:g} 核｜磁盘可用 {server.disk_free_mb} MiB\n"
            f"资源画像：{len(self.index.get('profiles', {}))} 条｜生成时间 {meta.get('generated_at', '未知')}\n"
            "说明：画像是静态风险估计，不是精确运行占用。"
        )

    @filter.command("插件推荐")
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
        yield event.plain_result("插件推荐（高到低）\n\n" + "\n\n".join(body))

    @filter.command("插件风险")
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
        yield event.plain_result(
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
    async def resource_profile(self, event: AstrMessageEvent, query: GreedyStr):
        """“插件风险”的同义命令，查看一个插件的资源风险画像。"""
        async for result in self.risk(event, query):
            yield result

    @filter.command("插件对比")
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
        yield event.plain_result("插件对比\n\n" + "\n\n".join(output))

    @filter.command("群需求分析")
    async def group_analysis(self, event: AstrMessageEvent):
        """显示当前群的去身份化聚合需求。"""
        if event.is_private_chat():
            yield event.plain_result("群需求分析仅适用于群聊。")
            return
        if not self.settings.enable_group_statistics:
            yield event.plain_result("去身份化群聊统计尚未启用，可在插件配置中开启。")
            return
        await self._ensure_market()
        summary = self.stats.summary_for(
            platform=event.get_platform_name(), group_id=event.get_group_id()
        )
        demand, keywords, topics, model_result, topic_map = await self._group_context(
            event
        )
        candidates = sorted(
            (
                (strength, self.record_by_id[plugin_id], names)
                for plugin_id, (strength, names) in topic_map.items()
                if plugin_id in self.record_by_id
            ),
            key=lambda item: (
                -item[0],
                -item[1].download_count,
                -item[1].stars,
                item[1].plugin_id,
            ),
        )
        topic_text = (
            "、".join(
                f"{item.name}({item.hit_count:g}次/可信{item.confidence:.0%})"
                for item in topics[:8]
            )
            or "样本中尚未形成稳定主题"
        )
        if not self.settings.enable_topic_classification:
            topic_text = "主题分类已关闭"
        keyword_text = (
            "、".join(f"{name}×{count}" for name, count in list(keywords.items())[:15])
            or "达到隐私阈值的高频词不足"
        )
        if not self.settings.enable_word_frequency:
            keyword_text = "高频词统计已关闭"
        candidate_text = (
            "、".join(
                f"{record.display_name or record.name}({record.plugin_id})"
                for _strength, record, _names in candidates[:5]
            )
            or "暂无可解释匹配"
        )
        messages = int(summary.get("messages", 0))
        model_text = ""
        if model_result:
            emerging = "、".join(
                str(item.get("label") or "")
                for item in model_result.get("emerging_needs", [])[:5]
                if item.get("label")
            )
            model_text = (
                f"\n模型聚合判断：{model_result['summary']}"
                f"（可信度 {model_result['confidence']:.0%}）"
                f"{f'｜补充需求：{emerging}' if emerging else ''}"
            )
        elif messages < self.settings.minimum_messages_for_analysis:
            model_text = (
                f"\n样本提示：至少 {self.settings.minimum_messages_for_analysis} 条消息后才调用模型；"
                f"当前 {messages} 条，仅展示确定性统计。"
            )
        yield event.plain_result(
            "群需求去身份化统计\n"
            f"消息 {summary.get('messages', 0)}｜图片 {summary.get('images', 0)}｜"
            f"视频 {summary.get('videos', 0)}｜文件 {summary.get('files', 0)}｜链接 {summary.get('links', 0)}\n"
            f"高频词：{keyword_text}\n"
            f"主题：{topic_text}\n"
            f"主题匹配候选：{candidate_text}\n"
            f"聚合需求计数：{json.dumps(demand, ensure_ascii=False)}"
            f"{model_text}\n"
            f"保留 {summary['retention_days']} 天；不保存原文、QQ号或明文群号，也不读取平台昵称字段。"
        )

    @filter.command("插件分类")
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
            yield event.plain_result(
                "插件类型总览（一个插件可属于多类）\n" + "\n".join(body)
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
            yield event.plain_result(
                f"分类 {', '.join(sorted(category_ids))}，共 {len(records)} 个；"
                f"显示前 {min(limit, len(records))} 个\n" + "\n".join(body)
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
        yield event.plain_result("插件分类结果\n\n" + "\n\n".join(body))

    @filter.command("插件排行")
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
        yield event.plain_result(
            f"全部插件排行 第 {safe_page}/{total_pages} 页｜共 {len(ranked)} 个\n"
            + "\n".join(body)
            + f"\n发送 /插件排行 {min(total_pages, safe_page + 1)} 查看下一页。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("刷新插件数据")
    async def refresh_plugin_data(self, event: AstrMessageEvent):
        """刷新官方市场缓存；资源索引随插件版本发布。"""
        await self._ensure_market(force=True)
        yield event.plain_result(
            f"官方市场数据已刷新，共 {len(self.records)} 个插件；"
            f"资源画像 {len(self.index.get('profiles', {}))} 条。"
        )

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-1000)
    async def collect_group_stats(self, event: AstrMessageEvent):
        if not self.settings.enable_group_statistics:
            return
        try:
            component_types = [type(item).__name__ for item in event.get_messages()]
            self.stats.observe(
                platform=event.get_platform_name(),
                group_id=event.get_group_id(),
                text=event.get_message_str(),
                component_types=component_types,
            )
            self._stats_dirty += 1
            if self._stats_dirty >= self.settings.stats_flush_interval_messages:
                self.stats.save()
                self._stats_dirty = 0
        except Exception as exc:
            logger.warning("群聊去身份化统计失败：%s", exc)

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
            logger.warning("读取 AstrBot 已安装插件清单失败，使用目录回退：%s", exc)
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
        except Exception as exc:
            logger.warning("保存群聊去身份化统计失败：%s", exc)
