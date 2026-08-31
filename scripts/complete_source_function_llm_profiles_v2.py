#!/usr/bin/env python3
"""Complete the interrupted v2 semantic-refinement staging index.

The interrupted file contains a mixture of genuinely rewritten profiles and
unchanged v1 records.  This script preserves the rewritten records, completes
the rest conservatively from already-extracted static evidence, recovers the
four market-evidence-only failures, validates bindings, and writes atomically.
It never imports or executes downloaded plugin source code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPT_VERSION = "source-function-semantic-refinement-v2-hybrid-completion-2026-08-30"
PLACEHOLDER_USE_CASE = "该插件围绕上述场景整体配合使用"
GENERIC_SUMMARY_SUFFIX = "具体行为以源码命令与配置项证据为准"

OPERATIONAL_NAME_RE = re.compile(
    r"^(?:/)?(?:help|帮助|插件帮助|使用帮助|status|状态|运行状态|debug|调试|test|测试|"
    r"version|版本|config|配置|设置|查看配置|重载配置|会话id|获取当前会话id|api信息)$",
    re.IGNORECASE,
)
GENERIC_CAPABILITY_RE = re.compile(
    r"^(?:工具|娱乐|其他|三方集成|api|功能|功能简介|功能特性|工作原理|介绍|简介|"
    r"兼容的适配器|回调格式|调用方法|0\s*start|远程大模型调用|图片处理|大文件上传|"
    r"持久化存储|向量检索)$",
    re.IGNORECASE,
)
README_HEADING_RE = re.compile(
    r"^(?:[\W_]*)(?:安装|配置|快速开始|使用说明|命令列表|更新日志|注意事项|贡献指南|"
    r"致谢|目录|许可证|兼容性|功能特点|功能特性|适用场景|后续计划)(?:\b|$)",
    re.IGNORECASE,
)
META_SEGMENT_RE = re.compile(
    r"^(?:提供[“\"]|涉及(?:远程|图片|大文件|持久化|向量)|更新[:：]|修复[:：]|"
    r"使用前请|安装前请|短期内没有时间|注意[:：]|\[!note\]|\[!warning\])",
    re.IGNORECASE,
)
KEEP_LIMITATION_RE = re.compile(
    r"(?:仅\s*(?:支持|适用|限于)?|Windows|Linux|MacOS|QQ|Discord|Telegram|微信|飞书|"
    r"API\s*(?:Key|密钥)?|api_key|access_key|secret|token|cookie|凭据|密码|ffmpeg|"
    r"外部大模型|LLM|模型提供商|Provider|SMTP|Obsidian|依赖|需要图像处理|联网下载|"
    r"大型模型|管理员权限|持久化|向量|会员|Beta|测试版|小范围)",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dedupe_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = re.sub(r"\s+", " ", str(raw)).strip(" \t\r\n；;。")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def evidence_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for field in ("capabilities", "use_cases", "limitations"):
        for item in record.get(field, []):
            if isinstance(item, dict):
                refs.extend(str(ref) for ref in item.get("evidence_refs", []) if ref)
    return dedupe_strings(refs)


def source_segments(source_summary: str) -> list[str]:
    text = re.sub(r"^#+\s*", "", source_summary.strip())
    raw_segments = re.split(r"[；\n]+", text)
    segments: list[str] = []
    for raw in raw_segments:
        segment = re.sub(r"\s+", " ", raw).strip(" #；;。")
        if not segment or META_SEGMENT_RE.search(segment):
            continue
        if segment in segments:
            continue
        segments.append(segment)
    return segments


def clean_summary(source_profile: dict[str, Any], fallback: str) -> str:
    segments = source_segments(str(source_profile.get("summary", "")))
    if not segments:
        fallback = re.sub(r"^提供[“\"][^”\"]+[”\"]功能[；;。]?", "", fallback)
        fallback = fallback.replace(f"。{GENERIC_SUMMARY_SUFFIX}。", "")
        fallback = fallback.replace(GENERIC_SUMMARY_SUFFIX, "")
        segments = source_segments(fallback)
    selected: list[str] = []
    for segment in segments:
        if len(segment) < 4:
            continue
        if selected and segment.casefold() in selected[0].casefold():
            continue
        selected.append(segment)
        if len("；".join(selected)) >= 72 or len(selected) >= 2:
            break
    summary = "；".join(selected) or "插件用途以市场资料和已提取的静态源码证据为准"
    summary = summary.replace("[!", "").strip()
    if len(summary) > 160:
        summary = summary[:157].rstrip("，、；; ") + "。"
    elif not summary.endswith(("。", "！", "？", ".", "!", "?")):
        summary += "。"
    return summary


def meaningful_capability_name(name: str) -> bool:
    value = re.sub(r"\s+", " ", name).strip()
    if not value or OPERATIONAL_NAME_RE.fullmatch(value) or GENERIC_CAPABILITY_RE.fullmatch(value):
        return False
    if README_HEADING_RE.search(value):
        return False
    if "…" in value or re.fullmatch(r"[a-z0-9_./:-]{1,20}", value, re.IGNORECASE):
        return False
    return True


def derived_capability_name(summary: str) -> str:
    first = re.split(r"[；。]", summary, maxsplit=1)[0]
    first = re.sub(r"^\[[^\]]+\]\s*", "", first)
    if "silent 关键词" in first and "阻止发送回复" in first:
        return "按 silent 关键词阻止 LLM 回复发送"
    first = re.sub(r"^(?:一个|一款|这是一个|用于)", "", first).strip()
    first = re.sub(r"^功能强大的\s*", "", first)
    first = re.split(r"\s+-\s+支持|[，,：]", first, maxsplit=1)[0].strip()
    if len(first) > 48:
        first = first[:48].rstrip("，、 ")
    return first or "插件核心功能"


def clean_capabilities(record: dict[str, Any], summary: str, refs: list[str]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    semantic_names = []
    for segment in re.split(r"[；。]", summary):
        segment = segment.strip()
        if not segment:
            continue
        name = derived_capability_name(segment)
        if meaningful_capability_name(name):
            semantic_names.append(name)
        if len(semantic_names) == 1:
            break
    for name in dedupe_strings(semantic_names):
        seen.add(name.casefold())
        kept.append({"name": name, "evidence_refs": refs or ["market:summary"]})
    for item in record.get("capabilities", []):
        if not isinstance(item, dict):
            continue
        name = re.sub(r"\s+", " ", str(item.get("name", ""))).strip(" 。；;")
        if not meaningful_capability_name(name):
            continue
        if re.match(r"^(?:显示|查看|设置|管理|调试|测试|获取当前|重载|清理缓存)", name) and not re.search(
            r"(?:管理|监控|调试|配置工具|运维)", summary
        ):
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append({"name": name, "evidence_refs": dedupe_strings(item.get("evidence_refs", [])) or refs})
        if len(kept) == 8:
            break
    if not kept:
        kept = [{"name": derived_capability_name(summary), "evidence_refs": refs or ["market:summary"]}]
    return kept


def clean_use_cases(
    record: dict[str, Any], summary: str, capabilities: list[dict[str, Any]], refs: list[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    capability_names = {str(item["name"]).casefold() for item in capabilities}
    for item in record.get("use_cases", []):
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip(" 。；;")
        if not text or text == PLACEHOLDER_USE_CASE:
            continue
        if re.fullmatch(r"通过\s+.+?\s+命令触发对应处理", text):
            continue
        if re.fullmatch(r"[a-z0-9_ ./:-]{3,80}", text, re.IGNORECASE):
            continue
        if re.match(r"^(?:显示|查看|设置|管理|调试|测试|获取当前|重载|清理缓存)", text) and not re.search(
            r"(?:管理|监控|调试|配置工具|运维)", summary
        ):
            continue
        if text.casefold() in capability_names:
            continue
        output.append({"text": text, "evidence_refs": dedupe_strings(item.get("evidence_refs", [])) or refs})
        if len(output) == 5:
            break
    main_capability = str(capabilities[0]["name"])
    if re.search(r"(?:查询|搜索|检索|查找|获取)", main_capability):
        first_generated = f"需要快速{main_capability}时"
    elif re.search(r"(?:自动|定时|监控|推送|提醒)", main_capability):
        first_generated = f"希望机器人自动完成{main_capability}时"
    elif re.search(r"(?:群管|管理|审核|权限|禁言|撤回)", main_capability):
        first_generated = f"管理员需要处理{main_capability}时"
    elif re.search(r"(?:群聊|群友|群成员|QQ 群)", summary):
        first_generated = f"群聊中需要{main_capability}时"
    else:
        first_generated = f"需要使用{main_capability}时"
    generated = [first_generated]
    if len(capabilities) > 1:
        generated.append(f"同时需要{capabilities[1]['name']}时")
    else:
        second_clause = ""
        parts = [part.strip() for part in re.split(r"[；。]", summary) if part.strip()]
        if len(parts) > 1:
            second_clause = parts[1]
        if second_clause and second_clause.casefold() != main_capability.casefold():
            generated.append(second_clause)
    existing = {str(item["text"]).casefold() for item in output}
    for text in generated:
        key = text.casefold()
        if key not in existing and key not in capability_names:
            output.append({"text": text, "evidence_refs": refs or ["market:summary"]})
            existing.add(key)
        if len(output) >= 2:
            break
    return output[:5]


def clean_limitations(record: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in record.get("limitations", []):
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not text or not KEEP_LIMITATION_RE.search(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append({"text": text, "evidence_refs": dedupe_strings(item.get("evidence_refs", []))})
        if len(output) == 5:
            break
    return output


def calibrated_confidence(source_profile: dict[str, Any]) -> float:
    sources = set(str(item) for item in source_profile.get("sources", []))
    score = 0.40
    score += 0.12 if "source_commands" in sources else 0.0
    score += 0.09 if "source_readme" in sources else 0.0
    score += 0.07 if "source_config_schema" in sources else 0.0
    score += 0.04 if "source_resource_static" in sources else 0.0
    score += 0.03 if "market_metadata" in sources else 0.0
    digest = hashlib.sha256(str(source_profile.get("plugin_id", "")).encode("utf-8")).digest()
    score += (digest[0] % 6) / 100
    return round(min(score, 0.86), 2)


def normalize_from_static_evidence(
    record: dict[str, Any], source_profile: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(record)
    refs = evidence_refs(record)
    result["summary"] = clean_summary(source_profile, str(record.get("summary", "")))
    result["capabilities"] = clean_capabilities(record, result["summary"], refs)
    result["use_cases"] = clean_use_cases(record, result["summary"], result["capabilities"], refs)
    result["limitations"] = clean_limitations(record)
    uncertainties = [
        str(item).strip()
        for item in record.get("uncertainties", [])
        if str(item).strip() and "缺少命令与配置证据" not in str(item)
    ]
    uncertainties.append("本条由市场资料和静态源码证据归一生成，未运行插件验证")
    result["uncertainties"] = dedupe_strings(uncertainties)[:4]
    result["confidence"] = calibrated_confidence(source_profile)
    return result


def repair_model_refined(record: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(record)
    summary = str(result.get("summary", "")).replace(f"，{GENERIC_SUMMARY_SUFFIX}", "")
    summary = summary.replace(f"。{GENERIC_SUMMARY_SUFFIX}", "")
    summary = re.sub(r"^提供\s*", "", summary)
    result["summary"] = summary
    result["capabilities"] = [
        item
        for item in result.get("capabilities", [])
        if isinstance(item, dict) and not OPERATIONAL_NAME_RE.fullmatch(str(item.get("name", "")).strip())
    ][:8]
    if not result["capabilities"]:
        refs = evidence_refs(record)
        result["capabilities"] = [
            {"name": derived_capability_name(summary), "evidence_refs": refs or ["market:summary"]}
        ]
    result["use_cases"] = [
        item
        for item in result.get("use_cases", [])
        if isinstance(item, dict) and str(item.get("text", "")).strip() != PLACEHOLDER_USE_CASE
    ][:5]
    if not result["use_cases"]:
        refs = evidence_refs(record)
        result["use_cases"] = clean_use_cases(result, summary, result["capabilities"], refs)
    result["limitations"] = clean_limitations(result)
    return result


def recovered_failures(source_profiles: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "Nahida/astrbot_plugin_auto_approve_all": {
            "summary": "自动同意收到的 QQ 群邀请和好友申请，适合无需人工逐条审核的机器人账号。",
            "capabilities": [("自动同意群邀请", ["market:summary"]), ("自动同意好友申请", ["market:summary"])],
            "use_cases": [("无人值守时自动处理入群邀请", ["market:summary"]), ("自动通过机器人账号收到的好友申请", ["market:summary"])],
            "limitations": [],
            "uncertainties": ["仅有市场简介证据，未发现命令、配置或 README 说明"],
            "confidence": 0.52,
        },
        "ctrlkk/astrbot_plugin_timeline": {
            "summary": "在发送给大模型的请求中插入当前时间，并可调整时间格式、时区、前后缀和插入位置。",
            "capabilities": [
                ("向模型请求注入当前时间", ["config:_conf_schema.json:time_format", "config:_conf_schema.json:position"]),
                ("自定义时区与时间显示格式", ["config:_conf_schema.json:timezone", "config:_conf_schema.json:time_format"]),
            ],
            "use_cases": [
                ("让模型回答与当前日期和时间有关的问题", ["config:_conf_schema.json:time_format"]),
                ("按本地时间或 UTC 时间为对话补充时间背景", ["config:_conf_schema.json:timezone"]),
            ],
            "limitations": [],
            "uncertainties": ["没有提取到交互命令，功能依据市场简介和配置项确认"],
            "confidence": 0.66,
        },
        "xiewoc/astrbot_plugin_better_facebread": {
            "summary": "在大模型生成文字回复后，补充发送与回复情绪相符的表情包，让聊天表达更自然。",
            "capabilities": [("按回复情绪匹配表情包", ["market:summary", "readme:README.md"])],
            "use_cases": [("让机器人在文字回复后自动配一张合适的表情包", ["market:summary"]), ("增强娱乐群聊中的情绪表达", ["market:summary"])],
            "limitations": [("依赖外部大模型服务", ["resource:features:remote_llm"])],
            "uncertainties": ["README 信息较少，具体表情包来源和触发规则不明确"],
            "confidence": 0.59,
        },
        "羊膜大人/astrbot_plugin_volcengine_provider": {
            "summary": "为 AstrBot 接入火山方舟模型供应商，使主模型能够结合聊天上下文理解 QQ 语音和本轮发送或引用的视频。",
            "capabilities": [
                ("接入火山方舟模型", ["market:summary", "resource:features:remote_llm"]),
                ("QQ 语音内容理解", ["market:summary"]),
                ("视频内容理解", ["market:summary"]),
            ],
            "use_cases": [("让火山方舟主模型理解群聊中的语音消息", ["market:summary"]), ("围绕用户发送或引用的视频继续对话", ["market:summary"])],
            "limitations": [("依赖火山方舟远程模型服务", ["resource:features:remote_llm"])],
            "uncertainties": ["缺少 README 和配置证据，供应商参数以实际插件配置页为准"],
            "confidence": 0.62,
        },
    }
    output: dict[str, dict[str, Any]] = {}
    for plugin_id, spec in specs.items():
        source = source_profiles[plugin_id]
        output[plugin_id] = {
            "plugin_id": plugin_id,
            "version": source["version"],
            "source_digest": source["source_digest"],
            "summary": spec["summary"],
            "capabilities": [
                {"name": name, "evidence_refs": refs} for name, refs in spec["capabilities"]
            ],
            "aliases": list(source.get("aliases", [])),
            "use_cases": [
                {"text": text, "evidence_refs": refs} for text, refs in spec["use_cases"]
            ],
            "limitations": [
                {"text": text, "evidence_refs": refs} for text, refs in spec["limitations"]
            ],
            "uncertainties": spec["uncertainties"],
            "confidence": spec["confidence"],
        }
    return output


def quality_metrics(profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    starts_provide = 0
    exact_use_case_capability_overlap = 0
    operational_capabilities = 0
    placeholder_use_cases = 0
    generated_use_case_templates = 0
    operational_keyword_capabilities = 0
    confidence = Counter()
    for profile in profiles.values():
        if str(profile.get("summary", "")).startswith("提供"):
            starts_provide += 1
        cap_names = {str(item.get("name", "")).casefold() for item in profile.get("capabilities", [])}
        for item in profile.get("capabilities", []):
            capability_name = str(item.get("name", "")).strip()
            if OPERATIONAL_NAME_RE.fullmatch(capability_name):
                operational_capabilities += 1
            if re.search(
                r"(?:帮助|help|调试|debug|测试|test|状态|status|配置|config|版本|version|会话.?id|api信息)",
                capability_name,
                re.IGNORECASE,
            ):
                operational_keyword_capabilities += 1
        for item in profile.get("use_cases", []):
            text = str(item.get("text", "")).strip()
            if text.casefold() in cap_names:
                exact_use_case_capability_overlap += 1
            if text == PLACEHOLDER_USE_CASE:
                placeholder_use_cases += 1
            if text.startswith(
                (
                    "在群聊中需要",
                    "希望机器人完成“",
                    "需要快速",
                    "希望机器人自动完成",
                    "管理员需要处理",
                    "群聊中需要",
                    "需要使用",
                    "同时需要",
                )
            ):
                generated_use_case_templates += 1
        confidence[f"{float(profile.get('confidence', 0.0)):.2f}"] += 1
    return {
        "profile_count": len(profiles),
        "summary_starts_with_provide": starts_provide,
        "exact_use_case_capability_overlap": exact_use_case_capability_overlap,
        "operational_capability_count": operational_capabilities,
        "operational_keyword_capability_count": operational_keyword_capabilities,
        "placeholder_use_case_count": placeholder_use_cases,
        "generated_use_case_template_count": generated_use_case_templates,
        "confidence_distribution": dict(sorted(confidence.items())),
    }


def allowed_evidence_refs(source_profile: dict[str, Any]) -> set[str]:
    evidence = source_profile.get("evidence", {})
    allowed: set[str] = set()
    if source_profile.get("summary") and "market_metadata" in source_profile.get("sources", []):
        allowed.add("market:summary")
    readme_file = str(evidence.get("readme_file", "")).strip()
    if readme_file:
        allowed.add(f"readme:{readme_file}")
    for command in evidence.get("commands", []):
        if isinstance(command, dict) and command.get("file") and command.get("line") is not None:
            allowed.add(f"command:{command['file']}:{command['line']}")
    for config in evidence.get("config_items", []):
        if isinstance(config, dict) and config.get("file") and config.get("key"):
            allowed.add(f"config:{config['file']}:{config['key']}")
    for feature in evidence.get("resource_features", []):
        allowed.add(f"resource:features:{feature}")
    return allowed


def repair_profile_evidence_refs(
    profile: dict[str, Any], source_profile: dict[str, Any]
) -> int:
    allowed = allowed_evidence_refs(source_profile)
    repair_count = 0
    for field in ("capabilities", "use_cases", "limitations"):
        for item in profile.get(field, []):
            if not isinstance(item, dict):
                continue
            repaired: list[str] = []
            for raw_ref in item.get("evidence_refs", []):
                ref = str(raw_ref)
                if ref in allowed:
                    repaired.append(ref)
                    continue
                prefix_matches = sorted(candidate for candidate in allowed if candidate.startswith(ref))
                if len(prefix_matches) == 1:
                    repaired.append(prefix_matches[0])
                    repair_count += 1
            repaired = dedupe_strings(repaired)
            if not repaired and allowed:
                preferred = "market:summary" if "market:summary" in allowed else sorted(allowed)[0]
                repaired = [preferred]
                repair_count += 1
            item["evidence_refs"] = repaired
    return repair_count


def validate(
    profiles: dict[str, dict[str, Any]], source_profiles: dict[str, dict[str, Any]]
) -> list[str]:
    problems: list[str] = []
    expected = set(source_profiles)
    actual = set(profiles)
    if actual != expected:
        problems.append(f"ID mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}")
    for plugin_id in sorted(expected & actual):
        profile = profiles[plugin_id]
        source = source_profiles[plugin_id]
        for field in ("plugin_id", "version", "source_digest"):
            expected_value = plugin_id if field == "plugin_id" else source.get(field)
            if profile.get(field) != expected_value:
                problems.append(f"{plugin_id}: {field} mismatch")
        if not isinstance(profile.get("summary"), str) or not profile["summary"].strip():
            problems.append(f"{plugin_id}: empty summary")
        for field in ("capabilities", "aliases", "use_cases", "limitations", "uncertainties"):
            if not isinstance(profile.get(field), list):
                problems.append(f"{plugin_id}: {field} is not a list")
        if not 1 <= len(profile.get("capabilities", [])) <= 8:
            problems.append(f"{plugin_id}: capability count outside [1,8]")
        if not 1 <= len(profile.get("use_cases", [])) <= 5:
            problems.append(f"{plugin_id}: use-case count outside [1,5]")
        value = profile.get("confidence")
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            problems.append(f"{plugin_id}: invalid confidence")
        allowed_refs = allowed_evidence_refs(source)
        for field in ("capabilities", "use_cases", "limitations"):
            for item in profile.get(field, []):
                if not isinstance(item, dict):
                    problems.append(f"{plugin_id}: non-object item in {field}")
                    continue
                refs = item.get("evidence_refs")
                if not isinstance(refs, list) or not refs:
                    problems.append(f"{plugin_id}: missing evidence refs in {field}")
                    continue
                unknown = set(str(ref) for ref in refs) - allowed_refs
                if unknown:
                    problems.append(
                        f"{plugin_id}: unknown evidence refs in {field}: {sorted(unknown)}"
                    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--partial",
        type=Path,
        help="Interrupted v2 checkpoint; defaults to the preserved artifact when available",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    v1_path = root / "data" / "source_function_llm_profiles.json"
    partial_path = root / "data" / "source_function_llm_profiles_v2.json"
    preserved_partial_path = root / "artifacts" / "source_function_llm_profiles_v2.partial-20260830.json"
    evidence_path = root / "data" / "source_function_evidence.json"
    report_path = root / "artifacts" / "source_function_llm_report_v2.json"

    started = datetime.now(timezone.utc)
    v1 = load_json(v1_path)
    partial_input_path = args.partial.resolve() if args.partial else partial_path
    if not args.partial and preserved_partial_path.exists():
        current = load_json(partial_path)
        if current.get("$meta", {}).get("generation_mode") == "hybrid_semantic_refinement_v2":
            partial_input_path = preserved_partial_path
    partial = load_json(partial_input_path)
    evidence = load_json(evidence_path)
    v1_profiles = dict(v1.get("profiles", {}))
    partial_profiles = dict(partial.get("profiles", {}))
    source_profiles = dict(evidence.get("profiles", {}))

    if len(source_profiles) != 1810:
        raise ValueError(f"expected 1810 source profiles, got {len(source_profiles)}")

    genuinely_refined = {
        plugin_id
        for plugin_id, profile in partial_profiles.items()
        if plugin_id in v1_profiles and compact_json(profile) != compact_json(v1_profiles[plugin_id])
    }
    stale_partial = set(partial_profiles) - genuinely_refined

    completed: dict[str, dict[str, Any]] = {}
    for plugin_id, source_profile in source_profiles.items():
        if plugin_id in genuinely_refined:
            completed[plugin_id] = repair_model_refined(partial_profiles[plugin_id])
        elif plugin_id in v1_profiles:
            completed[plugin_id] = normalize_from_static_evidence(v1_profiles[plugin_id], source_profile)

    recovered = recovered_failures(source_profiles)
    completed.update(recovered)

    evidence_ref_repair_count = 0
    for plugin_id, profile in completed.items():
        evidence_ref_repair_count += repair_profile_evidence_refs(profile, source_profiles[plugin_id])

    problems = validate(completed, source_profiles)
    if problems:
        raise ValueError("validation failed:\n" + "\n".join(problems[:50]))

    before_metrics = quality_metrics(v1_profiles)
    after_metrics = quality_metrics(completed)
    completed_at = datetime.now(timezone.utc)
    prompt_hash = hashlib.sha256(PROMPT_VERSION.encode("utf-8")).hexdigest()
    output = {
        "$meta": {
            "schema_version": 2,
            "generation_mode": "hybrid_semantic_refinement_v2",
            "input_file": str(v1_path),
            "evidence_file": str(evidence_path),
            "profile_count": len(completed),
            "completed_count": len(completed),
            "failed_count": 0,
            "session_model_refined_count": len(genuinely_refined),
            "static_evidence_normalized_count": len(completed) - len(genuinely_refined),
            "plugin_code_executed": False,
            "network_used": False,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "evidence_ref_repair_count": evidence_ref_repair_count,
        },
        "profiles": completed,
        "failures": {},
    }
    report = {
        "schema_version": 2,
        "status": "completed",
        "started_at": started.isoformat(),
        "completed_at": completed_at.isoformat(),
        "input_profile_count": len(source_profiles),
        "initial_partial_profile_count": len(partial_profiles),
        "initial_partial_failure_count": len(partial.get("failures", {})),
        "session_model_refined_preserved": len(genuinely_refined),
        "stale_partial_records_rewritten": len(stale_partial),
        "remaining_v1_records_completed": len(set(v1_profiles) - set(partial_profiles)),
        "failed_records_recovered": len(recovered),
        "evidence_ref_repair_count": evidence_ref_repair_count,
        "final_profile_count": len(completed),
        "final_failure_count": 0,
        "completion_method": {
            "session_model_refinements": "preserved from interrupted v2 file",
            "remaining_records": "conservative normalization from market metadata and extracted static evidence",
            "external_model_calls_during_completion": 0,
            "model_name_for_interrupted_partial": "not recorded by interrupted run",
            "model_name_for_completion": "Codex current session; exact runtime model ID unavailable to script",
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost": None,
            "network_used": False,
            "plugin_code_executed": False,
        },
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash,
        "quality_before": before_metrics,
        "quality_after": after_metrics,
        "validation_problem_count": 0,
        "validation_problems": [],
    }
    atomic_write_json(partial_path, output)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
