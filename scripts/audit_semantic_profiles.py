from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.index import atomic_write_json  # noqa: E402

DEFAULT_PROFILES = ROOT / "data" / "source_function_llm_profiles_v3_reviewed.json"
DEFAULT_EVIDENCE = ROOT / "data" / "source_function_evidence.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "semantic_profile_quality_report.json"

MARKETING_TERMS = (
    "一键搞定",
    "不浪费 token",
    "不贵",
    "后悔药",
    "百宝袋",
    "神器",
    "省钱",
    "秒回",
    "说人话",
    "精美",
)
PSEUDO_CAPABILITY_PATTERNS = (
    re.compile(r"状态.{0,2}测试(?:命令|指令)?", re.IGNORECASE),
    re.compile(r"(?:API|接口).{0,2}配置", re.IGNORECASE),
    re.compile(r"触发概率控制"),
    re.compile(r"数量.{0,3}(?:超时|会话超时)控制"),
    re.compile(r"同步状态记录"),
    re.compile(r"(?:帮助|测试|调试)(?:命令|指令)$"),
)
INTERNAL_TIER_PATTERN = re.compile(r"(?:（L[34]）|\(L[34]\))", re.IGNORECASE)
UNSCOPED_REQUIREMENT_PATTERN = re.compile(r"^(?:需要|必须|依赖)(?:配置|安装|使用|提供|部署|接入|启用|\s)")

OPERATOR_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "playwright": ("playwright", "chromium"),
    "selenium": ("selenium", "webdriver"),
    "faiss-cpu": ("faiss",),
}
BINARY_CONFIG_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ffmpeg_path": ("ffmpeg",),
    "ffprobe_path": ("ffprobe",),
}

_GENERIC_BIGRAMS = {
    "是否",
    "支持",
    "功能",
    "插件",
    "设置",
    "配置",
    "启用",
    "关闭",
    "显示",
    "控制",
    "相关",
    "使用",
}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _item_text(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("name") or item.get("text") or "").strip()


def _profile_text(profile: dict[str, Any], *, include_capabilities: bool = True) -> str:
    values = [str(profile.get("summary") or "")]
    fields = ["use_cases", "limitations"]
    if include_capabilities:
        fields.append("capabilities")
    for field in fields:
        values.extend(_item_text(item) for item in profile.get(field, []) or [])
    return " ".join(values).casefold()


def _semantic_tokens(value: str) -> set[str]:
    lowered = value.casefold()
    tokens = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", lowered))
    for chunk in re.findall(r"[\u3400-\u9fff]+", lowered):
        tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens - _GENERIC_BIGRAMS


def _config_descriptions(source_profile: dict[str, Any]) -> dict[str, str]:
    evidence = source_profile.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    result: dict[str, str] = {}
    for item in evidence.get("config_items", []) or []:
        if not isinstance(item, dict) or not item.get("file") or not item.get("key"):
            continue
        ref = f"config:{item['file']}:{item['key']}"
        result[ref] = str(item.get("description") or "")
    return result


def _finding(
    plugin_id: str,
    code: str,
    field: str,
    text: str,
    detail: str,
    *,
    severity: str = "review",
) -> dict[str, str]:
    return {
        "plugin_id": plugin_id,
        "code": code,
        "severity": severity,
        "field": field,
        "text": text,
        "detail": detail,
    }


def audit_documents(
    semantic_document: dict[str, Any], source_document: dict[str, Any]
) -> dict[str, Any]:
    profiles = semantic_document.get("profiles")
    source_profiles = source_document.get("profiles")
    if not isinstance(profiles, dict) or not isinstance(source_profiles, dict):
        raise ValueError("semantic profiles and source evidence must contain profile objects")

    findings: list[dict[str, str]] = []
    for plugin_id in sorted(profiles, key=str.casefold):
        profile = profiles[plugin_id]
        source_profile = source_profiles.get(plugin_id, {})
        if not isinstance(profile, dict) or not isinstance(source_profile, dict):
            continue
        prerequisite_text = _profile_text(profile, include_capabilities=False)

        for field in ("summary", "capabilities", "use_cases", "limitations"):
            values = [str(profile.get("summary") or "")] if field == "summary" else [
                _item_text(item) for item in profile.get(field, []) or []
            ]
            for text in values:
                lowered = text.casefold()
                terms = [term for term in MARKETING_TERMS if term.casefold() in lowered]
                if terms:
                    findings.append(
                        _finding(
                            plugin_id,
                            "promotional_language",
                            field,
                            text,
                            f"中性化这些表述：{', '.join(terms)}",
                        )
                    )
                if INTERNAL_TIER_PATTERN.search(text):
                    findings.append(
                        _finding(
                            plugin_id,
                            "internal_tier_marker",
                            field,
                            text,
                            "用户可见描述不应包含内部 L1-L4 分级",
                            severity="high",
                        )
                    )

        for item in profile.get("capabilities", []) or []:
            name = _item_text(item)
            if any(pattern.search(name) for pattern in PSEUDO_CAPABILITY_PATTERNS):
                findings.append(
                    _finding(
                        plugin_id,
                        "pseudo_capability",
                        "capabilities",
                        name,
                        "检查是否把配置、状态、测试或调试入口误列为用户核心能力",
                    )
                )

        for item in profile.get("limitations", []) or []:
            text = _item_text(item)
            if UNSCOPED_REQUIREMENT_PATTERN.search(text):
                findings.append(
                    _finding(
                        plugin_id,
                        "unscoped_requirement",
                        "limitations",
                        text,
                        "确认该依赖适用于整个插件还是仅适用于部分功能",
                    )
                )

        evidence = source_profile.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        dependencies = {str(value).casefold() for value in evidence.get("dependencies", []) or []}
        for dependency, aliases in OPERATOR_DEPENDENCIES.items():
            if dependency in dependencies and not any(
                alias.casefold() in prerequisite_text for alias in aliases
            ):
                findings.append(
                    _finding(
                        plugin_id,
                        "missing_prerequisite",
                        "limitations",
                        dependency,
                        "源码依赖中存在可能需要额外运行环境准备的组件，但摘要或限制没有说明",
                    )
                )

        config_items = evidence.get("config_items", []) or []
        config_keys = {
            str(item.get("key") or "").casefold()
            for item in config_items
            if isinstance(item, dict)
        }
        for key_suffix, aliases in BINARY_CONFIG_REQUIREMENTS.items():
            if any(key.endswith(key_suffix) for key in config_keys) and not any(
                alias.casefold() in prerequisite_text for alias in aliases
            ):
                findings.append(
                    _finding(
                        plugin_id,
                        "missing_prerequisite",
                        "limitations",
                        key_suffix,
                        "发现外部可执行文件路径配置，但摘要或限制没有说明运行环境要求",
                        severity="high",
                    )
                )
        if any("cookie" in key for key in config_keys) and "cookie" not in prerequisite_text:
            findings.append(
                _finding(
                    plugin_id,
                    "missing_credential_condition",
                    "limitations",
                    "cookie",
                    "发现 Cookie 配置；检查它是必需条件、可选条件还是仅针对部分功能",
                )
            )

        descriptions = _config_descriptions(source_profile)
        for field in ("capabilities", "use_cases", "limitations"):
            for item in profile.get(field, []) or []:
                if not isinstance(item, dict):
                    continue
                refs = [str(ref) for ref in item.get("evidence_refs", []) or []]
                if not refs or any(not ref.startswith("config:") for ref in refs):
                    continue
                known = [descriptions[ref] for ref in refs if descriptions.get(ref)]
                text = _item_text(item)
                if known and not (_semantic_tokens(text) & _semantic_tokens(" ".join(known))):
                    findings.append(
                        _finding(
                            plugin_id,
                            "possible_evidence_mismatch",
                            field,
                            text,
                            f"配置证据描述与声明缺少明显词义重合：{'；'.join(known)[:240]}",
                        )
                    )

    counts = Counter(finding["code"] for finding in findings)
    severity_counts = Counter(finding["severity"] for finding in findings)
    affected = {finding["plugin_id"] for finding in findings}
    return {
        "$meta": {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "profile_count": len(profiles),
            "affected_profile_count": len(affected),
            "finding_count": len(findings),
            "heuristic_report_only": True,
        },
        "counts": dict(sorted(counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report semantic profile quality risks without executing plugin code"
    )
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="return a non-zero status when high-severity review candidates exist",
    )
    args = parser.parse_args()
    try:
        report = audit_documents(load_object(args.profiles), load_object(args.evidence))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, report)
    except Exception as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({**report["$meta"], "counts": report["counts"]}, ensure_ascii=False, indent=2))
    if args.fail_on_high and report["severity_counts"].get("high", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
