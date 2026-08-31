from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.capabilities import MAX_TERMS, normalize_summary  # noqa: E402
from advisor.index import atomic_write_json  # noqa: E402
from advisor.models import PluginRecord  # noqa: E402
from advisor.taxonomy import PluginTaxonomy  # noqa: E402

DEFAULT_MARKET = ROOT / "data" / "market_snapshot.json"
DEFAULT_TAXONOMY = ROOT / "data" / "plugin_taxonomy.json"
DEFAULT_SOURCE_EVIDENCE = ROOT / "data" / "source_function_evidence.json"
DEFAULT_SEMANTIC_PROFILES = (
    ROOT / "data" / "source_function_llm_profiles_v3_reviewed.json"
)
DEFAULT_OUTPUT = ROOT / "data" / "plugin_capabilities.json"

_CONTROL_CHARACTERS = frozenset(chr(value) for value in range(32)) - {"\t", "\n", "\r"}


def _unique_terms(values: list[object], *, limit: int = MAX_TERMS) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()[:80]
        key = text.casefold()
        if len(text) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _market_document(path: Path) -> tuple[dict[str, Any], list[PluginRecord]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("plugins"), dict):
        raise ValueError("market snapshot is invalid")
    records = [
        PluginRecord.from_market(str(key), value)
        for key, value in raw["plugins"].items()
        if isinstance(value, dict)
    ]
    return raw, records


def _allowed_evidence_refs(source_profile: dict[str, Any]) -> set[str]:
    evidence = source_profile.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    allowed: set[str] = set()
    if source_profile.get("summary") and "market_metadata" in source_profile.get(
        "sources", []
    ):
        allowed.add("market:summary")
    readme_file = str(evidence.get("readme_file") or "").strip()
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


def _strict_text(value: object, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"semantic profile {field} must be a string")
    text = " ".join(value.split()).strip()
    if not text or len(text) > maximum or any(char in _CONTROL_CHARACTERS for char in text):
        raise ValueError(f"semantic profile {field} is invalid")
    return text


def _strict_string_list(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"semantic profile {field} count is invalid")
    return [
        _strict_text(item, field=f"{field} item", maximum=160) for item in value
    ]


def _strict_grounded_items(
    value: object,
    *,
    field: str,
    text_key: str,
    minimum: int,
    maximum: int,
    allowed_refs: set[str],
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"semantic profile {field} count is invalid")
    result: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {text_key, "evidence_refs"}:
            raise ValueError(f"semantic profile {field} item is invalid")
        text = _strict_text(item[text_key], field=f"{field}.{text_key}", maximum=220)
        refs = item["evidence_refs"]
        if not isinstance(refs, list) or not refs or not all(
            isinstance(ref, str) and ref in allowed_refs for ref in refs
        ):
            raise ValueError(f"semantic profile {field} evidence is invalid")
        result.append(text)
    return result


def _validated_semantic_profile(
    plugin_id: str,
    raw: dict[str, Any],
    source_profile: dict[str, Any],
) -> dict[str, Any]:
    if raw.get("plugin_id") != plugin_id:
        raise ValueError(f"semantic profile plugin_id mismatch: {plugin_id}")
    if str(raw.get("version") or "") != str(source_profile.get("version") or ""):
        raise ValueError(f"semantic profile version mismatch: {plugin_id}")
    if str(raw.get("source_digest") or "") != str(
        source_profile.get("source_digest") or ""
    ):
        raise ValueError(f"semantic profile source digest mismatch: {plugin_id}")
    summary = _strict_text(raw.get("summary"), field="summary", maximum=140)
    if len(summary) < 40:
        raise ValueError(f"semantic profile summary is too short: {plugin_id}")
    allowed_refs = _allowed_evidence_refs(source_profile)
    capabilities = _strict_grounded_items(
        raw.get("capabilities"),
        field="capabilities",
        text_key="name",
        minimum=1,
        maximum=8,
        allowed_refs=allowed_refs,
    )
    use_cases = _strict_grounded_items(
        raw.get("use_cases"),
        field="use_cases",
        text_key="text",
        minimum=2,
        maximum=5,
        allowed_refs=allowed_refs,
    )
    limitations = _strict_grounded_items(
        raw.get("limitations"),
        field="limitations",
        text_key="text",
        minimum=0,
        maximum=8,
        allowed_refs=allowed_refs,
    )
    aliases = _strict_string_list(
        raw.get("aliases"), field="aliases", maximum=20
    )
    _strict_string_list(
        raw.get("uncertainties"), field="uncertainties", maximum=20
    )
    confidence = raw.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"semantic profile confidence is invalid: {plugin_id}")
    return {
        "summary": summary,
        "capabilities": capabilities,
        "aliases": aliases,
        "use_cases": use_cases,
        "limitations": limitations,
        "confidence": float(confidence),
    }


def build_document(
    market_path: Path = DEFAULT_MARKET,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    source_evidence_path: Path | None = DEFAULT_SOURCE_EVIDENCE,
    semantic_profiles_path: Path | None = DEFAULT_SEMANTIC_PROFILES,
) -> dict[str, Any]:
    market, records = _market_document(market_path)
    source_document: dict[str, Any] = {}
    if source_evidence_path is not None and source_evidence_path.exists():
        loaded = json.loads(source_evidence_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("profiles"), dict):
            raise ValueError("source function evidence is invalid")
        if loaded.get("$meta", {}).get("plugin_code_executed") is not False:
            raise ValueError("source function evidence lacks the no-execution guarantee")
        source_document = loaded
    source_profiles = source_document.get("profiles", {})
    source_profiles = source_profiles if isinstance(source_profiles, dict) else {}
    semantic_document: dict[str, Any] = {}
    semantic_input_sha256 = ""
    if semantic_profiles_path is not None and semantic_profiles_path.exists():
        semantic_bytes = semantic_profiles_path.read_bytes()
        loaded = json.loads(semantic_bytes.decode("utf-8"))
        if (
            not isinstance(loaded, dict)
            or loaded.get("$meta", {}).get("schema_version") != 3
            or not isinstance(loaded.get("profiles"), dict)
            or not isinstance(loaded.get("failures"), (dict, list))
        ):
            raise ValueError("reviewed semantic profiles are invalid")
        if loaded["failures"]:
            raise ValueError("reviewed semantic profiles contain unresolved failures")
        semantic_document = loaded
        semantic_input_sha256 = hashlib.sha256(semantic_bytes).hexdigest()
    semantic_profiles = semantic_document.get("profiles", {})
    semantic_profiles = semantic_profiles if isinstance(semantic_profiles, dict) else {}
    taxonomy = PluginTaxonomy.from_file(taxonomy_path)
    classifications = taxonomy.classify_market(records)
    topics = {item.topic_id: item for item in taxonomy.topics}
    profiles: dict[str, Any] = {}
    semantic_profile_count = 0
    for record in sorted(records, key=lambda item: item.plugin_id.casefold()):
        classification = classifications.get(record.plugin_id)
        topic_ids = classification.topics if classification else ()
        topic_categories = {
            category
            for topic_id in topic_ids
            if topic_id in topics
            for category in topics[topic_id].categories
        }
        category_labels = [
            taxonomy.categories.get(category, category)
            for category in sorted(topic_categories)
            if category != "other"
        ]
        capabilities = _unique_terms(
            [*record.tags, record.category, *category_labels]
        )
        if not capabilities:
            capabilities = [record.category or "其他"]
        summary = normalize_summary(record.short_desc or record.desc)
        sources = ["market_metadata"]
        if category_labels:
            sources.append("deterministic_taxonomy")
        confidence = 0.45
        if len(summary) >= 20:
            confidence += 0.15
        if len(summary) >= 60:
            confidence += 0.05
        if record.tags:
            confidence += 0.1
        if category_labels:
            confidence += 0.1
        source_profile = source_profiles.get(record.plugin_id)
        source_used = False
        if isinstance(source_profile, dict) and str(
            source_profile.get("version") or ""
        ) == str(record.version or ""):
            source_used = True
            source_names = _unique_terms(
                list(source_profile.get("sources") or []), limit=8
            )
            sources = _unique_terms([*sources, *source_names], limit=8)
            semantic_profile = semantic_profiles.get(record.plugin_id)
            semantic_matches = (
                isinstance(semantic_profile, dict)
                and str(semantic_profile.get("version") or "")
                == str(record.version or "")
                and str(semantic_profile.get("source_digest") or "")
                == str(source_profile.get("source_digest") or "")
            )
            if semantic_matches:
                semantic = _validated_semantic_profile(
                    record.plugin_id, semantic_profile, source_profile
                )
                semantic_profile_count += 1
                summary = normalize_summary(semantic["summary"])
                capabilities = _unique_terms(
                    [*capabilities, *semantic["capabilities"]]
                )
                aliases = _unique_terms(semantic["aliases"])
                use_cases = _unique_terms(semantic["use_cases"])
                limitations = _unique_terms(semantic["limitations"], limit=8)
                sources = _unique_terms([*sources, "source_llm_reviewed"], limit=8)
                confidence = max(confidence, min(0.95, semantic["confidence"]))
            else:
                source_summary = normalize_summary(source_profile.get("summary"))
                if source_summary:
                    summary = source_summary
                capabilities = _unique_terms(
                    [*capabilities, *(source_profile.get("capabilities") or [])]
                )
                aliases = _unique_terms(list(source_profile.get("aliases") or []))
                use_cases = _unique_terms(list(source_profile.get("use_cases") or []))
                limitations = _unique_terms(
                    list(source_profile.get("limitations") or []), limit=8
                )
                try:
                    source_confidence = float(source_profile.get("confidence") or 0.0)
                except (TypeError, ValueError, OverflowError):
                    source_confidence = 0.0
                confidence = max(
                    confidence, min(0.95, max(0.0, source_confidence))
                )
        else:
            aliases = []
            use_cases = []
            limitations = []
        profiles[record.plugin_id] = {
            "plugin_id": record.plugin_id,
            "version": record.version,
            "summary": summary,
            "capabilities": capabilities,
            "aliases": aliases,
            "use_cases": use_cases,
            "limitations": limitations,
            "sources": sources,
            "confidence": round(min(0.95 if source_used else 0.85, confidence), 4),
        }
    canonical = json.dumps(
        profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    market_meta = market.get("$meta") if isinstance(market.get("$meta"), dict) else {}
    source_count = sum(
        "source_readme" in profile["sources"]
        or "source_commands" in profile["sources"]
        or "source_config_schema" in profile["sources"]
        or "source_resource_static" in profile["sources"]
        for profile in profiles.values()
    )
    meta = {
        "schema_version": 1,
        "generated_at": str(market_meta.get("generated_at") or ""),
        "market_version": str(market_meta.get("market_version") or ""),
        "profile_count": len(profiles),
        "generation_mode": (
            "market_taxonomy_source_evidence_and_reviewed_semantic_profiles"
            if semantic_profile_count
            else "market_taxonomy_and_local_source_function_evidence"
            if source_count
            else "market_metadata_and_deterministic_taxonomy"
        ),
        "source_code_downloaded": bool(source_count),
        "profiles_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    if source_count:
        meta["source_static_profile_count"] = source_count
        meta["plugin_code_executed"] = False
    if semantic_profile_count:
        meta["semantic_profile_count"] = semantic_profile_count
        meta["semantic_profiles_sha256"] = semantic_input_sha256
    return {
        "$meta": meta,
        "profiles": profiles,
    }


def write_document(document: dict[str, Any], output: Path) -> None:
    atomic_write_json(output, document)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic plugin capability index from market metadata"
    )
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--source-evidence", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument(
        "--semantic-profiles", type=Path, default=DEFAULT_SEMANTIC_PROFILES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = build_document(
        args.market, args.taxonomy, args.source_evidence, args.semantic_profiles
    )
    write_document(document, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profiles": document["$meta"]["profile_count"],
                "sha256": document["$meta"]["profiles_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
