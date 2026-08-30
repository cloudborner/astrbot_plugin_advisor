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
DEFAULT_OUTPUT = ROOT / "data" / "plugin_capabilities.json"


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


def build_document(
    market_path: Path = DEFAULT_MARKET,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    source_evidence_path: Path | None = DEFAULT_SOURCE_EVIDENCE,
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
    taxonomy = PluginTaxonomy.from_file(taxonomy_path)
    classifications = taxonomy.classify_market(records)
    topics = {item.topic_id: item for item in taxonomy.topics}
    profiles: dict[str, Any] = {}
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
            source_names = _unique_terms(
                list(source_profile.get("sources") or []), limit=8
            )
            sources = _unique_terms([*sources, *source_names], limit=8)
            try:
                source_confidence = float(source_profile.get("confidence") or 0.0)
            except (TypeError, ValueError, OverflowError):
                source_confidence = 0.0
            confidence = max(confidence, min(0.95, max(0.0, source_confidence)))
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
            "market_taxonomy_and_local_source_function_evidence"
            if source_count
            else "market_metadata_and_deterministic_taxonomy"
        ),
        "source_code_downloaded": bool(source_count),
        "profiles_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    if source_count:
        meta["source_static_profile_count"] = source_count
        meta["plugin_code_executed"] = False
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = build_document(args.market, args.taxonomy, args.source_evidence)
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
