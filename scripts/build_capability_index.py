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
) -> dict[str, Any]:
    market, records = _market_document(market_path)
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
        profiles[record.plugin_id] = {
            "plugin_id": record.plugin_id,
            "version": record.version,
            "summary": summary,
            "capabilities": capabilities,
            "aliases": [],
            "use_cases": [],
            "limitations": [],
            "sources": sources,
            "confidence": round(min(0.85, confidence), 4),
        }
    canonical = json.dumps(
        profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    market_meta = market.get("$meta") if isinstance(market.get("$meta"), dict) else {}
    return {
        "$meta": {
            "schema_version": 1,
            "generated_at": str(market_meta.get("generated_at") or ""),
            "market_version": str(market_meta.get("market_version") or ""),
            "profile_count": len(profiles),
            "generation_mode": "market_metadata_and_deterministic_taxonomy",
            "source_code_downloaded": False,
            "profiles_sha256": hashlib.sha256(canonical).hexdigest(),
        },
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = build_document(args.market, args.taxonomy)
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
