#!/usr/bin/env python3
"""Create a privacy-bounded demand report from an AstrBot SQLite database.

The database is opened read-only.  Raw messages, account/group identifiers and
conversation identifiers are never written to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from advisor.chat_stats import ChatStatsStore
from advisor.config import parse_config
from advisor.models import MAX_MARKET_PLUGINS, PluginRecord
from advisor.taxonomy import DEFAULT_TAXONOMY_PATH, PluginTaxonomy

MAX_DATABASE_BYTES = 4 * 1024 * 1024 * 1024
MAX_JSON_CHARS = 2 * 1024 * 1024
MAX_ROWS_PER_TABLE = 100_000
MAX_TEXT_PARTS = 256
MAX_REPORT_PLUGINS = 20
SYNTHETIC_PLATFORM = "offline-aggregate"
SYNTHETIC_GROUP = "all-user-messages"
SOURCE_TABLES = ("conversations", "platform_message_history")
USER_ROLES = {"user", "human", "member"}
NON_USER_ROLES = {"assistant", "bot", "system", "tool", "function"}


def _read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("database path is not a regular file")
    if resolved.stat().st_size > MAX_DATABASE_BYTES:
        raise ValueError("database exceeds the offline analysis size limit")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _iter_payloads(connection: sqlite3.Connection, table: str) -> Iterator[object]:
    if table not in SOURCE_TABLES or not _table_exists(connection, table):
        return
    # Table names are selected only from SOURCE_TABLES, never from user input.
    cursor = connection.execute(
        f'SELECT content FROM "{table}" LIMIT ?', (MAX_ROWS_PER_TABLE,)
    )
    for (content,) in cursor:
        if not isinstance(content, str) or not content or len(content) > MAX_JSON_CHARS:
            continue
        try:
            yield json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue


def _role(value: object) -> str:
    return str(value or "").strip().casefold()[:32]


def _text_parts(value: object) -> Iterator[str]:
    """Extract only explicit textual message fields, never metadata/IDs."""

    if isinstance(value, str):
        if value:
            yield value
        return
    if not isinstance(value, (list, tuple)):
        return
    for item in list(value)[:MAX_TEXT_PARTS]:
        if isinstance(item, str):
            if item:
                yield item
            continue
        if not isinstance(item, Mapping):
            continue
        kind = _role(item.get("type") or item.get("kind") or item.get("component"))
        if kind in {"text", "plain", "plaintext", "message", ""}:
            text = item.get("text")
            if isinstance(text, str) and text:
                yield text
            elif kind in {"text", "plain", "plaintext"}:
                content = item.get("content")
                if isinstance(content, str) and content:
                    yield content


def _component_types(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return ["text"] if isinstance(value, str) and value else []
    result: list[str] = []
    for item in list(value)[:MAX_TEXT_PARTS]:
        if isinstance(item, str):
            result.append("text")
        elif isinstance(item, Mapping):
            kind = _role(item.get("type") or item.get("kind") or item.get("component"))
            if kind:
                result.append(kind)
    return result[:64]


def _message_from_item(item: Mapping[str, object]) -> tuple[str, list[str]] | None:
    role = _role(item.get("role") or item.get("sender_role") or item.get("type"))
    if role in NON_USER_ROLES:
        return None
    if role and role not in USER_ROLES and "message" not in item:
        return None
    content = item.get("content", item.get("message", item.get("messages")))
    text = "\n".join(_text_parts(content)).strip()
    component_types = _component_types(content)
    if not text and not component_types:
        return None
    return text, component_types


def iter_user_messages(
    payload: object, *, source_table: str
) -> Iterator[tuple[str, list[str]]]:
    """Yield user-side message text/components from supported AstrBot JSON shapes."""

    if source_table == "conversations":
        rows = payload if isinstance(payload, list) else [payload]
        for item in list(rows)[:MAX_ROWS_PER_TABLE]:
            if not isinstance(item, Mapping):
                continue
            role = _role(item.get("role"))
            if role not in USER_ROLES:
                continue
            message = _message_from_item(item)
            if message is not None:
                yield message
        return

    rows: Sequence[object]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        nested = payload.get("messages")
        if isinstance(nested, list) and any(
            isinstance(item, Mapping) and "role" in item for item in nested
        ):
            rows = nested
        else:
            rows = [payload]
    else:
        return
    for item in list(rows)[:MAX_ROWS_PER_TABLE]:
        if not isinstance(item, Mapping):
            continue
        message = _message_from_item(item)
        if message is not None:
            yield message


def _load_market_snapshot(path: Path) -> list[PluginRecord]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("market snapshot exceeds size limit")
    raw = json.loads(path.read_text(encoding="utf-8"))
    plugins = raw.get("plugins") if isinstance(raw, dict) else None
    if not isinstance(plugins, dict):
        raise ValueError("market snapshot has no plugins object")
    if len(plugins) > MAX_MARKET_PLUGINS:
        raise ValueError("market snapshot has too many plugins")
    records: list[PluginRecord] = []
    for key, value in plugins.items():
        if not isinstance(value, dict):
            continue
        record = PluginRecord.from_market(str(key), value)
        if record.plugin_id:
            records.append(record)
    return records


def analyze_database(
    database: Path,
    *,
    market_snapshot: Path,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> dict[str, Any]:
    config = parse_config({})
    taxonomy = PluginTaxonomy.from_file(taxonomy_path)
    records = _load_market_snapshot(market_snapshot)
    source_rows: dict[str, int] = {}
    user_messages = 0
    cross_table_duplicates_skipped = 0
    conversation_fingerprints: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="astrbot-advisor-") as temp_dir:
        stats = ChatStatsStore(
            Path(temp_dir) / "stats.json",
            salt="offline-aggregate-v1",
            retention_days=1,
            stopwords=config.stop_words,
            min_word_length=config.word_min_length,
            ngram_max_length=config.word_ngram_max_length,
            top_n=config.word_frequency_top_n,
            keyword_min_count=config.word_min_count,
            topic_rules=config.topic_rules,
            max_text_length=config.max_message_chars,
            max_group_buckets=16,
        )
        with closing(_read_only_connection(database)) as connection:
            for table in SOURCE_TABLES:
                rows = 0
                for payload in _iter_payloads(connection, table):
                    rows += 1
                    for text, component_types in iter_user_messages(
                        payload, source_table=table
                    ):
                        fingerprint = hashlib.sha256(
                            (
                                text[: config.max_message_chars]
                                + "\0"
                                + "\0".join(component_types)
                            ).encode("utf-8", errors="replace")
                        ).hexdigest()
                        if (
                            table == "platform_message_history"
                            and fingerprint in conversation_fingerprints
                        ):
                            cross_table_duplicates_skipped += 1
                            continue
                        if table == "conversations":
                            conversation_fingerprints.add(fingerprint)
                        stats.observe(
                            platform=SYNTHETIC_PLATFORM,
                            group_id=SYNTHETIC_GROUP,
                            text=text,
                            component_types=component_types,
                        )
                        user_messages += 1
                source_rows[table] = rows

        summary = stats.summary_for(
            platform=SYNTHETIC_PLATFORM, group_id=SYNTHETIC_GROUP
        )
        keyword_counts = summary.get("top_keywords", {})
        demand_counts = summary.get("demand", {})
        sample_sufficient = user_messages >= config.minimum_messages_for_analysis
        topics = (
            taxonomy.infer_topics(keyword_counts, demand_counts)
            if sample_sufficient
            else []
        )
        matches = taxonomy.match_plugins(records, topics, limit=MAX_REPORT_PLUGINS)

    return {
        "$meta": {
            "schema_version": 1,
            "aggregate_only": True,
            "raw_messages_included": False,
            "identity_columns_included": False,
            "database_opened_read_only": True,
            "possible_cross_table_duplicates": True,
            "cross_table_exact_deduplication": True,
            "high_frequency_text_may_be_sensitive": True,
        },
        "source_rows": source_rows,
        "user_messages_analyzed": user_messages,
        "cross_table_duplicates_skipped": cross_table_duplicates_skipped,
        "sample_sufficient": sample_sufficient,
        "aggregate": {
            "messages": int(summary.get("messages", 0)),
            "text_chars": int(summary.get("text_chars", 0)),
            "media_counts": {
                name: int(summary.get(name, 0))
                for name in ("images", "videos", "audio", "files", "links")
            },
            "demand_counts": demand_counts,
            "top_terms": keyword_counts,
        },
        "topics": [item.to_dict() for item in topics],
        "matching_plugins": [item.to_dict() for item in matches],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze AstrBot user messages into privacy-bounded aggregates"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--market",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "market_snapshot.json",
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze_database(
        args.database, market_snapshot=args.market, taxonomy_path=args.taxonomy
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
