from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import aiohttp


async def run(astrbot_root: Path, base_url: str) -> dict[str, Any]:
    plugin_root = astrbot_root / "data" / "plugins" / "astrbot_plugin_advisor"
    sys.path[:0] = [str(astrbot_root), str(plugin_root)]

    from advisor.chat_history import OneBotHistoryProvider  # noqa: PLC0415

    action_counts: dict[str, int] = {}
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def call_action(action: str, **params: Any) -> dict[str, Any]:
            action_counts[action] = action_counts.get(action, 0) + 1
            async with session.post(f"{base_url.rstrip('/')}/{action}", json=params) as response:
                response.raise_for_status()
                payload = await response.json(content_type=None)
            if not isinstance(payload, dict):
                raise ValueError("OneBot action did not return an object")
            if int(payload.get("retcode") or 0) != 0:
                detail = str(payload.get("message") or payload.get("wording") or "")[:160]
                raise RuntimeError(
                    f"OneBot action failed: {action}; retcode={payload.get('retcode')}; "
                    f"status={payload.get('status')}; detail={detail}"
                )
            return payload

        group_payload = await call_action("get_group_list")
        groups = group_payload.get("data")
        if not isinstance(groups, list) or not groups:
            raise RuntimeError("NapCat account has no available groups")

        provider = OneBotHistoryProvider(call_action, timeout_seconds=30, page_size=60)
        result = None
        best_result = None
        selected_group_id = ""
        failed_groups = 0
        for row in groups[:20]:
            if not isinstance(row, dict) or not row.get("group_id"):
                continue
            try:
                candidate = await provider.fetch_group_history(
                    group_id=str(row["group_id"]),
                    limit=220,
                )
            except Exception:
                failed_groups += 1
                continue
            if best_result is None or len(candidate.messages) > len(best_result.messages):
                best_result = candidate
                selected_group_id = str(row["group_id"])
            if len(candidate.messages) >= 100:
                result = candidate
                selected_group_id = str(row["group_id"])
                break
        if result is None:
            result = best_result
        if result is None or len(result.messages) < 20:
            raise RuntimeError("No tested group returned enough history messages")

        latest_payload = await call_action(
            "get_group_msg_history",
            group_id=selected_group_id,
            count=20,
            reverseOrder=False,
        )
        latest_rows = (latest_payload.get("data") or {}).get("messages") or []
        latest_sequences = {
            int(row["message_seq"])
            for row in latest_rows
            if isinstance(row, dict) and row.get("message_seq") is not None
        }
        if not latest_sequences:
            raise RuntimeError("Pagination probe did not return message sequences")
        timed_rows = [
            row
            for row in latest_rows
            if isinstance(row, dict)
            and row.get("message_seq") is not None
            and row.get("time") is not None
        ]
        if not timed_rows:
            raise RuntimeError("Pagination probe did not return timestamps")
        earliest_row = min(timed_rows, key=lambda row: int(row["time"]))
        cursor = int(earliest_row["message_seq"])
        cursor_time = int(earliest_row["time"])
        reverse_payload = await call_action(
            "get_group_msg_history",
            group_id=selected_group_id,
            message_seq=str(cursor),
            count=20,
            reverse_order=True,
            reverseOrder=True,
        )
        reverse_rows = (reverse_payload.get("data") or {}).get("messages") or []
        reverse_sequences = {
            int(row["message_seq"])
            for row in reverse_rows
            if isinstance(row, dict) and row.get("message_seq") is not None
        }
        reverse_times = {
            int(row["time"])
            for row in reverse_rows
            if isinstance(row, dict) and row.get("time") is not None
        }
        forward_payload = await call_action(
            "get_group_msg_history",
            group_id=selected_group_id,
            message_seq=str(cursor),
            count=20,
            reverse_order=False,
            reverseOrder=False,
        )
        forward_rows = (forward_payload.get("data") or {}).get("messages") or []
        forward_sequences = {
            int(row["message_seq"])
            for row in forward_rows
            if isinstance(row, dict) and row.get("message_seq") is not None
        }
        forward_times = {
            int(row["time"])
            for row in forward_rows
            if isinstance(row, dict) and row.get("time") is not None
        }
        pagination_probe = {
            "latest_count": len(latest_rows),
            "reverse_count": len(reverse_rows),
            "reverse_older_count": sum(value < cursor_time for value in reverse_times),
            "reverse_overlap_count": len(latest_sequences & reverse_sequences),
            "forward_count": len(forward_rows),
            "forward_older_count": sum(value < cursor_time for value in forward_times),
            "forward_overlap_count": len(latest_sequences & forward_sequences),
        }

    messages = result.messages
    stable_keys = [message.stable_key for message in messages]
    timestamps = [message.timestamp for message in messages if message.timestamp is not None]
    if len(stable_keys) != len(set(stable_keys)):
        raise AssertionError("history result contains duplicate stable keys")
    if not timestamps or timestamps != sorted(timestamps):
        raise AssertionError("history result is not ordered chronologically")
    if any(not message.group_id for message in messages):
        raise AssertionError("normalized history contains a missing group identifier")
    if any(len(message.text) > 8_000 for message in messages):
        raise AssertionError("normalized history contains an oversized message")

    component_types = {
        component
        for message in messages
        for component in message.component_types
    }
    return {
        "status": "passed",
        "provider": result.provider,
        "fetched": len(messages),
        "requested": result.requested,
        "reached_limit": result.reached_limit,
        "history_calls": action_counts.get("get_group_msg_history", 0),
        "version_probe_calls": action_counts.get("get_version_info", 0),
        "deduplicated": len(stable_keys) == len(set(stable_keys)),
        "chronologically_ordered": True,
        "normalized_fields_present": True,
        "component_type_count": len(component_types),
        "warning_present": bool(result.warning),
        "groups_without_cached_history": failed_groups,
        "pagination_probe": pagination_probe,
        "privacy": "no group identifiers, sender identifiers, or message text emitted",
    }


def main() -> int:
    try:
        result = asyncio.run(run(Path("/AstrBot"), "http://napcat:3000"))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "stage": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
