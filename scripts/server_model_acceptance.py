from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path.name}")
    return value


def _find_by_id(rows: Any, item_id: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError("provider configuration must be a list")
    for row in rows:
        if isinstance(row, dict) and str(row.get("id") or "") == str(item_id or ""):
            return row
    raise ValueError("configured provider entry was not found")


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict)
        )
    return str(value or "")


def _json_object_text(value: str) -> str:
    cleaned = str(value or "").replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return cleaned[start : end + 1]


def run(astrbot_root: Path) -> dict[str, Any]:
    plugin_root = astrbot_root / "data" / "plugins" / "astrbot_plugin_advisor"
    sys.path.insert(0, str(plugin_root))
    from advisor.llm_fallback import (  # noqa: PLC0415
        build_context_analysis_prompt,
        parse_context_analysis,
    )

    plugin_config = _load_json(
        astrbot_root / "data" / "config" / "astrbot_plugin_advisor_config.json"
    )
    runtime_config = _load_json(astrbot_root / "data" / "cmd_config.json")
    general = plugin_config.get("general") or {}
    provider = _find_by_id(runtime_config.get("provider"), general.get("provider_id"))
    source = _find_by_id(
        runtime_config.get("provider_sources"), provider.get("provider_source_id")
    )
    raw_api_key = source.get("key")
    if isinstance(raw_api_key, list):
        api_key = next(
            (str(value).strip() for value in raw_api_key if str(value).strip()), ""
        )
    else:
        api_key = str(raw_api_key or "").strip()
    api_base = str(source.get("api_base") or "").strip()
    model = str(provider.get("model") or "").strip()
    if not api_key or not api_base or not model:
        raise ValueError("configured model endpoint is incomplete")

    payload = {
        "schema_version": 3,
        "privacy": {
            "deidentified": True,
            "group_identifier_included": False,
            "original_sender_identifiers_included": False,
        },
        "messages": [
            {
                "evidence_id": "消息0001",
                "sender": "成员001",
                "text": "大家经常发课件截图，希望机器人识别图片里的公式和文字并整理。",
                "image_ids": ["图片001"],
            },
            {
                "evidence_id": "消息0002",
                "sender": "成员002",
                "text": "还希望能查找以前上传过的资料文件，减少重复提问。",
                "image_ids": [],
            },
            {
                "evidence_id": "消息0003",
                "sender": "成员003",
                "text": "先把图片内容和群资料检索做好，不需要视频处理。",
                "image_ids": [],
            },
        ],
        "phrases": [
            {
                "phrase": "图片文字识别",
                "count": 5,
                "evidence_ids": ["消息0001", "图片001"],
                "user_edited": True,
                "kind": "phrase",
            },
            {
                "phrase": "群资料检索",
                "count": 3,
                "evidence_ids": ["消息0002"],
                "user_edited": False,
                "kind": "phrase",
            },
        ],
        "images": [
            {
                "evidence_id": "图片001",
                "message_evidence_id": "消息0001",
                "description": "本次验收临时测试图",
            }
        ],
    }
    system_prompt, user_prompt = build_context_analysis_prompt(payload)

    with tempfile.TemporaryDirectory(prefix="advisor-model-acceptance-") as directory:
        image_path = Path(directory) / "evidence.png"
        image = Image.new("RGB", (640, 360), "#F1F5FF")
        draw = ImageDraw.Draw(image)
        draw.rectangle((54, 48, 586, 312), outline="#365EDC", width=6)
        draw.text((112, 116), "OCR TEST 2026", fill="#172033")
        draw.text((112, 190), "DOCUMENT SEARCH", fill="#247A63")
        image.save(image_path, format="PNG")
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

        headers = source.get("custom_headers")
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=120,
            default_headers=headers if isinstance(headers, dict) else None,
        )
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 1600,
        }
        extra_body = provider.get("custom_extra_body")
        if isinstance(extra_body, dict) and extra_body:
            request["extra_body"] = extra_body
        started = time.perf_counter()
        vision_request: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "只读取图片中实际可见的文字。只输出JSON对象，字段必须恰好为"
                        "detected_text，不要解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "读出图中两行英文文字。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 240,
        }
        if isinstance(extra_body, dict) and extra_body:
            vision_request["extra_body"] = extra_body
        try:
            vision_response = client.chat.completions.create(**vision_request)
        except Exception as exc:
            raise RuntimeError("vision_request") from exc
        try:
            vision_text = _message_text(vision_response.choices[0].message.content)
            vision_payload = json.loads(_json_object_text(vision_text))
        except Exception as exc:
            raise RuntimeError("vision_response_validation") from exc
        detected_text = str(vision_payload.get("detected_text") or "").upper()
        if "OCR TEST 2026" not in detected_text or "DOCUMENT SEARCH" not in detected_text:
            raise RuntimeError("vision_content_validation")
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            raise RuntimeError("context_request") from exc
        elapsed = time.perf_counter() - started
        raw_text = _message_text(response.choices[0].message.content)
        try:
            parsed = parse_context_analysis(
                _json_object_text(raw_text),
                allowed_evidence_ids={
                    "消息0001",
                    "消息0002",
                    "消息0003",
                    "图片001",
                },
                evidence_text_by_id={
                    "消息0001": payload["messages"][0]["text"],
                    "消息0002": payload["messages"][1]["text"],
                    "消息0003": payload["messages"][2]["text"],
                    "图片001": "图片文字识别 OCR 文档资料",
                },
                confirmed_phrases=payload["phrases"],
                analyzed_image_ids={"图片001"},
            )
        except Exception as exc:
            raise RuntimeError("context_response_validation") from exc
        serialized = json.dumps(parsed, ensure_ascii=False).casefold()
        if "robomaster" in serialized or "机甲大师" in serialized:
            raise AssertionError("model introduced an unsupported unrelated theme")
        cited = {
            evidence
            for need in parsed["needs"]
            for evidence in need.get("evidence_ids", [])
        }
        if not cited:
            raise AssertionError("model returned no grounded evidence")
        image_cited = "图片001" in cited

    return {
        "status": "passed",
        "elapsed_seconds": round(elapsed, 2),
        "need_count": len(parsed["needs"]),
        "grounded_evidence_count": len(cited),
        "image_evidence_used": image_cited,
        "vision_probe_passed": True,
        "temporary_image_cleaned": not image_path.exists(),
        "unrelated_theme_absent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astrbot-root", type=Path, default=Path("/AstrBot"))
    args = parser.parse_args()
    try:
        result = run(args.astrbot_root)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "stage": str(exc) if isinstance(exc, RuntimeError) else "setup",
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
