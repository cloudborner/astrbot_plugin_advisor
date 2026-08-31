#!/usr/bin/env python3
"""合并一个批次的模型语义分析结果到 V3，并原子更新进度文件。

用法: python scripts/v3_merge_batch.py <batch_no> <result_json_path>

结果文件格式: {"<plugin_id>": {"summary":..., "capabilities":[{"name","evidence_refs"}],
  "aliases":[...], "use_cases":[{"text","evidence_refs"}], "limitations":[...],
  "uncertainties":[...], "confidence": float}, ...}

批次完成判定（严格模式）:
- 只有本批全部条目都成功合并时，该批次才计入 completed_batches；
- 任一条目 missing 或校验失败，该批次保持未完成，失败 plugin_id 与原因
  写入 failed_plugin_ids，failed_count 同步更新；
- current_batch 指向最早存在未完成记录的批次（按 batch_index 顺序推导）；
- 已成功合并的记录写入 V3 后即计入 completed_count（以 V3 实际存在为准，
  自愈式统计），同批已合并记录无需重复分析。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
V3_PATH = DATA / "source_function_llm_profiles_v3.json"
EV_PATH = DATA / "source_function_evidence.json"
MKT_PATH = DATA / "market_snapshot.json"
PROG_PATH = ART / "source_function_llm_progress_v3.json"
BATCH_DIR = ART / "v3_batches_v3"


def atomic_write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
        f.flush()
    tmp.replace(path)


def allowed_refs(ev_entry: dict, market_ok: bool) -> set[str]:
    allowed: set[str] = set()
    if market_ok:
        allowed.add("market:summary")
    e = ev_entry["evidence"]
    if e.get("readme_file"):
        allowed.add(f"readme:{e['readme_file']}")
    for c in e.get("commands", []):
        allowed.add(f"command:{c['file']}:{c['line']}")
    for c in e.get("config_items", []):
        allowed.add(f"config:{c['file']}:{c['key']}")
    for f in e.get("resource_features", []):
        allowed.add(f"resource:features:{f}")
    return allowed


def item_refs(item: dict) -> list[str]:
    out = []
    for r in item.get("evidence_refs", []) or []:
        if isinstance(r, str) and r:
            out.append(r)
    return out


def validate(rec: dict, allowed: set[str]) -> list[str]:
    problems: list[str] = []
    s = rec.get("summary") or ""
    if not (40 <= len(s) <= 140):
        problems.append(f"summary length {len(s)} outside 40..140")
    caps = rec.get("capabilities") or []
    ucs = rec.get("use_cases") or []
    lims = rec.get("limitations") or []
    if not (1 <= len(caps) <= 8):
        problems.append(f"capabilities count {len(caps)} outside 1..8")
    if not (2 <= len(ucs) <= 5):
        problems.append(f"use_cases count {len(ucs)} outside 2..5")
    conf = rec.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        problems.append(f"confidence invalid: {conf}")
    if not isinstance(rec.get("aliases"), list) or not rec.get("aliases"):
        problems.append("aliases missing/empty")
    for label, items in (("cap", caps), ("uc", ucs), ("lim", lims)):
        for it in items:
            text = (it.get("name") or it.get("text") or "").strip()
            if not text:
                problems.append(f"{label} item with empty name/text")
                continue
            refs = item_refs(it)
            if not refs:
                problems.append(f"{label} without refs: {text!r}")
                continue
            bad = [r for r in refs if r not in allowed]
            if bad:
                problems.append(f"{label} invalid refs {bad}: {text!r}")
    return problems


def main() -> int:
    batch_no = int(sys.argv[1])
    result_path = Path(sys.argv[2])
    with open(result_path, encoding="utf-8") as f:
        results = json.load(f)
    with open(EV_PATH, encoding="utf-8") as f:
        ev = json.load(f)["profiles"]
    with open(MKT_PATH, encoding="utf-8") as f:
        mkt = json.load(f)["plugins"]
    with open(BATCH_DIR / f"batch_{batch_no:03d}.json", encoding="utf-8") as f:
        batch_meta = json.load(f)
    batch_ids = [e["plugin_id"] for e in batch_meta["entries"]]

    with open(V3_PATH, encoding="utf-8") as f:
        v3 = json.load(f)
    with open(PROG_PATH, encoding="utf-8") as f:
        prog = json.load(f)

    merged, skipped = [], []
    for pid in batch_ids:
        rec = results.get(pid)
        if rec is None:
            reason = "missing in result file"
            skipped.append(f"{pid}: {reason}")
            prog.setdefault("failed_plugin_ids", {})[pid] = reason
            continue
        allowed = allowed_refs(
            ev[pid],
            bool((mkt.get(pid) or {}).get("desc") or (mkt.get(pid) or {}).get("short_desc")),
        )
        problems = validate(rec, allowed)
        if problems:
            reason = "; ".join(problems[:4])
            skipped.append(f"{pid}: {reason}")
            prog.setdefault("failed_plugin_ids", {})[pid] = reason
            continue
        out = {
            "plugin_id": pid,
            "version": ev[pid].get("version"),
            "source_digest": ev[pid].get("source_digest"),
            "summary": rec["summary"].strip(),
            "capabilities": [{"name": c["name"].strip(), "evidence_refs": item_refs(c)} for c in rec["capabilities"]],
            "aliases": [a.strip() for a in rec["aliases"] if str(a).strip()][:6],
            "use_cases": [{"text": u["text"].strip(), "evidence_refs": item_refs(u)} for u in rec["use_cases"]],
            "limitations": [
                {"text": limitation["text"].strip(), "evidence_refs": item_refs(limitation)}
                for limitation in rec["limitations"]
            ],
            "uncertainties": [str(u) for u in (rec.get("uncertainties") or [])][:4],
            "confidence": round(float(rec["confidence"]), 2),
        }
        v3["profiles"][pid] = out
        prog.setdefault("failed_plugin_ids", {}).pop(pid, None)
        merged.append(pid)

    # ---- 批次完成判定：本批全部条目都已存在于 V3 ----
    batch_complete = all(pid in v3["profiles"] for pid in batch_ids)
    done_batches = {int(b) for b in prog.get("completed_batches", [])}
    if batch_complete:
        done_batches.add(batch_no)
    else:
        done_batches.discard(batch_no)
    prog["completed_batches"] = sorted(done_batches)

    # ---- 自愈式统计：以 V3 实际存在 + 目标全集为准 ----
    all_target_ids: list[str] = []
    for b in sorted(prog.get("batch_index", {}).keys(), key=int):
        all_target_ids.extend(prog["batch_index"][b])
    present = [pid for pid in all_target_ids if pid in v3["profiles"]]
    present_set = set(present)

    failed = {
        pid: reason
        for pid, reason in prog.get("failed_plugin_ids", {}).items()
        if pid not in present_set
    }
    prog["failed_plugin_ids"] = failed
    prog["failed_count"] = len(failed)
    prog["completed_plugin_ids"] = sorted(present_set)
    prog["completed_count"] = len(present_set)
    prog["remaining_count"] = prog["target_count"] - len(present_set)

    # current_batch = 最早存在未完成记录的批次
    current_batch = prog["batches_total"]
    for b in sorted(prog.get("batch_index", {}).keys(), key=int):
        if any(pid not in present_set for pid in prog["batch_index"][b]):
            current_batch = int(b)
            break
    prog["current_batch"] = current_batch
    prog["updated_at"] = datetime.now(timezone.utc).isoformat()
    prog["status"] = "in_progress" if prog["remaining_count"] > 0 else "analyzing_complete"

    atomic_write_json(V3_PATH, v3)
    atomic_write_json(PROG_PATH, prog)
    print(
        json.dumps(
            {
                "batch": batch_no,
                "batch_entries": len(batch_ids),
                "merged_this_run": len(merged),
                "batch_complete": batch_complete,
                "skipped": skipped,
                "failed_total": prog["failed_count"],
                "total_completed": prog["completed_count"],
                "remaining": prog["remaining_count"],
                "current_batch": prog["current_batch"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
