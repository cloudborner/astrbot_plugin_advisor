#!/usr/bin/env python3
"""V3 语义精炼流水线：准备阶段。

1. 计算 450 已精炼集合（checkpoint 与 V1 规范化比较）与 1360 目标集合；
2. 初始化 data/source_function_llm_profiles_v3.json（先装入 450 条保留记录并校验/修复证据引用）；
3. 生成目标集合的紧凑证据 digest 批次文件（artifacts/v3_batches_v3/batch_NNN.json），
   仅供模型阅读原始证据用，不是分析产物；
4. 初始化进度文件 artifacts/source_function_llm_progress_v3.json。

只读 V1/V2/检查点/证据/市场资料；不修改它们。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
BATCH_DIR = ART / "v3_batches_v3"
BATCH_SIZE = 25

V1_PATH = DATA / "source_function_llm_profiles.json"
CK_PATH = ART / "source_function_llm_profiles_v2.partial-20260830.json"
V3_PATH = DATA / "source_function_llm_profiles_v3.json"
EV_PATH = DATA / "source_function_evidence.json"
MKT_PATH = DATA / "market_snapshot.json"
PROG_PATH = ART / "source_function_llm_progress_v3.json"
SET_PATH = ART / "v3_set_computation.json"


def norm(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def validate_record(rec: dict, allowed: set[str]) -> list[str]:
    problems: list[str] = []
    if not rec.get("summary"):
        problems.append("missing summary")
    for key, min_n in (("capabilities", 1), ("use_cases", 2)):
        items = rec.get(key) or []
        if len(items) < min_n:
            problems.append(f"{key} count < {min_n}")
        for it in items:
            if not item_refs(it):
                problems.append(f"{key} item without refs: {it.get('name') or it.get('text')}")
    for it in rec.get("limitations") or []:
        if not item_refs(it):
            problems.append(f"limitation without refs: {it.get('text')}")
    for sec in ("capabilities", "use_cases", "limitations"):
        for it in rec.get(sec) or []:
            for r in item_refs(it):
                if r not in allowed:
                    problems.append(f"invalid ref in {sec}: {r}")
    return problems


def repair_refs(rec: dict, allowed: set[str]) -> int:
    """最小修复：剔除无效引用；若某条目引用被清空，回退到 market:summary（或首个合法引用）。"""
    fallback = ["market:summary"] if "market:summary" in allowed else sorted(allowed)[:1]
    n = 0
    for sec in ("capabilities", "use_cases", "limitations"):
        for it in rec.get(sec) or []:
            refs = item_refs(it)
            kept = [r for r in refs if r in allowed]
            if len(kept) != len(refs):
                n += len(refs) - len(kept)
            if not kept and fallback:
                kept = list(fallback)
                n += 1
            it["evidence_refs"] = kept
    return n


def main() -> int:
    with open(V1_PATH, encoding="utf-8") as f:
        v1 = json.load(f)
    with open(CK_PATH, encoding="utf-8") as f:
        ck = json.load(f)
    with open(EV_PATH, encoding="utf-8") as f:
        ev = json.load(f)
    with open(MKT_PATH, encoding="utf-8") as f:
        mkt_all = json.load(f)["plugins"]

    v1p, ckp = v1["profiles"], ck["profiles"]
    evp = ev["profiles"]
    ev_ids = set(evp.keys())

    refined, unchanged = [], []
    for pid, rec in ckp.items():
        if pid not in v1p:
            print(f"FATAL: checkpoint record {pid} not in V1")
            return 2
        (refined if norm(rec) != norm(v1p[pid]) else unchanged).append(pid)

    target = sorted(ev_ids - set(refined))
    stats = {
        "v1_profile_count": len(v1p),
        "v1_failure_ids": sorted(v1["failures"].keys()),
        "checkpoint_profile_count": len(ckp),
        "checkpoint_failure_ids": sorted(ck["failures"].keys()),
        "evidence_plugin_count": len(ev_ids),
        "refined_count": len(refined),
        "unchanged_checkpoint_count": len(unchanged),
        "target_count": len(target),
        "target_breakdown": {
            "unchanged_checkpoint": sorted(set(unchanged) & set(target)),
            "v1_failures": sorted((set(v1["failures"]) | set(ck["failures"])) & set(target)),
            "not_yet_processed_count": len(target) - len(set(unchanged) & set(target)) - len((set(v1["failures"]) | set(ck["failures"])) & set(target)),
        },
    }
    if not (stats["refined_count"] == 450 and stats["target_count"] == 1360 and len(ev_ids) == 1810):
        print("FATAL: count mismatch", json.dumps(stats, ensure_ascii=False)[:400])
        atomic_write_json(SET_PATH, {"status": "mismatch", **stats})
        return 2
    atomic_write_json(SET_PATH, {"status": "verified", **stats})
    print("set verification OK: refined=450 target=1360 evidence=1810")

    # ---- 初始化 V3 profiles：先装入 450 条保留记录 ----
    market_ok = {}
    for pid, e in evp.items():
        m = mkt_all.get(pid) or {}
        market_ok[pid] = bool(m.get("desc") or m.get("short_desc"))

    refined_recs = {}
    repair_total = 0
    refined_problems = []
    for pid in sorted(refined):
        rec = json.loads(norm(ckp[pid]))  # 深拷贝
        allowed = allowed_refs(evp[pid], market_ok[pid])
        problems = validate_record(rec, allowed)
        if problems:
            refined_problems.append({"plugin_id": pid, "problems": problems})
        n = repair_refs(rec, allowed)
        repair_total += n
        # 绑定字段以证据为准
        rec["plugin_id"] = pid
        rec["version"] = evp[pid].get("version")
        rec["source_digest"] = evp[pid].get("source_digest")
        refined_recs[pid] = rec
    print(f"refined 450 loaded; invalid-ref repairs: {repair_total}; records with problems: {len(refined_problems)}")

    meta = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "semantic analysis of source_function_evidence + market_snapshot",
        "preserved_refined_from_checkpoint": sorted(refined),
        "preserved_count": len(refined),
        "evidence_ref_repairs_in_preserved": repair_total,
        "analysis_model": "builtin:bigmodel-start-plan/GLM-5.3-Flash (ZCode agent)",
        "prompt_version": "source-function-semantic-refinement-v3-2026-08-30",
    }
    atomic_write_json(V3_PATH, {"$meta": meta, "profiles": refined_recs, "failures": {}})
    print(f"initialized {V3_PATH.name} with {len(refined_recs)} preserved records")

    # ---- digest 批次 ----
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = []
    for i in range(0, len(target), BATCH_SIZE):
        chunks = target[i : i + BATCH_SIZE]
        bno = i // BATCH_SIZE + 1
        entries = []
        for pid in chunks:
            e = evp[pid]
            ed = e["evidence"]
            m = mkt_all.get(pid) or {}
            cmds = [
                {"name": c.get("name"), "desc": (c.get("description") or "")[:160], "file": c["file"], "line": c["line"]}
                for c in ed.get("commands", [])
            ]
            cfgs = [
                {"key": c["key"], "desc": (c.get("description") or "")[:120], "file": c["file"]}
                for c in ed.get("config_items", [])
            ]
            entries.append(
                {
                    "plugin_id": pid,
                    "version": e.get("version"),
                    "market": {
                        "desc": (m.get("desc") or m.get("short_desc") or "")[:300],
                        "category": m.get("category"),
                        "tags": m.get("tags") or [],
                    },
                    "readme": {
                        "file": ed.get("readme_file"),
                        "summary": (ed.get("readme_summary") or "")[:400],
                        "headings": (ed.get("readme_headings") or [])[:12],
                    },
                    "commands": cmds,
                    "configs": cfgs,
                    "resource_features": ed.get("resource_features") or [],
                    "resource_level": ed.get("resource_level"),
                    "dependencies": ed.get("dependencies") or [],
                }
            )
        atomic_write_json(BATCH_DIR / f"batch_{bno:03d}.json", {"batch_no": bno, "entries": entries})
        batches.append({"batch_no": bno, "plugin_ids": chunks})

    print(f"digest batches written: {len(batches)} (size {BATCH_SIZE})")

    progress = {
        "schema_version": 3,
        "status": "in_progress",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target_count": len(target),
        "completed_count": 0,
        "remaining_count": len(target),
        "failed_count": 0,
        "preserved_refined_count": len(refined),
        "completed_plugin_ids": [],
        "failed_plugin_ids": {},
        "current_batch": 1,
        "batches_total": len(batches),
        "completed_batches": [],
        "set_computation": {"refined": 450, "target": 1360, "evidence": 1810, "verified": True},
        "batch_index": {str(b["batch_no"]): b["plugin_ids"] for b in batches},
    }
    atomic_write_json(PROG_PATH, progress)
    print(f"progress file initialized: {PROG_PATH.name}")
    print(json.dumps({k: stats[k] for k in ("refined_count", "target_count", "evidence_plugin_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
