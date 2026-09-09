#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连通器记录的确定性答案校验。

从 scene_params（图可见事实 + 题面设定，不含答案）确定性重建标准答案，
验证 records.jsonl 的 answer 与重建一致（证明 scene 信息完备），或验证外部解释声称的答案。

答案重建规则（与 generate_vessels_v2.build_question 完全一致，含退化分支：
q_type=never_filled_or_final 但无"接不到水"容器时退化为问 first_full）：
  first_full            -> labels[sim.first_full]
  never_filled_or_final -> "没有"（全连通）或 "、".join(labels[i])（按序号）
  final_level           -> str(sim.final_level)

用法：
  校验生成结果:
    python scripts/verify_vessels.py --jsonl outputs/vessels_20/records.jsonl
  在其他工具中复用:
    from verify_vessels import verify_claimed
    correct, standard = verify_claimed(scene_params, claimed)
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# 与生成器共用 simulate/LABELS，避免逻辑漂移
from generate_vessels_v2 import simulate, LABELS


def rebuild_answer(scene_params):
    """从 scene_params 重建标准答案（字符串，与 rec['answer'] 格式一致）"""
    sim = simulate(scene_params)
    q_type = scene_params["q_type"]
    labels = scene_params["labels"]

    # 退化分支：never_filled_or_final 但无 never_filled -> 问 first_full
    if q_type in {"first_full", "vessels_overflow"} or (q_type == "never_filled_or_final" and not sim["never_filled"]):
        return labels[sim["first_full"]]
    if q_type in {"never_filled_or_final", "vessels_never_fills"}:
        if not sim["never_filled"]:
            return "没有"
        return "、".join(labels[i] for i in sim["never_filled"])
    return str(sim["final_level"])


def verify_claimed(scene_params, claimed):
    """返回 (正确?, 标准答案)"""
    standard = rebuild_answer(scene_params)
    return claimed == standard, standard


def check_records(jsonl_path):
    records = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    n_ok = n_fail = 0
    fails = []
    for rec in records:
        sp = rec.get("scene") or rec["think"]
        standard = rebuild_answer(sp)
        if standard == rec["answer"]:
            n_ok += 1
        else:
            n_fail += 1
            fails.append((rec.get("id") or rec.get("orig_id"), standard, rec["answer"]))
    print(f"反推校验: {n_ok}/{len(records)} 一致（scene_params 信息完备）")
    if n_fail:
        print(f"!! {n_fail} 条不一致:")
        for oid, std, ans in fails:
            print(f"  {oid}: 重建={std} 记录={ans}")
        raise SystemExit(1)
    return n_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    check_records(args.jsonl)
