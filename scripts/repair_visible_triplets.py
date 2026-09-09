#!/usr/bin/env python3
"""Rebuild cube_visible_triplet items so each MCQ has exactly one real corner."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_opensource_pack import layout_from_scene, load_jsonl, render_visible_mcq, write_jsonl
from generate_cube_fold import fold_propagate
import generate_cube_fold_v3 as cube_v3

cube_v3.COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]

VISIBLE_Q = (
    "左侧是正方体展开图（六个面颜色各不相同）。将其折叠成正方体后，"
    "右侧四幅轴测图表示同一固定朝向（右、左、顶三个可见面）。"
    "请问哪一幅与折叠后该朝向实际看到的颜色一致？"
    "请从选项 A、B、C、D 中选择一个字母（选项在图内）。"
)


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "cube_fold" / "records.jsonl"
    rows = load_jsonl(path)
    n = 0
    for row in rows:
        if row["task"] != "cube_visible_triplet":
            continue
        orig = row["scene_id"]
        layout_name, layout_cells, cell_colors = layout_from_scene(row["scene"])
        normals = fold_propagate(layout_cells)
        rng = random.Random(int(hashlib.sha256(f"visible:{orig}".encode()).hexdigest(), 16) % (2**32))
        correct = cube_v3.visible_colors(normals, cell_colors)
        distractors = cube_v3.generate_distractors(correct, normals, cell_colors, rng, k=3)
        all_opts = [correct] + distractors
        rng.shuffle(all_opts)
        letter_of = dict(zip("ABCD", all_opts))
        answer = next(letter for letter, trip in letter_of.items() if trip == correct)
        out_img = root / "cube_fold" / row["image"]
        render_visible_mcq(layout_cells, cell_colors, letter_of, out_img)
        row["question"] = VISIBLE_Q
        row["choices"] = ["A", "B", "C", "D"]
        row["answer"] = answer
        row["scene"]["correct_triplet"] = list(correct)
        row["scene"]["options"] = {k: list(v) for k, v in letter_of.items()}
        row["scene"]["variant"] = "color_only"
        n += 1
    write_jsonl(path, rows)
    print(json.dumps({"repaired_visible": n, "out": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
