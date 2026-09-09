#!/usr/bin/env python3
"""Re-render pack images with updated layout. Does not change answers or options."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_opensource_pack import layout_from_scene, load_jsonl, render_visible_mcq
from generate_cube_fold import fold_propagate, render_sample
from generate_vessels_v2 import render_scene
import generate_cube_fold_v3 as cube_v3

cube_v3.COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]
ROOT = Path(__file__).resolve().parents[1]


def rerender_visible() -> int:
    rows = load_jsonl(ROOT / "cube_fold" / "records.jsonl")
    n = 0
    for row in rows:
        _, layout_cells, cell_colors = layout_from_scene(row["scene"])
        letter_of = {k: list(v) for k, v in row["scene"]["options"].items()}
        render_visible_mcq(layout_cells, cell_colors, letter_of, ROOT / "cube_fold" / row["image"])
        n += 1
    return n


def rerender_opposite(name: str) -> int:
    dest = ROOT / name
    rows = load_jsonl(dest / "records.jsonl")
    n = 0
    seen = set()
    for row in rows:
        img = dest / row["image"]
        key = str(img)
        if key in seen:
            continue
        seen.add(key)
        _, layout_cells, cell_colors = layout_from_scene(row["scene"])
        normals = fold_propagate(layout_cells)
        render_sample(row["scene"]["layout_name"], layout_cells, cell_colors, normals, str(img))
        n += 1
    return n


def rerender_vessels() -> int:
    rows = load_jsonl(ROOT / "vessels" / "records.jsonl")
    n = 0
    seen = set()
    for row in rows:
        img = ROOT / "vessels" / row["image"]
        key = str(img)
        if key in seen:
            continue
        seen.add(key)
        render_scene(row["scene"], str(img))
        n += 1
    return n


def main() -> None:
    report = {
        "visible": rerender_visible(),
        "cube_opposite": rerender_opposite("cube_opposite"),
        "leak_contrast": rerender_opposite("leak_contrast"),
        "vessels": rerender_vessels(),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
