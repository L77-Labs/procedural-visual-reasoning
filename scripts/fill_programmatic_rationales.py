#!/usr/bin/env python3
"""Fill verified programmatic rationales for cube_fold and vessels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_cube_fold import NET_LAYOUTS, fold_propagate
import generate_cube_fold_v3 as cube_v3
from generate_vessels_v2 import vessels_rationale

cube_v3.COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]
ROOT = Path(__file__).resolve().parents[1]
LETTER_RE = __import__("re").compile(r"选[ABCD]|选项[ABCD]")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cube_mapping(scene: dict):
    layout = NET_LAYOUTS[scene["layout_name"]]
    mapping = {}
    for color, rc in scene["cell_colors"].items():
        mapping[(int(rc[0]) - 1, int(rc[1]) - 1)] = color
    return layout, mapping


def keep_teacher(row: dict) -> None:
    if row.get("rationale_teacher") or row.get("rationale_source") in ("geometry", "physics"):
        return
    old = (row.get("rationale") or "").strip()
    if old and row.get("teacher_model"):
        row["rationale_teacher"] = old


def fill_cube() -> int:
    path = ROOT / "cube_fold" / "records.jsonl"
    rows = load_jsonl(path)
    for row in rows:
        layout, mapping = cube_mapping(row["scene"])
        normals = fold_propagate(layout)
        text = cube_v3.visible_rationale(
            row["scene"]["correct_triplet"],
            row["scene"]["options"],
            normals,
            mapping,
        )
        if LETTER_RE.search(text):
            raise AssertionError(f"letter leak {row['id']}: {text}")
        row["rationale"] = text
        row["rationale_source"] = "geometry"
        row["teacher_model"] = None
    write_jsonl(path, rows)
    return len(rows)


def fill_vessels() -> tuple[int, int]:
    path = ROOT / "vessels" / "records.jsonl"
    rows = load_jsonl(path)
    kept = 0
    for row in rows:
        keep_teacher(row)
        if row.get("rationale_teacher"):
            kept += 1
        text = vessels_rationale(row["scene"], row["task"])
        if LETTER_RE.search(text):
            raise AssertionError(f"letter leak {row['id']}")
        row["rationale"] = text
        row["rationale_source"] = "physics"
        row["teacher_model"] = "qwen3.8-max" if row.get("rationale_teacher") else None
    write_jsonl(path, rows)
    return len(rows), kept


def main() -> None:
    n_cube = fill_cube()
    n_v, n_teacher = fill_vessels()
    print(f"cube_fold rationale {n_cube}")
    print(f"vessels rationale {n_v} (kept teacher {n_teacher})")


if __name__ == "__main__":
    main()
