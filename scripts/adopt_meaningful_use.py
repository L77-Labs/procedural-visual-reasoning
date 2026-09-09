#!/usr/bin/env python3
"""Rewrite the packaged dataset toward recommended uses.

- cube_fold: visible-triplet SFT only
- cube_opposite: diagnostic opposite-face (not default train)
- vessels: questions require reading figure labels
- add use = sft | diagnostic | probe | contrast
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vessels_v2 import vision_question  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def split_cube() -> None:
    src = ROOT / "cube_fold" / "records.jsonl"
    rows = load_jsonl(src)
    if all(r["task"] == "cube_visible_triplet" for r in rows) and (ROOT / "cube_opposite" / "records.jsonl").exists():
        print("cube already split")
        return
    vis = []
    opp = []
    for row in rows:
        if row["task"] == "cube_visible_triplet":
            row["use"] = "sft"
            vis.append(row)
        elif row["task"] == "cube_opposite_face":
            row["use"] = "diagnostic"
            old = ROOT / "cube_fold" / row["image"]
            dest_dir = ROOT / "cube_opposite" / "images"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(row["image"]).name
            if old.exists() and old.resolve() != dest.resolve():
                shutil.move(str(old), str(dest))
            elif dest.exists():
                pass
            else:
                raise FileNotFoundError(old)
            row["image"] = f"images/{dest.name}"
            opp.append(row)
        else:
            raise ValueError(row["task"])
    write_jsonl(ROOT / "cube_fold" / "records.jsonl", vis)
    write_jsonl(ROOT / "cube_opposite" / "records.jsonl", opp)
    leftover = ROOT / "cube_fold" / "images" / "opposite"
    if leftover.is_dir() and not any(leftover.iterdir()):
        leftover.rmdir()


def rewrite_vessels() -> None:
    path = ROOT / "vessels" / "records.jsonl"
    rows = load_jsonl(path)
    for row in rows:
        row["use"] = "sft"
        row["question"] = vision_question(row["scene"], row["task"])
    write_jsonl(path, rows)


def tag(name: str, use: str) -> None:
    path = ROOT / name / "records.jsonl"
    rows = load_jsonl(path)
    for row in rows:
        row["use"] = use
    write_jsonl(path, rows)


def main() -> None:
    split_cube()
    rewrite_vessels()
    tag("structure_count", "probe")
    tag("leak_contrast", "contrast")
    print("adopted meaningful-use layout")


if __name__ == "__main__":
    main()
