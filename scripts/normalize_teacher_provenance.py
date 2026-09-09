#!/usr/bin/env python3
"""Normalize teacher provenance fields in the packaged JSONL files."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3.8-max"
PROVIDER = "Alibaba Cloud Model Studio API"


def rewrite(name: str) -> int:
    path = ROOT / name / "records.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        has_teacher = name in {"cube_opposite", "leak_contrast"} or bool(row.get("rationale_teacher"))
        if has_teacher:
            row["teacher_model"] = MODEL
            row["teacher_provider"] = PROVIDER
            row["teacher_verified"] = False
        if name in {"cube_opposite", "leak_contrast"}:
            row["rationale_source"] = "teacher_model"
        if name == "leak_contrast":
            if not row["id"].startswith("leak_"):
                row["id"] = f"leak_{row['id']}"
            if not row["scene_id"].startswith("leak_"):
                row["scene_id"] = f"leak_{row['scene_id']}"
            row.setdefault("quality", {})["think_letter_leak"] = bool(
                re.search(r"选[ABCD]", row.get("rationale") or "")
            )
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    print(json.dumps({name: rewrite(name) for name in ("cube_opposite", "leak_contrast", "vessels")}))


if __name__ == "__main__":
    main()
