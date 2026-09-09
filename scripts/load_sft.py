#!/usr/bin/env python3
"""Load the recommended SFT mix: visible cube fold + read-the-figure vessels."""
from __future__ import annotations

import json
from pathlib import Path


def load_sft(
    root: str | Path = ".",
    split: str = "train",
    subsets: tuple[str, ...] | list[str] | None = None,
) -> list[dict]:
    root = Path(root)
    rows = []
    names = tuple(subsets) if subsets is not None else ("cube_fold", "vessels")
    unknown = set(names) - {"cube_fold", "vessels"}
    if unknown:
        raise ValueError(f"not a recommended SFT subset: {sorted(unknown)}")
    for name in names:
        for line in (root / name / "records.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") == split and rec.get("use", "sft") == "sft":
                image = root / name / rec["image"]
                rec = dict(rec)
                rec["image_path"] = str(image)
                rows.append(rec)
    return rows


if __name__ == "__main__":
    data = load_sft(Path(__file__).resolve().parents[1])
    print(len(data), "sft train rows")
