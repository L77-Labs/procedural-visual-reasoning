#!/usr/bin/env python3
"""Run repository-level checks required before publishing a release."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cube_fold", "vessels", "cube_opposite", "structure_count", "leak_contrast")
FORBIDDEN = re.compile(
    r"千问杯|SpatialViz|TallyQA|/root/autodl|/Users/l_77|remymaddox@op\.pl|"
    r"DASHSCOPE_API_KEY|(?:api|secret|access)[_-]?(?:key|token)\s*[=:]",
    re.IGNORECASE,
)


def run_verifier() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_pack.py")], cwd=ROOT, check=True)


def check_schema_contract() -> list[str]:
    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    required = set(schema["required"])
    properties = schema["properties"]
    allowed = {
        key: set(value["enum"])
        for key, value in properties.items()
        if isinstance(value, dict) and "enum" in value
    }
    problems = []
    for name in DATASETS:
        path = ROOT / name / "records.jsonl"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                problems.append(f"{path}:{lineno} missing {sorted(missing)}")
            for key, values in allowed.items():
                if key in row and row[key] not in values:
                    problems.append(f"{path}:{lineno} {key}={row[key]!r} not in schema enum")
    return problems


def check_full_json_schema() -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("NOTE: install requirements-dev.txt to enable full JSON Schema validation")
        return []

    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    problems = []
    for name in DATASETS:
        path = ROOT / name / "records.jsonl"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            for error in validator.iter_errors(row):
                location = ".".join(str(part) for part in error.absolute_path) or "<record>"
                problems.append(f"{path}:{lineno} {location}: {error.message}")
    return problems


def check_text_hygiene() -> list[str]:
    problems = []
    suffixes = {".py", ".md", ".json", ".txt", ".cff"}
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or path.suffix not in suffixes
            or path.name == "release_check.py"
        ):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                problems.append(f"{path.relative_to(ROOT)}:{lineno} forbidden release text")
    return problems


def main() -> None:
    run_verifier()
    problems = check_schema_contract() + check_full_json_schema() + check_text_hygiene()
    if problems:
        print("RELEASE CHECK FAILED", file=sys.stderr)
        print("\n".join(problems[:100]), file=sys.stderr)
        raise SystemExit(1)
    print("RELEASE CHECK OK")


if __name__ == "__main__":
    main()
