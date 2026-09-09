#!/usr/bin/env python3
"""Machine-check every record in the packaged dataset. Exit 1 on any failure."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_cube_fold import NET_LAYOUTS, fold_propagate, opposite_pairs
import generate_cube_fold_v3 as cube_v3
from generate_vessels_v2 import simulate, vessels_rationale

def square_count(rows, cols):
    return sum((rows - k + 1) * (cols - k + 1) for k in range(1, min(rows, cols) + 1))


def marked_square_count(rows, cols, rr, cc):
    total = 0
    for k in range(1, min(rows, cols) + 1):
        for r in range(1, rows - k + 2):
            for c in range(1, cols - k + 2):
                total += r <= rr < r + k and c <= cc < c + k
    return total


cube_v3.COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cube_fold", "cube_opposite", "vessels", "structure_count", "leak_contrast")
REQUIRED = {"id", "scene_id", "task", "split", "use", "image", "question", "answer", "scene", "answer_source"}
ALLOWED_USE = {"sft", "diagnostic", "probe", "contrast"}
ALLOWED_ANSWER_SOURCE = {"geometry", "physics", "combinatorics"}


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / name / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def cube_mapping(scene: dict):
    layout = NET_LAYOUTS[scene["layout_name"]]
    mapping = {}
    for color, rc in scene["cell_colors"].items():
        r, c = rc
        mapping[(int(r) - 1, int(c) - 1)] = color
    if set(mapping) != set(layout):
        raise AssertionError(f"cell_colors mismatch {scene['layout_name']}")
    return layout, mapping


VESSEL_TEXT_LEAK = re.compile(r"顶沿高度为|顶沿高度=|管道高度为|管道高度=|向[甲乙丙丁戊己]中注水")


def check_cube_visible() -> dict:
    rows = load("cube_fold")
    problems = []
    vis = [r for r in rows if r["task"] == "cube_visible_triplet"]
    scene_split = defaultdict(set)
    ans_vis = Counter()
    for r in rows:
        if r.get("use") != "sft":
            problems.append(f"use {r['id']}")
        if r["task"] != "cube_visible_triplet":
            problems.append(f"unexpected task {r['id']}")
        scene_split[r["scene_id"]].add(r["split"])
        if not (ROOT / "cube_fold" / r["image"]).exists():
            problems.append(f"missing {r['image']}")
    for sid, splits in scene_split.items():
        if len(splits) != 1:
            problems.append(f"split mismatch {sid}")
    for r in vis:
        ans_vis[r["answer"]] += 1
        layout, mapping = cube_mapping(r["scene"])
        normals = fold_propagate(layout)
        correct = cube_v3.visible_colors(normals, mapping)
        if list(r["scene"]["correct_triplet"]) != list(correct):
            problems.append(f"vis geom {r['id']}")
        letter_of = {k: list(v) for k, v in r["scene"]["options"].items()}
        if letter_of[r["answer"]] != list(correct):
            problems.append(f"vis letter {r['id']}")
        real = {frozenset(cube_v3.color_of_triplet(t, normals, mapping)) for t in cube_v3.corner_triplets(normals)}
        n_real = sum(1 for trip in letter_of.values() if frozenset(trip) in real)
        if n_real != 1:
            problems.append(f"vis not unique {r['id']} n_real={n_real}")
        n_ordered = sum(1 for trip in letter_of.values() if trip == list(correct))
        if n_ordered != 1:
            problems.append(f"vis ordered {r['id']}")
        if "可能看到" in r["question"]:
            problems.append(f"vis wording {r['id']}")
        expect = cube_v3.visible_rationale(correct, r["scene"]["options"], normals, mapping)
        if (r.get("rationale") or "") != expect:
            problems.append(f"vis rationale {r['id']}")
        if re.search(r"选[ABCD]|选项[ABCD]", r.get("rationale") or ""):
            problems.append(f"vis rationale leak {r['id']}")
        if r.get("rationale_source") != "geometry":
            problems.append(f"vis rationale_source {r['id']}")
    if len(vis) != 900:
        problems.append(f"counts vis={len(vis)}")
    if min(ans_vis.values() or [0]) < 150:
        problems.append(f"vis answer skew {dict(ans_vis)}")
    return {"n": len(rows), "problems": problems, "vis_answers": dict(ans_vis)}


def check_cube_opposite() -> dict:
    rows = load("cube_opposite")
    problems = []
    for r in rows:
        if r.get("use") != "diagnostic":
            problems.append(f"use {r['id']}")
        if not (ROOT / "cube_opposite" / r["image"]).resolve().exists():
            problems.append(f"missing {r['image']}")
        layout, mapping = cube_mapping(r["scene"])
        pairs = opposite_pairs(fold_propagate(layout))
        target = r["scene"]["target_color"]
        tcell = next(c for c, col in mapping.items() if col == target)
        opp_color = mapping[pairs[tcell]]
        if r["scene"]["options"][r["answer"]] != opp_color:
            problems.append(f"opp {r['id']}")
        if re.search(r"选[ABCD]", r.get("rationale") or ""):
            problems.append(f"opp leak {r['id']}")
        if r.get("rationale_source") != "teacher_model":
            problems.append(f"rationale_source {r['id']}")
        if r.get("teacher_model") != "qwen3.8-max":
            problems.append(f"teacher_model {r['id']}")
        if r.get("teacher_provider") != "Alibaba Cloud Model Studio API":
            problems.append(f"teacher_provider {r['id']}")
        if r.get("teacher_verified") is not False:
            problems.append(f"teacher_verified {r['id']}")
    if len(rows) != 900:
        problems.append(f"counts opp={len(rows)}")
    return {"n": len(rows), "problems": problems}


def check_leak_contrast() -> dict:
    rows = load("leak_contrast")
    problems = []
    leaked = 0
    seen_ids = set()
    seen_scenes = set()
    for r in rows:
        if r.get("id") in seen_ids:
            problems.append(f"duplicate id {r.get('id')}")
        seen_ids.add(r.get("id"))
        if r.get("scene_id") in seen_scenes:
            problems.append(f"duplicate scene {r.get('scene_id')}")
        seen_scenes.add(r.get("scene_id"))
        if r.get("use") != "contrast":
            problems.append(f"use {r['id']}")
        if r.get("split") != "train":
            problems.append(f"split {r['id']}")
        if not (ROOT / "leak_contrast" / r["image"]).exists():
            problems.append(f"missing {r['image']}")
        layout, mapping = cube_mapping(r["scene"])
        pairs = opposite_pairs(fold_propagate(layout))
        target = r["scene"]["target_color"]
        tcell = next(c for c, col in mapping.items() if col == target)
        opp_color = mapping[pairs[tcell]]
        if r["scene"]["options"][r["answer"]] != opp_color:
            problems.append(f"geometry {r['id']}")
        has_leak = bool(re.search(r"选[ABCD]", r.get("rationale") or ""))
        leaked += has_leak
        if r.get("teacher_model") != "qwen3.8-max":
            problems.append(f"teacher_model {r['id']}")
        if r.get("teacher_provider") != "Alibaba Cloud Model Studio API":
            problems.append(f"teacher_provider {r['id']}")
        if r.get("teacher_verified") is not False:
            problems.append(f"teacher_verified {r['id']}")
    if len(rows) != 300:
        problems.append(f"counts leak={len(rows)}")
    if rows and leaked / len(rows) < 0.99:
        problems.append(f"leak rate {leaked}/{len(rows)}")
    return {"n": len(rows), "leaked": leaked, "problems": problems}


def check_vessels() -> dict:
    rows = load("vessels")
    problems = []
    by = defaultdict(list)
    teacher_count = 0
    for r in rows:
        by[r["scene_id"]].append(r)
        if r.get("use") != "sft":
            problems.append(f"use {r['id']}")
        if VESSEL_TEXT_LEAK.search(r["question"] or ""):
            problems.append(f"text shortcut {r['id']}")
        if not (ROOT / "vessels" / r["image"]).exists():
            problems.append(f"missing {r['image']}")
        if r.get("rationale_teacher"):
            teacher_count += 1
            if r.get("teacher_model") != "qwen3.8-max":
                problems.append(f"teacher_model {r['id']}")
            if r.get("teacher_provider") != "Alibaba Cloud Model Studio API":
                problems.append(f"teacher_provider {r['id']}")
            if r.get("teacher_verified") is not False:
                problems.append(f"teacher_verified {r['id']}")
    for sid, group in by.items():
        sc = group[0]["scene"]
        sim = simulate(sc)
        labels = sc["labels"]
        expect = {
            "vessels_overflow": labels[sim["first_full"]],
            "vessels_never_fills": "、".join(labels[i] for i in sim["never_filled"]) if sim["never_filled"] else "没有",
            "vessels_final_level": str(sim["final_level"]),
        }
        have = {r["task"]: r for r in group}
        for task, ans in expect.items():
            rec = have[task]
            if str(rec["answer"]) != str(ans):
                problems.append(f"{rec['id']} {rec['answer']}!={ans}")
            want = vessels_rationale(sc, task)
            if (rec.get("rationale") or "") != want:
                problems.append(f"rationale {rec['id']}")
            if rec.get("rationale_source") != "physics":
                problems.append(f"rationale_source {rec['id']}")
            if re.search(r"选[ABCD]", rec.get("rationale") or ""):
                problems.append(f"rationale leak {rec['id']}")
    if teacher_count != 327:
        problems.append(f"teacher_count {teacher_count}")
    return {"n": len(rows), "scenes": len(by), "teacher": teacher_count, "problems": problems}


def struct_pred(mech, p):
    if mech == "grid_segments":
        r, c = p["rows"], p["cols"]
        return (r + 1) * math.comb(c + 1, 2) + (c + 1) * math.comb(r + 1, 2)
    if mech == "grid_squares":
        return square_count(p["rows"], p["cols"])
    if mech == "missing_corner_squares":
        return square_count(p["rows"], p["cols"]) - min(p["rows"], p["cols"])
    if mech == "point_pairs":
        return math.comb(p["points"], 2)
    if mech == "t_noncollinear":
        h, v = p["horizontal"], p["vertical"]
        total = h + v - 1
        return math.comb(total, 3) - math.comb(h, 3) - math.comb(v, 3)
    if mech == "fan_triangles":
        return math.comb(p["rays"], 2) * p.get("layers", 1)
    if mech == "fan_segment_delta":
        return p["rays"]
    if mech == "layered_fan":
        return p["layers"] * math.comb(p["rays"], 2)
    if mech == "marked_squares":
        rr, cc = p["marked"]
        return marked_square_count(p["rows"], p["cols"], rr, cc)
    if mech == "marked_rectangles":
        r0, c0 = p["rows"], p["cols"]
        rr, cc = p["marked"]
        return rr * (r0 - rr + 1) * cc * (c0 - cc + 1)
    raise KeyError(mech)


def check_structure() -> dict:
    rows = load("structure_count")
    problems = []
    train, test = set(), set()
    for r in rows:
        mech, p = r["scene"]["mechanism"], r["scene"]["params"]
        key = (mech, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items())))
        (test if r["split"] == "test" else train).add(key)
        if r.get("use") != "probe":
            problems.append(f"use {r['id']}")
        if r.get("answer_source") != "combinatorics":
            problems.append(f"answer_source {r['id']}")
        if struct_pred(mech, p) != int(r["answer"]):
            problems.append(f"formula {r['id']}")
        for rel in (r["image"], r.get("image_low")):
            if rel and not (ROOT / "structure_count" / rel).exists():
                problems.append(f"missing {rel}")
    if train & test:
        problems.append(f"split overlap {train & test}")
    return {"n": len(rows), "problems": problems}


def check_global() -> dict:
    problems = []
    seen_ids = {}
    for name in DATASETS:
        for r in load(name):
            missing = REQUIRED - set(r)
            if missing:
                problems.append(f"{name}:{r.get('id')} missing {sorted(missing)}")
            rid = r.get("id")
            if rid in seen_ids:
                problems.append(f"duplicate id {rid}: {seen_ids[rid]} and {name}")
            seen_ids[rid] = name
            if r.get("split") not in {"train", "test"}:
                problems.append(f"{name}:{rid} split {r.get('split')}")
            if r.get("use") not in ALLOWED_USE:
                problems.append(f"{name}:{rid} use {r.get('use')}")
            if r.get("answer_source") not in ALLOWED_ANSWER_SOURCE:
                problems.append(f"{name}:{rid} answer_source {r.get('answer_source')}")
            image = Path(r.get("image", ""))
            if image.is_absolute() or ".." in image.parts:
                problems.append(f"{name}:{rid} unsafe image path {image}")
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {
        "cube_records": len(load("cube_fold")),
        "cube_opposite_records": len(load("cube_opposite")),
        "vessels_records": len(load("vessels")),
        "structure_records": len(load("structure_count")),
        "leak_records": len(load("leak_contrast")),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            problems.append(f"manifest {key}={manifest.get(key)} expected {value}")
    return {"n": len(seen_ids), "problems": problems}


def main():
    report = {
        "cube_fold": check_cube_visible(),
        "cube_opposite": check_cube_opposite(),
        "leak_contrast": check_leak_contrast(),
        "vessels": check_vessels(),
        "structure_count": check_structure(),
        "global": check_global(),
    }
    n_prob = sum(len(v["problems"]) for v in report.values())
    print(json.dumps({k: {kk: vv for kk, vv in val.items() if kk != "problems" or vv} | {"n_problems": len(val["problems"])} for k, val in report.items()}, ensure_ascii=False, indent=2))
    if n_prob:
        for name, val in report.items():
            for p in val["problems"][:20]:
                print(f"{name}: {p}")
        raise SystemExit(1)
    print("VERIFY OK")


if __name__ == "__main__":
    main()
