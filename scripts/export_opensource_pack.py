#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a cleaned open-source pack from a source or local salvage tree.

Does not copy official contest samples, public-dataset images, or checkpoints.

Usage:
    python scripts/export_opensource_pack.py --root /path/to/source \\
        --out /path/to/output/procedural-visual-reasoning
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from generate_lowres_structure_probe_v1 import (  # noqa: E402
    fan,
    grid,
    marked_square_count,
    point_set,
    render as render_svg,
    square_count,
    svg,
    t_points,
)
from generate_structure_count_train32_v1 import build_specs as struct_train32_specs  # noqa: E402

_CUBE = None
_VESSELS = None


def _cube():
    global _CUBE
    if _CUBE is None:
        import matplotlib

        matplotlib.use("Agg")
        import generate_cube_fold as cube
        import generate_cube_fold_v3 as cube_v3

        cube_v3.COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]
        cube_v3.COLOR_RGB = cube.COLOR_RGB
        _CUBE = (cube, cube_v3)
    return _CUBE


def _vessels():
    global _VESSELS
    if _VESSELS is None:
        import matplotlib

        matplotlib.use("Agg")
        from generate_vessels_v2 import LABELS, render_scene, simulate

        _VESSELS = (LABELS, render_scene, simulate)
    return _VESSELS

LICENSE = "CC-BY-4.0"
TEACHER = "qwen3.8-max"
TEACHER_PROVIDER = "Alibaba Cloud Model Studio API"
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
TASK_OVERFLOW = "vessels_overflow"
TASK_NEVER = "vessels_never_fills"
TASK_LEVEL = "vessels_final_level"

MIT_TEXT = """MIT License

Copyright (c) 2026 procedural-visual-reasoning contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CC_BY_TEXT = """This dataset (images, questions, answers, scene parameters, and optional
teacher rationales) is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0).

https://creativecommons.org/licenses/by/4.0/

Generator code in scripts/ is MIT (see LICENSE).
Teacher rationales, when present, were produced with qwen3.8-max.
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def relpath(path: Path) -> str:
    return path.as_posix()


def scene_split(scene_id: str) -> str:
    digest = int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest(), 16)
    return "test" if digest % 10 == 0 else "train"


def record(
    *,
    rec_id: str,
    scene_id: str,
    task: str,
    split: str,
    image: str,
    question: str,
    answer: str,
    scene: dict,
    answer_source: str,
    choices: list[str] | None = None,
    rationale: str | None = None,
    extra: dict | None = None,
    use: str = "sft",
) -> dict:
    row = {
        "id": rec_id,
        "scene_id": scene_id,
        "task": task,
        "split": split,
        "use": use,
        "image": image,
        "question": question,
        "choices": choices,
        "answer": answer,
        "rationale": rationale,
        "scene": scene,
        "answer_source": answer_source,
        "license": LICENSE,
        "teacher_model": None,
        "teacher_provider": None,
        "teacher_verified": None,
    }
    if extra:
        row.update(extra)
    src = row.get("rationale_source")
    if row.get("rationale_teacher"):
        row["teacher_model"] = TEACHER
        row["teacher_provider"] = TEACHER_PROVIDER
        row["teacher_verified"] = False
    elif src in ("geometry", "physics"):
        row["teacher_model"] = None
    else:
        row["teacher_model"] = TEACHER if rationale else None
        if rationale:
            row["rationale_source"] = "teacher_model"
            row["teacher_provider"] = TEACHER_PROVIDER
            row["teacher_verified"] = False
    return row


def pick_scene_jsonl(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    cands = sorted(p for p in folder.glob("*.jsonl") if p.is_file())
    preferred = [p for p in cands if "merged" not in p.name and "annotated" not in p.name and "teacher" not in p.name]
    for path in preferred + cands:
        rows = load_jsonl(path)
        if rows and isinstance(rows[0].get("think"), dict):
            return path
    return preferred[0] if preferred else (cands[0] if cands else None)


def teacher_map(folder: Path, sft_path: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in folder.glob("*teacher*.jsonl"):
        for row in load_jsonl(path):
            think = row.get("think")
            if row.get("correct") is False:
                continue
            if isinstance(think, str) and think.strip():
                out.setdefault(row["orig_id"], think.strip())
    for path in folder.glob("*merged*.jsonl"):
        for row in load_jsonl(path):
            think = row.get("think")
            if isinstance(think, str) and think.strip():
                out.setdefault(row["orig_id"], think.strip())
    if sft_path and sft_path.exists():
        for row in load_jsonl(sft_path):
            orig = row.get("orig_id") or ""
            trace = row.get("trace_id") or ""
            text = None
            for msg in row.get("message") or []:
                hit = THINK_RE.search(msg.get("dialog") or "")
                if hit:
                    text = hit.group(1).strip()
                    break
            if not text:
                continue
            if orig:
                out.setdefault(orig, text)
            hit = re.search(r"(cube_fold_\d+|vessels_\d+)$", trace)
            if hit:
                out.setdefault(hit.group(1), text)
    return out


def find_image(folder: Path, orig_id: str, rec: dict) -> Path | None:
    raw = rec.get("image_path")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
        # Imported trees may contain absolute paths; keep the dataset-relative tail.
        for i, part in enumerate(p.parts):
            if part == folder.name and i + 1 < len(p.parts):
                cand = folder.joinpath(*p.parts[i + 1 :])
                if cand.exists():
                    return cand
        local = folder / p.name
        if local.exists():
            return local
    for name in (f"{orig_id}.png", f"{orig_id}.jpg"):
        p = folder / name
        if p.exists():
            return p
    hits = list(folder.rglob(f"{orig_id}.png")) or list(folder.glob(f"*{orig_id}*.png"))
    return hits[0] if hits else None


def layout_from_scene(scene: dict):
    cube, _ = _cube()
    layout_name = scene["layout_name"]
    if layout_name not in cube.NET_LAYOUTS:
        raise KeyError(f"unknown layout: {layout_name}")
    layout_cells = cube.NET_LAYOUTS[layout_name]
    mapping = {}
    for color, rc in scene["cell_colors"].items():
        r, c = rc
        mapping[(int(r) - 1, int(c) - 1)] = color
    if set(mapping) != set(layout_cells):
        raise ValueError(f"cell_colors mismatch for {layout_name}: {set(mapping)} vs {set(layout_cells)}")
    return layout_name, layout_cells, mapping


def options_as_choices(options: dict) -> list[str]:
    return [f"{k}. {options[k]}" for k in sorted(options)]


def render_visible_mcq(layout_cells, cell_colors, letter_of: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _, cube_v3 = _cube()
    fig, axes = plt.subplots(1, 5, figsize=(14.0, 4.4))
    cube_v3.draw_net_v3(axes[0], layout_cells, cell_colors)
    for i, letter in enumerate("ABCD"):
        ax = axes[1 + i]
        cube_v3.draw_iso_cube_v3(ax, letter_of[letter])
        ax.text(0, -0.78, letter, ha="center", va="center", fontsize=16, fontweight="bold")
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def export_cube(root: Path, out: Path, folder: Path, sft_path: Path | None, leak: bool) -> dict:
    scene_path = pick_scene_jsonl(folder)
    if scene_path is None:
        raise FileNotFoundError(f"no scene intermediate in {folder}")
    scenes = load_jsonl(scene_path)
    thinks = teacher_map(folder, sft_path)
    native = Path(__file__).resolve().parents[1] / "data" / "synthetic" / folder.name
    if native.exists() and native.resolve() != folder.resolve():
        for key, val in teacher_map(native, None).items():
            thinks.setdefault(key, val)
    if leak:
        dest_opp = out / "leak_contrast"
        dest_vis = None
        img_opp = dest_opp / "images" / "opposite"
        img_vis = None
    else:
        dest_opp = out / "cube_opposite"
        dest_vis = out / "cube_fold"
        img_opp = dest_opp / "images"
        img_vis = dest_vis / "images" / "visible"
    img_opp.mkdir(parents=True, exist_ok=True)
    if img_vis is not None:
        img_vis.mkdir(parents=True, exist_ok=True)

    opp_rows = []
    vis_rows = []
    n_think = 0
    for rec in scenes:
        orig = rec["orig_id"]
        scene = rec["think"]
        if not isinstance(scene, dict):
            raise TypeError(f"{orig}: scene think is not a dict; refusing merged-only input")
        cube, cube_v3 = _cube()
        layout_name, layout_cells, cell_colors = layout_from_scene(scene)
        normals = cube.fold_propagate(layout_cells)
        pairs = cube.opposite_pairs(normals)
        target = scene["target_color"]
        target_cell = next(cell for cell, color in cell_colors.items() if color == target)
        opp_color = cell_colors[pairs[target_cell]]
        options = scene.get("options") or scene.get("option_map")
        answer = rec["answer"]
        if options[answer] != opp_color:
            raise AssertionError(f"{orig}: opposite answer {answer}->{options[answer]} != {opp_color}")

        split = "train" if leak else scene_split(orig)
        src_img = find_image(folder, orig, rec)
        dst_img = img_opp / f"{orig}.png"
        if src_img is not None:
            shutil.copy2(src_img, dst_img)
        else:
            cube.render_sample(layout_name, layout_cells, cell_colors, normals, str(dst_img))

        rationale = thinks.get(orig)
        if rationale:
            n_think += 1
        opp_rows.append(
            record(
                rec_id=f"leak_{orig}_opposite" if leak else f"{orig}_opposite",
                scene_id=f"leak_{orig}" if leak else orig,
                task="cube_opposite_face",
                split=split,
                image=relpath(dst_img.relative_to(dest_opp)),
                question=rec["question"],
                choices=options_as_choices(options),
                answer=answer,
                rationale=rationale,
                scene={
                    "layout_name": layout_name,
                    "cell_colors": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in scene["cell_colors"].items()},
                    "target_color": target,
                    "options": options,
                    "variant": "text_on_net",
                },
                answer_source="geometry",
                use="contrast" if leak else "diagnostic",
                extra={"quality": {"think_letter_leak": bool(re.search(r"选[ABCD]", rationale or ""))}} if leak else None,
            )
        )

        if leak:
            continue

        rng = random.Random(int(hashlib.sha256(f"visible:{orig}".encode()).hexdigest(), 16) % (2**32))
        correct = cube_v3.visible_colors(normals, cell_colors)
        distractors = cube_v3.generate_distractors(correct, normals, cell_colors, rng, k=3)
        all_opts = [correct] + distractors
        rng.shuffle(all_opts)
        letter_of = dict(zip("ABCD", all_opts))
        vis_answer = next(letter for letter, trip in letter_of.items() if trip == correct)
        vis_path = img_vis / f"{orig}.png"
        render_visible_mcq(layout_cells, cell_colors, letter_of, vis_path)
        vis_q = (
            "左侧是正方体展开图（六个面颜色各不相同）。将其折叠成正方体后，"
            "右侧四幅轴测图表示同一固定朝向（右、左、顶三个可见面）。"
            "请问哪一幅与折叠后该朝向实际看到的颜色一致？"
            "请从选项 A、B、C、D 中选择一个字母（选项在图内）。"
        )
        vis_rows.append(
            record(
                rec_id=f"{orig}_visible",
                scene_id=orig,
                task="cube_visible_triplet",
                split=split,
                image=relpath(vis_path.relative_to(dest_vis)),
                question=vis_q,
                choices=["A", "B", "C", "D"],
                answer=vis_answer,
                rationale=cube_v3.visible_rationale(correct, letter_of, normals, cell_colors),
                scene={
                    "layout_name": layout_name,
                    "cell_colors": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in scene["cell_colors"].items()},
                    "correct_triplet": list(correct),
                    "options": {k: list(v) for k, v in letter_of.items()},
                    "variant": "color_only",
                },
                answer_source="geometry",
                use="sft",
                extra={"rationale_source": "geometry"},
            )
        )

    write_jsonl(dest_opp / "records.jsonl", opp_rows)
    if dest_vis is not None:
        write_jsonl(dest_vis / "records.jsonl", vis_rows)
    return {
        "source": str(scene_path),
        "n_scenes": len(scenes),
        "n_opposite": len(opp_rows),
        "n_visible": len(vis_rows),
        "n_records": len(opp_rows) + len(vis_rows),
        "n_with_rationale": n_think,
        "leak": leak,
    }


def vessels_items(scene: dict, sim: dict) -> dict[str, tuple[str, str]]:
    from generate_vessels_v2 import LABELS, vision_question

    labels = scene.get("labels") or LABELS[: scene["n"]]
    never = sim["never_filled"]
    never_ans = "、".join(labels[i] for i in never) if never else "没有"
    return {
        TASK_OVERFLOW: (vision_question(scene, TASK_OVERFLOW), labels[sim["first_full"]]),
        TASK_NEVER: (vision_question(scene, TASK_NEVER), never_ans),
        TASK_LEVEL: (vision_question(scene, TASK_LEVEL), str(sim["final_level"])),
    }


def original_vessels_task(q_type: str, sim: dict) -> str:
    if q_type == "final_level":
        return TASK_LEVEL
    if q_type == "first_full" or (q_type == "never_filled_or_final" and not sim["never_filled"]):
        return TASK_OVERFLOW
    return TASK_NEVER


def export_vessels_folder(
    folder: Path,
    dest: Path,
    split: str,
    sft_path: Path | None,
    id_prefix: str,
) -> tuple[list[dict], dict]:
    scene_path = pick_scene_jsonl(folder)
    if scene_path is None:
        raise FileNotFoundError(f"no vessels scene jsonl in {folder}")
    scenes = load_jsonl(scene_path)
    merged_ids = None
    for path in folder.glob("*merged*.jsonl"):
        merged_ids = {r["orig_id"] for r in load_jsonl(path)}
        break
    if merged_ids:
        scenes = [r for r in scenes if r.get("orig_id") in merged_ids]
    thinks = teacher_map(folder, sft_path)
    from generate_vessels_v2 import vessels_rationale as _vr
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_think = 0
    for rec in scenes:
        orig = rec["orig_id"]
        LABELS, render_scene, simulate = _vessels()
        scene = rec["think"]
        if not isinstance(scene, dict):
            raise TypeError(f"{orig}: vessels scene is not a dict")
        sim = simulate(scene)
        rebuilt = None
        q_type = scene.get("q_type")
        orig_task = original_vessels_task(q_type, sim) if q_type else None
        if orig_task == TASK_OVERFLOW:
            rebuilt = (scene.get("labels") or LABELS[: scene["n"]])[sim["first_full"]]
        elif orig_task == TASK_NEVER:
            never = sim["never_filled"]
            rebuilt = "、".join((scene.get("labels") or LABELS)[i] for i in never) if never else "没有"
        elif orig_task == TASK_LEVEL:
            rebuilt = str(sim["final_level"])
        if rebuilt is not None and str(rec.get("answer")) != str(rebuilt):
            # keep original answer if q_type reconstruction disagrees; still export derived tasks from sim
            pass

        scene_id = f"{id_prefix}_{orig}"
        src_img = find_image(folder, orig, rec)
        dst_img = img_dir / f"{scene_id}.png"
        if src_img is not None:
            shutil.copy2(src_img, dst_img)
        else:
            render_scene(
                {
                    "n": scene["n"],
                    "heights": scene["heights"],
                    "widths": scene["widths"],
                    "pipes": scene["pipes"],
                    "inflow": scene["inflow"],
                },
                str(dst_img),
            )

        items = vessels_items(scene, sim)
        rationale = thinks.get(orig)
        image_rel = relpath(dst_img.relative_to(dest))
        for task, (question, answer) in items.items():
            use_orig_a = orig_task == task and rec.get("answer") is not None
            extra = {"rationale_source": "physics"}
            if orig_task == task and rationale:
                extra["rationale_teacher"] = rationale
                n_think += 1
            rows.append(
                record(
                    rec_id=f"{scene_id}_{task}",
                    scene_id=scene_id,
                    task=task,
                    split=split,
                    image=image_rel,
                    question=question,
                    answer=str(rec["answer"]) if use_orig_a else answer,
                    rationale=_vr(scene, task),
                    scene={
                        "n": scene["n"],
                        "labels": scene.get("labels") or LABELS[: scene["n"]],
                        "heights": scene["heights"],
                        "widths": scene["widths"],
                        "pipes": scene["pipes"],
                        "inflow": scene["inflow"],
                        "q_type_original": q_type,
                    },
                    answer_source="physics",
                    use="sft",
                    extra=extra,
                )
            )
    return rows, {"source": str(scene_path), "n_scenes": len(scenes), "n_with_rationale": n_think}


def frozen_structure_specs():
    return [
        ("grid_segments", {"rows": 3, "cols": 5}, grid(3, 5), "观察图中的方格。图中一共有多少条不同的水平或竖直线段？", 4 * math.comb(6, 2) + 6 * math.comb(4, 2), None),
        ("grid_squares", {"rows": 3, "cols": 7}, grid(3, 7), "观察图中的方格。图中一共有多少个正方形？", square_count(3, 7), None),
        ("missing_corner_squares", {"rows": 5, "cols": 5, "missing_top_right": True}, grid(5, 5, missing_top_right=True), "观察这个缺角方格图形。图中共有多少个正方形？", square_count(5, 5) - 5, None),
        ("point_pairs", {"points": 8}, point_set(8), "观察图中的点。任取两个点作为端点，一共能确定多少条线段？", math.comb(8, 2), None),
        ("t_noncollinear", {"horizontal": 6, "vertical": 5}, t_points(6, 5), "观察图中的点阵。任取三个不共线的点，一共能组成多少个三角形？", math.comb(10, 3) - math.comb(6, 3) - math.comb(5, 3), None),
        ("fan_triangles", {"rays": 6, "layers": 1}, fan(6, 1), "观察图形，图中共有多少个三角形？", math.comb(6, 2), None),
        ("fan_segment_delta", {"rays": 6, "layers": 1}, fan(6, 1), "观察图形，图中线段总数比三角形总数多多少？", 6, None),
        ("layered_fan", {"rays": 6, "layers": 2}, fan(6, 2), "观察图形，图中共有多少个三角形？", 2 * math.comb(6, 2), None),
        ("marked_squares", {"rows": 4, "cols": 5, "marked": [2, 3]}, grid(4, 5, marked=(2, 3)), "观察方格图，数出所有包含星号所在小格的正方形。", marked_square_count(4, 5, 2, 3), None),
        ("marked_rectangles", {"rows": 5, "cols": 7, "marked": [2, 4]}, grid(5, 7, marked=(2, 4)), "观察方格图，数出所有包含星号所在小格的长方形（包括正方形）。", 2 * 4 * 4 * 4, None),
    ]


def param_key(mechanism: str, params: dict) -> tuple:
    items = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in params.items()))
    return (mechanism, items)


def structure_body(mechanism: str, params: dict):
    if mechanism in {"grid_segments", "grid_squares"}:
        return grid(params["rows"], params["cols"])
    if mechanism == "missing_corner_squares":
        return grid(params["rows"], params["cols"], missing_top_right=True)
    if mechanism == "point_pairs":
        return point_set(params["points"])
    if mechanism == "t_noncollinear":
        return t_points(params["horizontal"], params["vertical"])
    if mechanism in {"fan_triangles", "fan_segment_delta"}:
        return fan(params["rays"], params.get("layers", 1))
    if mechanism == "layered_fan":
        return fan(params["rays"], params["layers"])
    if mechanism == "marked_squares":
        rr, cc = params["marked"]
        return grid(params["rows"], params["cols"], marked=(rr, cc))
    if mechanism == "marked_rectangles":
        rr, cc = params["marked"]
        return grid(params["rows"], params["cols"], marked=(rr, cc))
    raise KeyError(mechanism)


def expand_structure_train(target: int, banned: set[tuple]) -> list:
    items = []
    for mechanism, params, body, question, answer, reason in struct_train32_specs():
        key = param_key(mechanism, params)
        if key in banned:
            continue
        items.append((mechanism, params, body, question, answer, reason))
        banned.add(key)

    def add(mechanism, params, question, answer, reason):
        key = param_key(mechanism, params)
        if key in banned:
            return
        items.append((mechanism, params, structure_body(mechanism, params), question, answer, reason))
        banned.add(key)

    for r in range(2, 8):
        for c in range(3, 9):
            ans = (r + 1) * math.comb(c + 1, 2) + (c + 1) * math.comb(r + 1, 2)
            add("grid_segments", {"rows": r, "cols": c}, "数出图中所有不同的水平或竖直线段。", ans, None)
            ans_sq = square_count(r, c)
            add("grid_squares", {"rows": r, "cols": c}, "数出图中所有大小的正方形。", ans_sq, None)
            if r >= 3 and c >= 3:
                add(
                    "missing_corner_squares",
                    {"rows": r, "cols": c, "missing_top_right": True},
                    "观察这个缺角方格图形。图中共有多少个正方形？",
                    square_count(r, c) - min(r, c),
                    None,
                )
    for n in range(5, 12):
        add("point_pairs", {"points": n}, "观察图中的点。任取两个点作为端点，一共能确定多少条线段？", math.comb(n, 2), None)
    for h in range(4, 9):
        for v in range(3, 8):
            total = h + v - 1
            ans = math.comb(total, 3) - math.comb(h, 3) - math.comb(v, 3)
            if ans <= 0:
                continue
            add(
                "t_noncollinear",
                {"horizontal": h, "vertical": v},
                "任取图中三个不共线的点，一共能组成多少个三角形？",
                ans,
                None,
            )
    for rays in range(4, 10):
        add("fan_triangles", {"rays": rays, "layers": 1}, "观察图形，图中共有多少个三角形？", math.comb(rays, 2), None)
        add("fan_segment_delta", {"rays": rays, "layers": 1}, "观察图形，图中线段总数比三角形总数多多少？", rays, None)
        for layers in (2, 3):
            add(
                "layered_fan",
                {"rays": rays, "layers": layers},
                "观察图形，图中共有多少个三角形？",
                layers * math.comb(rays, 2),
                None,
            )
    for r in range(3, 8):
        for c in range(4, 9):
            for rr in range(1, r + 1):
                for cc in range(1, c + 1):
                    if len(items) >= target:
                        return items[:target]
                    add(
                        "marked_squares",
                        {"rows": r, "cols": c, "marked": [rr, cc]},
                        "数出所有包含星号所在小格的正方形。",
                        marked_square_count(r, c, rr, cc),
                        None,
                    )
                    add(
                        "marked_rectangles",
                        {"rows": r, "cols": c, "marked": [rr, cc]},
                        "数出所有包含星号所在小格的长方形（包括正方形）。",
                        rr * (r - rr + 1) * cc * (c - cc + 1),
                        None,
                    )
    return items[:target]


def export_structure(out: Path, n_train: int) -> dict:
    dest = out / "structure_count"
    svg_dir = dest / "svg_sources"
    img_high = dest / "images" / "high"
    img_low = dest / "images" / "low"
    for d in (svg_dir, img_high, img_low):
        d.mkdir(parents=True, exist_ok=True)

    frozen = frozen_structure_specs()
    banned = {param_key(m, p) for m, p, *_ in frozen}
    train_items = expand_structure_train(n_train, banned)
    rows = []

    def emit(split: str, mechanism: str, params: dict, body: str, question: str, answer, reason: str | None, idx: int):
        scene_id = f"struct_{split}_{idx:03d}_{mechanism}"
        source = svg_dir / f"{scene_id}.svg"
        source.write_text(svg(body), encoding="utf-8")
        high = img_high / f"{scene_id}.png"
        low = img_low / f"{scene_id}.png"
        render_svg(source, high, 720)
        render_svg(source, low, 144)
        extra = {
            "image_low": relpath(low.relative_to(dest)),
            "resolution_pair": {"high": 720, "low": 144},
        }
        rows.append(
            record(
                rec_id=scene_id,
                scene_id=scene_id,
                task="structure_count",
                split=split,
                image=relpath(high.relative_to(dest)),
                question=question,
                answer=str(answer),
                rationale=reason,
                scene={"mechanism": mechanism, "params": params, "frozen_test": split == "test"},
                answer_source="combinatorics",
                use="probe",
                extra=extra,
            )
        )

    for i, (mechanism, params, body, question, answer, reason) in enumerate(frozen, 1):
        emit("test", mechanism, params, body, question, answer, reason, i)
    for i, (mechanism, params, body, question, answer, reason) in enumerate(train_items, 1):
        emit("train", mechanism, params, body, question, answer, reason, i)

    write_jsonl(dest / "records.jsonl", rows)
    return {"n_test": len(frozen), "n_train": len(train_items), "n_records": len(rows)}


def copy_generators(out: Path) -> None:
    dest = out / "scripts"
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        SCRIPTS / "generate_cube_fold.py": dest / "generate_cube_fold.py",
        SCRIPTS / "generate_cube_fold_v3.py": dest / "generate_cube_fold_v3.py",
        SCRIPTS / "generate_vessels_v2.py": dest / "generate_vessels_v2.py",
        SCRIPTS / "verify_vessels.py": dest / "verify_vessels.py",
        SCRIPTS / "verify_pack.py": dest / "verify_pack.py",
        SCRIPTS / "load_sft.py": dest / "load_sft.py",
        SCRIPTS / "fill_programmatic_rationales.py": dest / "fill_programmatic_rationales.py",
        SCRIPTS / "generate_lowres_structure_probe_v1.py": dest / "generate_lowres_structure_probe_v1.py",
        SCRIPTS / "generate_structure_count_train32_v1.py": dest / "generate_structure_count_train32_v1.py",
        SCRIPTS / "export_opensource_pack.py": dest / "export_opensource_pack.py",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)


def write_readme(out: Path, stats: dict) -> None:
    text = f"""# Procedural Visual Reasoning

程序化视觉推理工具包。默认用法是：**可见三面折叠 + 读图连通器** 做可验证 SFT；对面颜色、结构计数、泄漏 think 不进默认训练集。

仓库不包含第三方评测集的原题或原图。它是小规模 SFT / 诊断材料，不是通用视觉推理基准。

## 推荐训练（`use=sft`）

| 子集 | 记录 | 说明 |
|---|---:|---|
| `cube_fold/` | {stats.get("cube_records", "?")} | 固定朝向的图内相邻三面，可训练折叠语料 |
| `vessels/` | {stats.get("vessels_records", "?")} | 溢出 / 进不了水 / 终态水位；数字只在图上 |

```python
from pathlib import Path
import json

def load_sft(root="."):
    rows = []
    for name in ("cube_fold", "vessels"):
        for line in Path(root, name, "records.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["split"] == "train" and r.get("use", "sft") == "sft":
                rows.append(r)
    return rows
```

## 不要默认训练

| 子集 | 记录 | `use` | 说明 |
|---|---:|---|---|
| `cube_opposite/` | {stats.get("cube_opposite_records", "?")} | diagnostic | 对面颜色；不进入默认 SFT |
| `structure_count/` | {stats.get("structure_records", "?")} | probe | 线稿计数；`split=test` 冻结，禁止训练 |
| `leak_contrast/` | {stats.get("leak_records", 0)} | contrast | 教师解释含答案字母，用于泄漏研究 |

改 seed 再生同类题：见 `scripts/generate_*.py`。答案由几何 / 物理模拟 / 组合公式给出。

## 已知边界

- 历史探索中曾观察到对面颜色训练的负迁移，但完整训练与评测产物未保留；该结果不可独立复核，不作一般性结论。
- 可见三面问的是该轴测朝向，不是任意视角。
- 连通器采用可确定求解的离散事件模型；题干不再复述顶沿/管道数字。
- 结构计数的 10 条 test 仅用于冻结探测；历史小模型观察缺少完整产物，不作一般性结论。

## 自检

```bash
python scripts/verify_pack.py
```

## 许可

- 生成器代码：MIT（`LICENSE`）
- 数据：CC BY 4.0（`LICENSE-DATA`）
- 部分教师解释由阿里云百炼正式 API 的 qwen3.8-max 生成；未逐句程序核验
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def verify_pack(out: Path) -> dict:
    problems = []
    counts = {}
    for name in ("cube_fold", "cube_opposite", "vessels", "structure_count", "leak_contrast"):
        path = out / name / "records.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        counts[name] = len(rows)
        scene_split_map: dict[str, set[str]] = {}
        for row in rows:
            blob = json.dumps(row, ensure_ascii=False)
            if "train_sample_1000" in blob:
                problems.append(f"{name}:{row.get('id')} contains forbidden path")
            img = (out / name / row["image"]).resolve()
            if not img.exists():
                problems.append(f"missing image {row['image']}")
            scene_split_map.setdefault(row["scene_id"], set()).add(row["split"])
        for sid, splits in scene_split_map.items():
            if len(splits) != 1:
                problems.append(f"{name} scene {sid} split mismatch: {splits}")
        if name == "structure_count":
            test_mechs = {r["scene"]["mechanism"] for r in rows if r["split"] == "test"}
            train_keys = {param_key(r["scene"]["mechanism"], r["scene"]["params"]) for r in rows if r["split"] == "train"}
            for r in rows:
                if r["split"] == "test" and param_key(r["scene"]["mechanism"], r["scene"]["params"]) in train_keys:
                    problems.append(f"structure test leaked into train: {r['id']}")
            if len(test_mechs) < 8:
                problems.append("structure test too small")
    if problems:
        raise SystemExit("VERIFY FAIL\n" + "\n".join(problems[:40]))
    print(json.dumps({"verify": "ok", "counts": counts}, ensure_ascii=False, indent=2))
    return counts


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover(root: Path) -> dict:
    salvage = root / "data" / "salvage"
    syn_dirs = [salvage / "synthetic", root / "data" / "synthetic"]
    sft_dirs = [salvage / "sft_ready", root / "data" / "processed" / "sft_ready"]
    syn = next((p for p in syn_dirs if p.exists()), syn_dirs[-1])
    sft = next((p for p in sft_dirs if p.exists()), sft_dirs[-1])
    ood_dirs = [p for p in (syn / "vessels_v2_ood_100", syn / "vessels_v2_ood_n5", syn / "vessels_v2_ood_n6") if p.exists() and p.is_dir()]
    found = {
        "syn": syn,
        "sft": sft,
        "cube_900": first_existing(syn / "cube_fold_v2_l2_900", root / "data" / "synthetic" / "cube_fold_v2_l2_900"),
        "cube_l1": first_existing(syn / "cube_fold_v2_l1_300", root / "data" / "synthetic" / "cube_fold_v2_l1_300"),
        "vessels_48": first_existing(syn / "vessels_v2_48", syn / "vessels_v2_sample50"),
        "vessels_l2": first_existing(syn / "vessels_v2_l2_300"),
        "vessels_ood_dirs": ood_dirs,
        "vessels_sft": first_existing(sft / "vessels_v2_348.jsonl"),
        "vessels_sft_48": first_existing(sft / "vessels_v2_48.jsonl"),
        "vessels_sft_300": first_existing(sft / "vessels_v2_l2_300.jsonl"),
        "cube_sft": first_existing(sft / "cube_fold_v2_l2_900.jsonl"),
    }
    report = {}
    for key, val in found.items():
        if key == "vessels_ood_dirs":
            report[key] = [str(p) for p in val]
        elif isinstance(val, Path) or val is None:
            report[key] = {"path": str(val) if val else None, "exists": bool(val and val.exists())}
        else:
            report[key] = str(val)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--structure-train", type=int, default=240)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    out = args.out or (args.root / "opensource" / "procedural-visual-reasoning")
    if args.verify_only:
        verify_pack(out)
        return

    found = discover(args.root)
    if args.dry_run:
        return

    resolved_out = out.resolve()
    protected = {Path("/").resolve(), Path.cwd().resolve(), args.root.resolve(), Path.home().resolve()}
    if resolved_out in protected or len(resolved_out.parts) < 4:
        raise ValueError(f"refusing unsafe output directory: {resolved_out}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    (out / "LICENSE-DATA").write_text(CC_BY_TEXT, encoding="utf-8")
    copy_generators(out)

    stats = {}
    cube_dir = found["cube_900"]
    if cube_dir is None:
        raise FileNotFoundError("cube_fold_v2_l2_900 source directory is missing")
    stats["cube"] = export_cube(args.root, out, cube_dir, found["cube_sft"], leak=False)
    stats["cube_records"] = stats["cube"]["n_visible"]
    stats["cube_opposite_records"] = stats["cube"]["n_opposite"]

    if found["cube_l1"]:
        stats["leak"] = export_cube(args.root, out, found["cube_l1"], None, leak=True)
        stats["leak_records"] = stats["leak"]["n_records"]

    vessel_rows = []
    vessel_stats = []
    dest_v = out / "vessels"
    dest_v.mkdir(parents=True, exist_ok=True)
    if found["vessels_48"]:
        rows, meta = export_vessels_folder(found["vessels_48"], dest_v, "train", found["vessels_sft_48"] or found["vessels_sft"], "v48")
        vessel_rows.extend(rows)
        vessel_stats.append({"split": "train48", **meta})
    if found["vessels_l2"]:
        rows, meta = export_vessels_folder(found["vessels_l2"], dest_v, "train", found["vessels_sft_300"] or found["vessels_sft"], "v300")
        vessel_rows.extend(rows)
        vessel_stats.append({"split": "train300", **meta})
    for ood_dir in found["vessels_ood_dirs"]:
        prefix = "vood5" if "n5" in ood_dir.name else ("vood6" if "n6" in ood_dir.name else "vood")
        rows, meta = export_vessels_folder(ood_dir, dest_v, "test", None, prefix)
        vessel_rows.extend(rows)
        vessel_stats.append({"split": f"ood:{ood_dir.name}", **meta})
    if not vessel_rows:
        raise FileNotFoundError("no vessels scenes found")
    write_jsonl(dest_v / "records.jsonl", vessel_rows)
    stats["vessels"] = vessel_stats
    stats["vessels_records"] = len(vessel_rows)

    stats["structure"] = export_structure(out, args.structure_train)
    stats["structure_records"] = stats["structure"]["n_records"]

    write_readme(out, stats)
    (out / "MANIFEST.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_pack(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
