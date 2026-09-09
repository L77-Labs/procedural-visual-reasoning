#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定朝向可见三面选择题生成器。

本生成器把“展开图折叠”拆成固定、可核验的视觉任务：
  - 左侧：6 面展开图（纯色块，无格内文字、无标题）
  - 右侧：4 幅候选轴测图（A/B/C/D 标注），每幅显示折叠后 3 个可见面
  - 正确项 = 真实折叠后 +z/+x/+y 视角的 3 面颜色组合
  - 干扰项 = 单色错配的轴测图（替换 1 个可见面颜色），
    且通过"8 个真实角组合"穷举校验保证该组合不可能出现（唯一正确）
  - 使用 Material 风格的自定义六色色板
  - 题干明确限定为图中的固定轴测朝向；选项只出现在图内

几何推导复用 v2：fold_propagate 算出每面法线方向（6 面双射），
可见面 = 固定轴测视角 (+z/+x/+y)；8 个真实角组合 = 从 6 个法线中选
3 个两两相邻（共享顶点）的面。

用法:
    python generate_cube_fold_v3.py --count 30 --out-dir ./cube_fold_v3_sample
"""
import argparse
import itertools
import json
import math
import os
import random

from generate_cube_fold import fold_propagate, NET_LAYOUTS, neg

_MPL = None


def _mpl():
    global _MPL
    if _MPL is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        _MPL = {"plt": plt, "Rectangle": Rectangle, "Polygon": Polygon}
    return _MPL

# ------------------------------------------------------------------
# 自定义 Material 六色色板，与发布数据保持一致。
# ------------------------------------------------------------------
COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]
COLOR_RGB = {
    "红色": "#e53935",
    "蓝色": "#1e88e5",
    "绿色": "#43a047",
    "黄色": "#fdd835",
    "紫色": "#8e24aa",
    "橙色": "#fb8c00",
}

# 轴测视角可见的三个法线方向（与 v2 draw_iso_cube 一致：右 +x / 左 +y / 顶 +z）
VISIBLE_NORMALS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def corner_triplets(normals):
    """8 个真实角组合：从 6 个法线里选 3 个两两相邻（任一对不相对）的面。

    正方体任意视角恰好看到共享一个顶点的 3 个面 => 3 面中任一对都不能是
    对面。枚举 6 选 3 共 20 种组合，过滤含相对对的，剩 8 种。
    """
    all_dirs = set(normals.values())
    triplets = []
    for combo in itertools.combinations(all_dirs, 3):
        if all(neg(a) not in combo for a in combo):
            triplets.append(set(combo))
    assert len(triplets) == 8, f"角组合数应为 8，实际 {len(triplets)}"
    return triplets


def visible_colors(normals, cell_colors):
    """固定轴测视角 (+z/+x/+y) 可见的 3 个面的颜色（按 视角面 排序）。"""
    color_of = {}
    for cell, n in normals.items():
        color_of[n] = cell_colors[cell]
    return [color_of[n] for n in VISIBLE_NORMALS]


def color_of_triplet(triplet, normals, cell_colors):
    color_of = {}
    for cell, n in normals.items():
        color_of[n] = cell_colors[cell]
    return [color_of[n] for n in triplet]


FACE_NAMES = ("右", "左", "顶")


def visible_rationale(correct, options, normals=None, cell_colors=None):
    """Fold-derived explanation. Does not name option letters (avoids 选A shortcuts)."""
    correct = list(correct)
    real = None
    if normals is not None and cell_colors is not None:
        real = {
            frozenset(color_of_triplet(t, normals, cell_colors))
            for t in corner_triplets(normals)
        }
    parts = [
        "展开图折叠后，该固定朝向能看到的三个面从右、左、顶依次是"
        f"{correct[0]}、{correct[1]}、{correct[2]}。"
    ]
    for letter in "ABCD":
        trip = list(options[letter])
        if trip == correct:
            continue
        diffs = "、".join(
            f"{FACE_NAMES[i]}面被换成{trip[i]}" for i in range(3) if trip[i] != correct[i]
        )
        if real is not None and frozenset(trip) in real:
            why = "这三色能构成某个真角，但右、左、顶的顺序与该朝向不符。"
        else:
            why = "这三色不能出现在立方体的同一个顶点上。"
        parts.append(f"{trip[0]}、{trip[1]}、{trip[2]}这一组合{diffs}，{why}")
    parts.append(f"与折叠结果一致的三色顺序是{correct[0]}、{correct[1]}、{correct[2]}。")
    return "".join(parts)


# ------------------------------------------------------------------
# 渲染：无格内文字、无标题、纯色块
# ------------------------------------------------------------------

def draw_net_v3(ax, layout_cells, cell_colors):
    """展开图：纯色块 + 黑边框，无格内文字、无标题。"""
    Rectangle = _mpl()["Rectangle"]
    max_r = max(r for r, c in layout_cells)
    for (r, c) in layout_cells:
        color_name = cell_colors[(r, c)]
        rect = Rectangle((c, max_r - r), 1, 1, facecolor=COLOR_RGB[color_name],
                          edgecolor="black", linewidth=2)
        ax.add_patch(rect)
    cols = [c for r, c in layout_cells]
    rows = [r for r, c in layout_cells]
    ax.set_xlim(min(cols) - 0.3, max(cols) + 1.3)
    ax.set_ylim(-0.3, max_r + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_iso_cube_v3(ax, color_triplet):
    """画轴测立方体（3 个可见面），颜色由调用方指定（用于正确项和干扰项）。"""
    Polygon = _mpl()["Polygon"]
    L = 1.0
    r_vec = (math.cos(math.radians(-30)) * L, math.sin(math.radians(-30)) * L)
    l_vec = (math.cos(math.radians(210)) * L, math.sin(math.radians(210)) * L)
    u_vec = (0, L)

    O = (0, 0)

    def add(p, v):
        return (p[0] + v[0], p[1] + v[1])

    p_r = add(O, r_vec)
    p_l = add(O, l_vec)
    p_u = add(O, u_vec)
    p_ru = add(p_r, u_vec)
    p_lu = add(p_l, u_vec)
    p_rlu = add(add(p_r, l_vec), u_vec)

    # 右面 / 左面 / 顶面（顺序对应 VISIBLE_NORMALS: +x, +y, +z）
    right_c, left_c, top_c = color_triplet
    ax.add_patch(Polygon([O, p_r, p_ru, p_u], closed=True, facecolor=COLOR_RGB[right_c],
                          edgecolor="black", linewidth=2))
    ax.add_patch(Polygon([O, p_l, p_lu, p_u], closed=True, facecolor=COLOR_RGB[left_c],
                          edgecolor="black", linewidth=2))
    ax.add_patch(Polygon([p_u, p_lu, p_rlu, p_ru], closed=True, facecolor=COLOR_RGB[top_c],
                          edgecolor="black", linewidth=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.05, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")


def render_sample_v3(layout_cells, cell_colors, visible_triplet, distractor_triplets, out_path):
    """左侧展开图 + 右侧 4 幅候选轴测图（A/B/C/D 标在下方）。"""
    plt = _mpl()["plt"]
    n_opt = 1 + len(distractor_triplets)
    fig, axes = plt.subplots(1, 1 + n_opt, figsize=(4.4 + 2.3 * n_opt, 4.4))
    draw_net_v3(axes[0], layout_cells, cell_colors)
    all_opts = [visible_triplet] + distractor_triplets
    for i, triplet in enumerate(all_opts):
        ax = axes[1 + i]
        draw_iso_cube_v3(ax, triplet)
        ax.text(0, -0.78, "ABCD"[i], ha="center", va="center", fontsize=16, fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def generate_distractors(correct_triplet, normals, cell_colors, rng, k=3):
    """生成 k 个必错干扰项：错配颜色，且三色集合不是 8 个真实角之一。

    必须用颜色集合过滤。旧实现把颜色 frozenset 去和法线 frozenset 比较，
    过滤永不生效，干扰项经常仍是另一个真实角。
    """
    real_color_corners = {
        frozenset(color_of_triplet(t, normals, cell_colors))
        for t in corner_triplets(normals)
    }
    others = [c for c in COLOR_NAMES if c not in correct_triplet]
    distractors, seen = [], set()

    def accept(combo):
        key = tuple(combo)
        if key in seen or frozenset(combo) in real_color_corners:
            return False
        seen.add(key)
        distractors.append(list(combo))
        return True

    attempts = 0
    while len(distractors) < k and attempts < 2000:
        attempts += 1
        combo = list(correct_triplet)
        combo[rng.randrange(3)] = rng.choice(others)
        accept(combo)
    while len(distractors) < k and attempts < 4000:
        attempts += 1
        combo = list(correct_triplet)
        for pos in rng.sample(range(3), 2):
            combo[pos] = rng.choice(others)
        accept(combo)
    assert len(distractors) == k, f"重试 {attempts} 次仍凑不齐 {k} 个合法干扰项"
    return distractors


def generate_one(idx, out_dir, rng, seen=None):
    for _attempt in range(100):
        layout_name = rng.choice(list(NET_LAYOUTS.keys()))
        layout_cells = NET_LAYOUTS[layout_name]
        normals = fold_propagate(layout_cells)

        colors = COLOR_NAMES[:]
        rng.shuffle(colors)
        cell_colors = {cell: colors[i] for i, cell in enumerate(layout_cells)}

        correct_triplet = visible_colors(normals, cell_colors)
        distractors = generate_distractors(correct_triplet, normals, cell_colors, rng, k=3)

        # 4 选项：正确 + 3 干扰，随机打乱字母
        all_opts = [correct_triplet] + distractors
        rng.shuffle(all_opts)
        letters = ["A", "B", "C", "D"]
        letter_of = dict(zip(letters, all_opts))
        correct_letter = [l for l, t in letter_of.items() if t == correct_triplet][0]

        if seen is None:
            break
        key = (layout_name, tuple(sorted(tuple(t) for t in all_opts)))
        if key not in seen:
            seen.add(key)
            break
    else:
        raise RuntimeError(f"布局组合空间耗尽（重试 {_attempt + 1} 次仍撞车）")

    img_name = f"cube_fold_v3_{idx:04d}.png"
    img_path = os.path.join(out_dir, img_name)
    render_sample_v3(layout_cells, cell_colors, correct_triplet, distractors, img_path)

    question = (
        "左侧图像显示了一个正方体的展开图，六个面分别涂有不同的颜色。"
        "将展开图向上折叠成一个正方体。从轴测（3D）视角观察该正方体，"
        "图中固定轴测朝向对应哪一组相邻颜色？请从选项A、B、C或D中选择一个字母回答。"
    )

    # 保存求解所需的结构化场景，使标签能够从输出记录中重新计算。
    cell_coords = {color: [r + 1, c + 1] for (r, c), color in cell_colors.items()}
    scene_params = {
        "layout_name": layout_name,
        "cell_colors": cell_coords,
        "correct_triplet": list(correct_triplet),
        "options": {l: list(t) for l, t in letter_of.items()},
    }

    return {
        "id": f"cube_fold_v3_{idx:04d}_visible",
        "scene_id": f"cube_fold_v3_{idx:04d}",
        "task": "cube_visible_triplet",
        "split": "train",
        "use": "sft",
        "question": question,
        "image": img_name,
        "answer": correct_letter,
        "choices": ["A", "B", "C", "D"],
        "rationale": visible_rationale(correct_triplet, letter_of, normals, cell_colors),
        "rationale_source": "geometry",
        "scene": scene_params,
        "answer_source": "geometry",
        "license": "CC-BY-4.0",
        "source": "synthetic_cube_fold_v3",
    }, {
        "layout_name": layout_name,
        "cell_colors": cell_colors,
        "correct_triplet": list(correct_triplet),
        "options": {l: list(t) for l, t in letter_of.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out-dir", type=str, default="./cube_fold_v3_sample")
    ap.add_argument("--jsonl", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = args.jsonl or os.path.join(args.out_dir, "records.jsonl")

    rng = random.Random(args.seed)

    for name, cells in NET_LAYOUTS.items():
        n = fold_propagate(cells)
        assert len(corner_triplets(n)) == 8
        print(f"[自检通过] 布局「{name}」: 折叠几何合法，角组合 8 个")

    records, debug_infos = [], []
    seen = set()
    for i in range(args.count):
        rec, dbg = generate_one(i, args.out_dir, rng, seen)
        records.append(rec)
        debug_infos.append(dbg)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"生成完成: {len(records)} 条样本 -> {jsonl_path}")
    for i in range(min(3, len(records))):
        print("\n--- 样本", i, "---")
        print("正确3面:", debug_infos[i]["correct_triplet"])
        print("选项:", debug_infos[i]["options"])
        print("答案:", records[i]["answer"])
        print("题目:", records[i]["question"])


if __name__ == "__main__":
    main()
