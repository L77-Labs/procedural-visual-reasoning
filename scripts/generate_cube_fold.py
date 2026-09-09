#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间推理维度 - 正方体展开图折叠 procedural generator。

核心设计（"合成数据的参考锚点分类原则"）：
    答案不是人工/模型编写，而是从同一套几何生成代码里严格推导出来的。
    具体做法：把"展开图折叠成正方体"建模成一个刚体旋转传播问题——
    从网格里选一个根方块，赋予其局部坐标系 (u=右, v=下, n=法线/朝外方向)，
    然后沿着展开图里各方块间的公共边，用折叠 90 度的规则把坐标系传播到
    所有相邻方块。传播规则（一律用 +-1/0 的整数向量，不用浮点数，避免精度问题）：

        向右折 (dc=+1):  n' = u,    u' = -n,   v' = v
        向左折 (dc=-1):  n' = -u,   u' = n,    v' = v
        向下折 (dr=+1):  n' = v,    v' = -n,   u' = u
        向上折 (dr=-1):  n' = -v,   v' = n,    u' = u

    这四条规则可以用一个具体例子验证：把顶面(法线=+z, u=+x, v=+y)向右折，
    结果的法线变成旧的 u=+x —— 也就是立方体的右侧面朝向 +x，这与常识吻合。

    折完之后，每个方块都会得到一个"法线方向"，六个方块的法线必须恰好是
    {+x,-x,+y,-y,+z,-z} 六个方向各出现一次（互不重复）——代码里做了硬性
    assert 校验这一点，如果某个展开图布局不合法（不能无重叠地折成正方体），
    生成时就会直接报错，而不会产出静默错误的数据。
    "对面"关系此时就是纯几何事实：法线互为相反数(n2 == -n1)的两个面即为对面，
    不需要任何额外校验或人工标注。

用法:
    python generate_cube_fold.py --count 20 --out-dir ./cube_fold_data
"""
import argparse
import json
import math
import os
import random

# matplotlib 仅渲染需要；fold_propagate 可无绘图库自检。

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Ubuntu / Debian
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",           # Linux fallback
    "/System/Library/Fonts/STHeiti Medium.ttc",                 # macOS
    "/System/Library/Fonts/PingFang.ttc",                       # macOS 备用
]

_MPL = None


def _mpl():
    global _MPL
    if _MPL is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from matplotlib.patches import Polygon, Rectangle

        font = None
        for path in _CJK_FONT_CANDIDATES:
            if os.path.exists(path):
                font = fm.FontProperties(fname=path)
                break
        if font is None:
            print("[警告] 未找到可用的中文字体文件，图片中的中文可能显示为方框。")
            font = fm.FontProperties()
        _MPL = {"plt": plt, "Rectangle": Rectangle, "Polygon": Polygon, "font": font}
    return _MPL

# ------------------------------------------------------------------
# 1. 展开图布局定义 (至少 4 种不同的六联骨牌 hexomino 布局)
#    每个布局是一个 (row, col) 坐标列表，共 6 个方块。
#    这些坐标本身只是声明；是否能无重叠折成立方体，由后面的几何传播算法
#    自动校验（校验 6 个法线互不相同），不是靠人工保证。
# ------------------------------------------------------------------
NET_LAYOUTS = {
    "十字形 (cross)": [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)],
    "T字偏移十字 (off-cross)": [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)],
    "楼梯形 (staircase)": [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
    "两端凸出条形 (offset-strip)": [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)],
    "Z字形阶梯 (2-2-2 staircase)": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    # 2026-08-10 复盘新增（阶梯②放量前）：仅 5 种布局时 question 组合空间仅 289 种，
    # 300 条撞出 14 组同题面样本（强化背题面捷径）。以下 5 个均通过 fold_propagate 自检。
    "T形 (t-shape)": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1)],
    "L形 (l-shape)": [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)],
    "长Z形 (long-z)": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    "侧翼条形 (side-wing)": [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3)],
    "偏移L (offset-l)": [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
}

COLOR_NAMES = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色"]
COLOR_RGB = {
    "红色": "#e53935",
    "蓝色": "#1e88e5",
    "绿色": "#43a047",
    "黄色": "#fdd835",
    "紫色": "#8e24aa",
    "橙色": "#fb8c00",
}
# 深色底用白字，浅色底(黄)用黑字，保证文字可读
DARK_TEXT_ON = {"黄色"}


def neg(v):
    return (-v[0], -v[1], -v[2])


def fold_propagate(layout_cells):
    """从展开图坐标 -> 每个方块折叠后的法线方向 (3D单位向量, 用整数元组表示)。

    返回: dict {(r,c): n_vector}
    并对结果做严格自检：6 个法线必须恰好覆盖 {+x,-x,+y,-y,+z,-z}。
    """
    cell_set = set(layout_cells)
    root = layout_cells[0]
    frames = {root: {"u": (1, 0, 0), "v": (0, 1, 0), "n": (0, 0, 1)}}

    # BFS 沿着网格 4 邻接展开（合法展开图的邻接关系一定是一棵树：6 节点 5 条边）
    from collections import deque

    visited = {root}
    queue = deque([root])
    edges = 0
    while queue:
        r, c = queue.popleft()
        fr = frames[(r, c)]
        for dr, dc, name in [(0, 1, "right"), (0, -1, "left"), (1, 0, "down"), (-1, 0, "up")]:
            nxt = (r + dr, c + dc)
            if nxt not in cell_set or nxt in visited:
                continue
            u1, v1, n1 = fr["u"], fr["v"], fr["n"]
            if name == "right":
                n2, u2, v2 = u1, neg(n1), v1
            elif name == "left":
                n2, u2, v2 = neg(u1), n1, v1
            elif name == "down":
                n2, u2, v2 = v1, u1, neg(n1)
            else:  # up
                n2, u2, v2 = neg(v1), u1, n1
            frames[nxt] = {"u": u2, "v": v2, "n": n2}
            visited.add(nxt)
            queue.append(nxt)
            edges += 1

    assert visited == cell_set, f"展开图不连通: {cell_set - visited}"
    assert edges == len(cell_set) - 1, f"展开图邻接关系不是树 (edges={edges}, 应为{len(cell_set)-1})，说明布局非法或有重叠"

    normals = {cell: frames[cell]["n"] for cell in layout_cells}
    all_dirs = set(normals.values())
    expected = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    assert all_dirs == expected, f"法线方向不构成六个方向的双射，布局非法: {all_dirs}"

    return normals


def opposite_pairs(normals):
    """由法线方向直接算出对面关系: n 与 -n 所在的两个方块互为对面。"""
    pairs = {}
    for cell, n in normals.items():
        opp_cell = [c2 for c2, n2 in normals.items() if n2 == neg(n)]
        assert len(opp_cell) == 1
        pairs[cell] = opp_cell[0]
    return pairs


# ------------------------------------------------------------------
# 2. 渲染: 左侧展开图 (2D) + 右侧折叠后的等角(isometric)立方体渲染
# ------------------------------------------------------------------

def draw_net(ax, layout_cells, cell_colors):
    mpl = _mpl()
    Rectangle, CJK_FONT = mpl["Rectangle"], mpl["font"]
    max_r = max(r for r, c in layout_cells)
    for (r, c) in layout_cells:
        color_name = cell_colors[(r, c)]
        rect = Rectangle((c, max_r - r), 1, 1, facecolor=COLOR_RGB[color_name],
                          edgecolor="black", linewidth=2)
        ax.add_patch(rect)
        txt_color = "black" if color_name in DARK_TEXT_ON else "white"
        ax.text(c + 0.5, max_r - r + 0.5, color_name, ha="center", va="center",
                 fontsize=13, color=txt_color, fontweight="bold", fontproperties=CJK_FONT)
    cols = [c for r, c in layout_cells]
    rows = [r for r, c in layout_cells]
    ax.set_xlim(min(cols) - 0.3, max(cols) + 1.3)
    ax.set_ylim(-0.3, max_r + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("展开图 (未折叠)", fontsize=12, fontproperties=CJK_FONT)


def draw_iso_cube(ax, normals, cell_colors):
    """画一个简单等角视图立方体，展示 3 个可见面: 法线分别为 +z(顶) / +x(右) / +y(左)。"""
    mpl = _mpl()
    Polygon, CJK_FONT = mpl["Polygon"], mpl["font"]
    color_of = {}
    for cell, n in normals.items():
        color_of[n] = cell_colors[cell]

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

    # 右侧可见面 <- 法线 +x
    right_color = COLOR_RGB[color_of[(1, 0, 0)]]
    ax.add_patch(Polygon([O, p_r, p_ru, p_u], closed=True, facecolor=right_color,
                          edgecolor="black", linewidth=2))
    # 左侧可见面 <- 法线 +y
    left_color = COLOR_RGB[color_of[(0, 1, 0)]]
    ax.add_patch(Polygon([O, p_l, p_lu, p_u], closed=True, facecolor=left_color,
                          edgecolor="black", linewidth=2))
    # 顶部可见面 <- 法线 +z
    top_color = COLOR_RGB[color_of[(0, 0, 1)]]
    ax.add_patch(Polygon([p_u, p_lu, p_rlu, p_ru], closed=True, facecolor=top_color,
                          edgecolor="black", linewidth=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.75, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("折叠后的立方体 (仅能看到3个面)", fontsize=12, fontproperties=CJK_FONT)


def render_sample(layout_name, layout_cells, cell_colors, normals, out_path):
    mpl = _mpl()
    plt, CJK_FONT = mpl["plt"], mpl["font"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8))
    draw_net(axes[0], layout_cells, cell_colors)
    draw_iso_cube(axes[1], normals, cell_colors)
    fig.suptitle(f"展开图类型: {layout_name}", fontsize=10, color="gray", fontproperties=CJK_FONT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ------------------------------------------------------------------
# 3. 单条样本生成
# ------------------------------------------------------------------

def generate_one(idx, out_dir, rng, seen=None):
    # 2026-08-10 复盘新增 dedup：question 组合空间有限（布局×目标×选项映射），
    # 随机撞出"同题面"样本会强化模型背题面不看图。撞到已用组合就重新 roll。
    for _attempt in range(100):
        layout_name = rng.choice(list(NET_LAYOUTS.keys()))
        layout_cells = NET_LAYOUTS[layout_name]

        normals = fold_propagate(layout_cells)
        pairs = opposite_pairs(normals)  # cell -> opposite cell

        colors = COLOR_NAMES[:]
        rng.shuffle(colors)
        cell_colors = {cell: colors[i] for i, cell in enumerate(layout_cells)}

        # 随机选一个目标面 (询问它的对面颜色)
        target_cell = rng.choice(layout_cells)
        target_color = cell_colors[target_cell]
        opp_cell = pairs[target_cell]
        correct_color = cell_colors[opp_cell]

        # 构造 4 选项 MCQ: 正确答案 + 3 个干扰项 (从其余 4 种颜色里随机选 3 个)
        other_colors = [c for c in COLOR_NAMES if c not in (target_color, correct_color)]
        distractors = rng.sample(other_colors, 3)
        options = distractors + [correct_color]
        rng.shuffle(options)
        letters = ["A", "B", "C", "D"]
        letter_of = dict(zip(letters, options))
        correct_letter = [l for l, c in letter_of.items() if c == correct_color][0]

        if seen is None:
            break  # 不启用 dedup
        key = (layout_name, target_color, tuple(sorted(letter_of.items())))
        if key not in seen:
            seen.add(key)
            break
    else:
        raise RuntimeError(f"布局组合空间耗尽（重试 {_attempt + 1} 次仍撞车）")

    options_text = "  ".join(f"{l}. {c}" for l, c in letter_of.items())
    # 2026-08-10 复盘修正：题面加入布局名（图上方 suptitle 已标注，属画面可见事实）。
    # 此前 question 不含布局信息，dedup 只能按(布局,目标,选项)去重，导致"同题面不同图"
    # 的样本（55 组/115 条）强化模型背题面不看图。现在题面携带布局 → 与 dedup key 一一对应。
    layout_cn = layout_name.split(" ")[0]
    question = (
        f"下图左侧是一个{layout_cn}的正方体展开图(折叠前)，右侧是折叠后立方体的示意图(仅显示3个可见面，"
        f"帮助你确认朝向)。请你在脑海中把左侧的展开图折叠成一个正方体，"
        f"判断：与「{target_color}」的面相对(即折叠后处于正方体两个相对位置)的是哪一个颜色的面？\n"
        f"{options_text}\n请只回答选项字母。"
    )

    img_name = f"cube_fold_{idx:04d}.png"
    img_path = os.path.join(out_dir, img_name)
    render_sample(layout_name, layout_cells, cell_colors, normals, img_path)

    # 保存图上可见事实，使答案能够从输出记录中重新计算。
    cell_coords = {color: [r + 1, c + 1] for (r, c), color in cell_colors.items()}
    scene_params = {
        "layout_name": layout_name,
        "cell_colors": cell_coords,
        "target_color": target_color,
        "options": letter_of,
    }

    return {
        "id": f"cube_fold_{idx:04d}_opposite",
        "scene_id": f"cube_fold_{idx:04d}",
        "task": "cube_opposite_face",
        "split": "train",
        "use": "diagnostic",
        "question": question,
        "image": img_name,
        "answer": correct_letter,
        "choices": [f"{letter}. {color}" for letter, color in letter_of.items()],
        "rationale": (
            f"沿展开图相邻边传播各面的折叠朝向后，{target_color}与{correct_color}的"
            "法线方向互为相反方向，因此两者是对面。"
        ),
        "rationale_source": "geometry",
        "scene": scene_params,
        "answer_source": "geometry",
        "license": "CC-BY-4.0",
        "source": "synthetic_cube_fold",
    }, {
        "layout_name": layout_name,
        "cell_colors": cell_colors,
        "target_color": target_color,
        "correct_color": correct_color,
        "options": letter_of,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="./cube_fold_data")
    ap.add_argument("--jsonl", type=str, default=None, help="JSONL 输出路径，默认 out-dir/records.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = args.jsonl or os.path.join(args.out_dir, "records.jsonl")

    rng = random.Random(args.seed)

    # 先对每个布局做一次几何自检 (assert 若失败会直接抛异常，中止生成)
    for name, cells in NET_LAYOUTS.items():
        n = fold_propagate(cells)
        opposite_pairs(n)
        print(f"[自检通过] 布局「{name}」: 6 个法线方向互不相同，折叠几何合法。")

    records = []
    debug_infos = []
    seen = set()
    for i in range(args.count):
        rec, dbg = generate_one(i, args.out_dir, rng, seen)
        records.append(rec)
        debug_infos.append(dbg)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"生成完成: {len(records)} 条样本 -> {jsonl_path}")
    print(f"图片输出目录: {args.out_dir}")

    # 打印前几条的调试信息，便于人工抽查
    for i in range(min(3, len(records))):
        print("\n--- 样本", i, "---")
        print("场景参数:", records[i]["scene"])
        print("目标色:", debug_infos[i]["target_color"], "-> 正确对面色:", debug_infos[i]["correct_color"])
        print("选项:", debug_infos[i]["options"])
        print("题目:", records[i]["question"])
        print("答案:", records[i]["answer"])


if __name__ == "__main__":
    main()
