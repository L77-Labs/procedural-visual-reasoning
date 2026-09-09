#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连通器 (connected vessels) 程序化视觉推理数据生成器。

每条记录保存结构化场景、模拟器答案和程序化事件解释，可直接按本仓库
schema 使用；物理模型与已发布数据采用相同的简化假设。

scene_params 字段（全部是图上标注/题面可见，不含答案）：
  n / labels   容器数与编号（图底部标甲乙丙丁）
  heights      各容器顶沿高度（图右侧标"顶沿高度=N"）
  pipes        相邻容器间管道高度（图上蓝色横条上方标"管道高度=N"）
  inflow       注水容器下标（该容器正上方有"持续注水"箭头）
  q_type       题型标识
  widths       容器宽度（图上可见，不影响答案）

答案反推：simulate(scene) + q_type 确定唯一标准答案（确定性物理模拟，见 verify_vessels.py）。

用法:
    python generate_vessels_v2.py --count 50 --out-dir ./vessels_50 --seed 2026
"""
import argparse
import json
import os
import random
import re

# matplotlib 惰性导入：纯逻辑路径（simulate/scene_params）不依赖渲染库，
# 逻辑校验工具可 import 本模块而不必安装 matplotlib。
# 渲染路径（render_scene）内局部导入。

LABELS = ["甲", "乙", "丙", "丁", "戊", "己"]  # 容器编号 (中文习惯，6 个供 OOD 5-6 容器 held-out 使用)

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def _resolve_cjk_font(fm):
    for path in _CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
    print("[警告] 未找到可用的中文字体文件，图片中的中文可能显示为方框。")
    return fm.FontProperties()


# ------------------------------------------------------------------
# 1. 场景随机生成: N 个容器 + N-1 根连接管道 + 1 个注水点
# ------------------------------------------------------------------

def generate_scene(rng, n_vessels=None):
    n = n_vessels or rng.choice([3, 3, 4, 4, 5])  # 3~4个居多，5个偶尔出现增加难度
    heights = rng.sample(range(6, 20), n)  # 各容器顶沿高度，取互不相同的整数，避免"打平手"
    widths = [rng.choice([1, 1.5, 2, 2.5, 3]) for _ in range(n)]  # 截面宽度仅用于画图

    # 相邻管道高度: pipe[i] 连接 vessel i 和 vessel i+1，要求 0 < pipe[i] <= min(h_i, h_{i+1}) - 1
    # (严格小于两侧顶沿，否则"管道等于顶沿"会有边界歧义)
    pipes = []
    for i in range(n - 1):
        upper = min(heights[i], heights[i + 1]) - 1
        if upper < 1:
            upper = 1
        p = rng.randint(1, upper)
        pipes.append(p)

    inflow = rng.randrange(n)

    return {
        "n": n,
        "heights": heights,
        "widths": widths,
        "pipes": pipes,  # len = n-1, pipes[i] 连接 i, i+1
        "inflow": inflow,
    }


# ------------------------------------------------------------------
# 2. 物理模拟（事件驱动，仅比较高度数值）
# ------------------------------------------------------------------

def simulate(scene):
    """事件驱动模拟，显式维护当前公共水位 level (单调不减)。

    每一轮先做"吸收阶段"：只要左右边界外的相邻管道高度 <= 当前 level，
    就立刻把那一侧容器并入连通区域 (不消耗额外的水位上升)。吸收阶段跑到
    不动点之后，才计算下一次真正需要"水位上升"才能触发的最小阈值。
    这保证了 events 列表里的阈值高度是严格非递减的，可以直接读成一段
    真实、按时间顺序发生的物理过程。
    """
    n = scene["n"]
    heights = scene["heights"]
    pipes = scene["pipes"]
    inflow = scene["inflow"]

    L = R = inflow
    level = 0
    events = []  # 记录 (阈值高度, 事件描述)

    while True:
        progressed = True
        while progressed:
            progressed = False
            if L > 0 and pipes[L - 1] <= level:
                events.append((level, f"水位达到 {level}（不低于容器{L-1}与{L}之间的管道高度 {pipes[L-1]}），容器{L-1}被并入连通区域"))
                L -= 1
                progressed = True
            if R < n - 1 and pipes[R] <= level:
                events.append((level, f"水位达到 {level}（不低于容器{R}与{R+1}之间的管道高度 {pipes[R]}），容器{R+1}被并入连通区域"))
                R += 1
                progressed = True

        cap = min(heights[i] for i in range(L, R + 1))
        if level >= cap:
            overflow_vessels = [i for i in range(L, R + 1) if heights[i] == cap]
            events.append((level, f"水位已达到容器范围[{L},{R}]中最矮的顶沿 {cap}，无法继续上升"))
            return {
                "final_level": cap,
                "final_range": (L, R),
                "first_full": overflow_vessels[0],
                "events": events,
                "never_filled": [i for i in range(n) if i < L or i > R],
            }

        left_pipe = pipes[L - 1] if L > 0 else None
        right_pipe = pipes[R] if R < n - 1 else None
        candidates = [p for p in (left_pipe, right_pipe) if p is not None]

        if not candidates or min(candidates) > cap:
            level = cap
            overflow_vessels = [i for i in range(L, R + 1) if heights[i] == cap]
            events.append((cap, f"水位升到 {cap}，容器范围[{L},{R}]中最矮的顶沿被打满"))
            return {
                "final_level": cap,
                "final_range": (L, R),
                "first_full": overflow_vessels[0],
                "events": events,
                "never_filled": [i for i in range(n) if i < L or i > R],
            }

        level = min(candidates)


# ------------------------------------------------------------------
# 3. 渲染: 容器 + 管道 + 注水箭头 (只画结构，不画水，答案靠推理)
# ------------------------------------------------------------------

def render_scene(scene, out_path):
    # 渲染路径才需要 matplotlib（惰性导入，见文件头注释）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import Rectangle
    cjk_font = _resolve_cjk_font(fm)

    n = scene["n"]
    heights = scene["heights"]
    widths = scene["widths"]
    pipes = scene["pipes"]
    inflow = scene["inflow"]

    gap = 3.4 if n >= 5 else 2.5
    xs = [0.0]
    for i in range(1, n):
        xs.append(xs[-1] + widths[i - 1] / 2 + gap + widths[i] / 2)

    fig, ax = plt.subplots(figsize=(max(8.5, 3.15 * n + 1.4), 7.2))
    rim_fs = 8 if n >= 5 else 9
    pipe_fs = 8 if n >= 5 else 9

    max_h = max(heights)
    for i in range(n):
        x_left = xs[i] - widths[i] / 2
        h = heights[i]
        ax.add_patch(Rectangle((x_left, 0), 0.12, h, facecolor="black"))
        ax.add_patch(Rectangle((x_left + widths[i] - 0.12, 0), 0.12, h, facecolor="black"))
        ax.add_patch(Rectangle((x_left, 0), widths[i], 0.12, facecolor="black"))
        ax.text(
            x_left + widths[i] + 0.18,
            h + 0.18,
            f"顶沿高度={h}",
            ha="left",
            va="bottom",
            fontsize=rim_fs,
            fontproperties=cjk_font,
        )
        ax.text(xs[i], -1.15, LABELS[i], ha="center", fontsize=15, fontweight="bold", fontproperties=cjk_font)

    for i in range(n - 1):
        p = pipes[i]
        x1 = xs[i] + widths[i] / 2 - 0.12
        x2 = xs[i + 1] - widths[i + 1] / 2 + 0.12
        ax.add_patch(Rectangle((x1, p - 0.1), x2 - x1, 0.2, facecolor="#1e88e5"))
        near_rim = min(abs(p - heights[i]), abs(p - heights[i + 1])) <= 2.6
        ly = (p - 0.58) if (near_rim and p >= 1.3) else (p + 0.48)
        ax.text(
            (x1 + x2) / 2,
            ly,
            f"管道高度={p}",
            ha="center",
            va="center",
            fontsize=pipe_fs,
            color="#1e88e5",
            fontproperties=cjk_font,
        )

    x_in = xs[inflow]
    h_in = heights[inflow]
    ax.annotate(
        "",
        xy=(x_in, h_in + 0.15),
        xytext=(x_in, h_in + 2.25),
        arrowprops=dict(facecolor="#1565c0", edgecolor="#1565c0", width=3, headwidth=12),
    )
    ax.text(
        x_in - max(0.85, widths[inflow] * 0.55),
        h_in + 2.05,
        "持续注水",
        ha="right",
        va="center",
        fontsize=11,
        color="#1565c0",
        fontproperties=cjk_font,
    )

    ax.set_xlim(xs[0] - widths[0] / 2 - 2.0, xs[-1] + widths[-1] / 2 + 3.4)
    ax.set_ylim(-2.1, max_h + 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "连通器结构示意图 (容器顶部开口，粗黑线为容器壁，蓝色横条为连接管道)",
        fontsize=11,
        fontproperties=cjk_font,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


# ------------------------------------------------------------------
# 4. 题目文本生成：数字只在图上，题干不得复述顶沿/管道/注水容器编号
# ------------------------------------------------------------------

OVERFLOW_TAIL = (
    "请问：哪一个容器会最先被装满水（即最先出现水从顶沿溢出）？请直接回答容器编号（如“甲”）。"
)
NEVER_TAIL = (
    "假设注水会持续进行很长时间（水从最先装满的容器溢出后也不会停止注水）。"
    "请问：最终有哪些容器完全接不到水（水位始终为0）？如果所有容器最终都能进水，请回答“没有”。"
)
LEVEL_TAIL = (
    "假设注水持续足够长的时间直到水位不再变化。请问：此时连通水域的最终水面高度是多少？"
    "（只需回答一个数字）"
)

TASK_OVERFLOW = "vessels_overflow"
TASK_NEVER = "vessels_never_fills"
TASK_LEVEL = "vessels_final_level"


def vision_stem(scene):
    labels = scene.get("labels") or LABELS[: scene["n"]]
    return (
        f"图中从左到右依次是开口容器 {'、'.join(labels)}，相邻容器由水平管道连通。"
        "各容器的顶沿高度、管道高度都标注在图中；容器底部在同一水平地面上。"
        "现在持续向图中标有「持续注水」的容器注水，注水速率恒定。"
    )


def vision_question(scene, task):
    stem = vision_stem(scene)
    if task in (TASK_OVERFLOW, "first_full"):
        return stem + OVERFLOW_TAIL
    if task in (TASK_NEVER, "never_filled_or_final"):
        return stem + NEVER_TAIL
    if task in (TASK_LEVEL, "final_level"):
        return stem + LEVEL_TAIL
    raise KeyError(task)


def _relabel_event(desc: str, labels: list) -> str:
    text = re.sub(
        r"容器范围\[(\d+),(\d+)\]",
        lambda m: f"连通范围{labels[int(m.group(1))]}到{labels[int(m.group(2))]}",
        desc,
    )
    text = re.sub(r"容器(\d+)", lambda m: labels[int(m.group(1))], text)
    text = re.sub(r"与(\d+)之间", lambda m: f"与{labels[int(m.group(1))]}之间", text)
    return text


def vessels_rationale(scene, task):
    """Event-trace explanation from simulate(). Numbers come from the figure, not the stem."""
    sim = simulate(scene)
    labels = scene.get("labels") or LABELS[: scene["n"]]
    n = scene["n"]
    heights = scene["heights"]
    pipes = scene["pipes"]
    inflow = scene["inflow"]
    facts = [f"{labels[i]}顶沿{heights[i]}" for i in range(n)]
    facts.extend(f"{labels[i]}与{labels[i + 1]}之间管道{pipes[i]}" for i in range(n - 1))
    parts = [
        f"图上持续注水的是{labels[inflow]}。标注为：{'，'.join(facts)}。",
        "水位从注水容器上升，到达管道高度就连通邻瓶，直到某瓶顶沿溢出。",
    ]
    for _level, desc in sim["events"]:
        text = _relabel_event(desc, labels)
        if not text.endswith("。"):
            text += "。"
        parts.append(text)
    first = labels[sim["first_full"]]
    never = [labels[i] for i in sim["never_filled"]]
    if task in (TASK_OVERFLOW, "first_full"):
        parts.append(f"最先装满并从顶沿溢出的是{first}。")
    elif task in (TASK_NEVER, "never_filled_or_final"):
        if never:
            parts.append(f"始终接不到水的是{'、'.join(never)}。")
        else:
            parts.append("所有容器最终都能进水，没有接不到水的容器。")
    elif task in (TASK_LEVEL, "final_level"):
        parts.append(f"连通水域最终水面高度等于最先溢出处的顶沿，为{sim['final_level']}。")
    else:
        raise KeyError(task)
    return "".join(parts)


def build_question(scene, sim, rng, idx):
    labels = scene.get("labels") or LABELS[: scene["n"]]
    q_type = rng.choice(["first_full", "never_filled_or_final", "final_level"])

    if q_type == "first_full" or (q_type == "never_filled_or_final" and not sim["never_filled"]):
        return vision_question(scene, TASK_OVERFLOW), labels[sim["first_full"]], TASK_OVERFLOW

    if q_type == "never_filled_or_final":
        never = sim["never_filled"]
        return vision_question(scene, TASK_NEVER), "、".join(labels[i] for i in never), TASK_NEVER

    return vision_question(scene, TASK_LEVEL), str(sim["final_level"]), TASK_LEVEL


# ------------------------------------------------------------------
# 5. 单条样本生成
# ------------------------------------------------------------------

def generate_one(idx, out_dir, rng, n_vessels=None):
    scene = generate_scene(rng, n_vessels)
    sim = simulate(scene)
    question, answer, task = build_question(scene, sim, rng, idx)

    img_name = f"vessels_{idx:04d}.png"
    img_path = os.path.join(out_dir, img_name)
    render_scene(scene, img_path)

    scene_params = {
        "n": scene["n"],
        "labels": LABELS[: scene["n"]],
        "heights": scene["heights"],
        "widths": scene["widths"],
        "pipes": scene["pipes"],
        "inflow": scene["inflow"],
        "q_type": task,
    }

    rec = {
        "id": f"vessels_{idx:04d}_{task}",
        "scene_id": f"vessels_{idx:04d}",
        "task": task,
        "split": "train",
        "use": "sft",
        "question": question,
        "image": img_name,
        "answer": answer,
        "rationale": vessels_rationale(scene_params, task),
        "rationale_source": "physics",
        "scene": scene_params,
        "answer_source": "physics",
        "license": "CC-BY-4.0",
        "source": "synthetic_vessels",
    }
    debug = {"scene": scene, "sim": sim, "q_type": task}
    return rec, debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="./vessels_v2_sample50")
    ap.add_argument("--jsonl", type=str, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-vessels", type=int, default=None,
                    help="固定生成的容器数，例如 4、5 或 6")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = args.jsonl or os.path.join(args.out_dir, "records.jsonl")

    rng = random.Random(args.seed)

    records = []
    debug_infos = []
    for i in range(args.count):
        rec, dbg = generate_one(i, args.out_dir, rng, args.n_vessels)
        records.append(rec)
        debug_infos.append(dbg)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"生成完成: {len(records)} 条样本 -> {jsonl_path}")
    print(f"图片输出目录: {args.out_dir}")

    for i in range(min(4, len(records))):
        print("\n--- 样本", i, "---")
        print("场景:", debug_infos[i]["scene"])
        print("模拟结果:", debug_infos[i]["sim"])
        print("题目:", records[i]["question"])
        print("答案:", records[i]["answer"])


if __name__ == "__main__":
    main()
