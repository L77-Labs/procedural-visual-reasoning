#!/usr/bin/env python3
"""Generate deterministic high/low-resolution paired structure-counting probes."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# PIL 仅渲染 PNG 需要；公式函数可无 Pillow 自检。


def line(x1, y1, x2, y2, width=4):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111" stroke-width="{width}"/>'


def circle(x, y, r=6):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="#111"/>'


def svg(body, size=720):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 720 720">
<rect width="720" height="720" fill="white"/>
<g stroke-linecap="round" stroke-linejoin="round">{body}</g>
</svg>'''


def grid(rows, cols, *, marked=None, missing_top_right=False):
    x0, y0, step = 110, 115, min(500 // cols, 440 // rows)
    parts = []
    for r in range(rows + 1):
        for c in range(cols):
            if missing_top_right and r == 0 and c == cols - 1:
                continue
            parts.append(line(x0 + c * step, y0 + r * step, x0 + (c + 1) * step, y0 + r * step))
    for c in range(cols + 1):
        for r in range(rows):
            if missing_top_right and r == 0 and c == cols:
                continue
            parts.append(line(x0 + c * step, y0 + r * step, x0 + c * step, y0 + (r + 1) * step))
    if marked:
        r, c = marked
        cx, cy = x0 + (c - .5) * step, y0 + (r - .5) * step
        points = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = step * (.27 if i % 2 == 0 else .11)
            points.append(f"{cx + rr * math.cos(a):.1f},{cy + rr * math.sin(a):.1f}")
        parts.append(f'<polygon points="{" ".join(points)}" fill="none" stroke="#111" stroke-width="4"/>')
    return "".join(parts)


def point_set(n):
    pts = [(115 + i * 70, 390) for i in range(n - 3)] + [(220, 220), (430, 205), (520, 520)]
    return "".join(circle(x, y, 8) for x, y in pts)


def t_points(horizontal, vertical):
    cx, cy, step = 360, 260, 65
    parts = []
    xs = [cx + (i - (horizontal - 1) / 2) * step for i in range(horizontal)]
    parts.append(line(xs[0], cy, xs[-1], cy))
    ys = [cy + i * step for i in range(vertical)]
    parts.append(line(cx, ys[0], cx, ys[-1]))
    parts.extend(circle(x, cy) for x in xs)
    # The crossing is an explicit selectable point even when `horizontal` is
    # even and no horizontal sample lands exactly at cx.
    parts.append(circle(cx, cy))
    parts.extend(circle(cx, y) for y in ys[1:])
    return "".join(parts)


def fan(rays, layers=1):
    apex, y0, left, right = (360, 100), 590, 90, 630
    bottoms = [(left + i * (right-left)/(rays-1), y0) for i in range(rays)]
    parts = [line(apex[0], apex[1], x, y) for x, y in bottoms]
    for layer in range(1, layers + 1):
        frac = layer / layers
        y = apex[1] + frac * (y0-apex[1])
        xl = apex[0] + frac * (left-apex[0]); xr = apex[0] + frac * (right-apex[0])
        parts.append(line(xl, y, xr, y))
    return "".join(parts)


def square_count(rows, cols):
    return sum((rows-k+1)*(cols-k+1) for k in range(1, min(rows, cols)+1))


def marked_square_count(rows, cols, rr, cc):
    total = 0
    for k in range(1, min(rows, cols)+1):
        for r in range(1, rows-k+2):
            for c in range(1, cols-k+2):
                total += r <= rr < r+k and c <= cc < c+k
    return total


def specs():
    return [
        ("grid_segments", grid(3,5), "观察图中的方格。图中一共有多少条不同的水平或竖直线段？", 4*math.comb(6,2)+6*math.comb(4,2)),
        ("grid_squares", grid(3,7), "观察图中的方格。图中一共有多少个正方形？", square_count(3,7)),
        ("missing_corner_squares", grid(5,5,missing_top_right=True), "观察这个缺角方格图形。图中共有多少个正方形？", square_count(5,5)-5),
        ("point_pairs", point_set(8), "观察图中的点。任取两个点作为端点，一共能确定多少条线段？", math.comb(8,2)),
        ("t_noncollinear", t_points(6,5), "观察图中的点阵。任取三个不共线的点，一共能组成多少个三角形？", math.comb(10,3)-math.comb(6,3)-math.comb(5,3)),
        ("fan_triangles", fan(6,1), "观察图形，图中共有多少个三角形？", math.comb(6,2)),
        ("fan_segment_delta", fan(6,1), "观察图形，图中线段总数比三角形总数多多少？", 6),
        ("layered_fan", fan(6,2), "观察图形，图中共有多少个三角形？", 2*math.comb(6,2)),
        ("marked_squares", grid(4,5,marked=(2,3)), "观察方格图，数出所有包含星号所在小格的正方形。", marked_square_count(4,5,2,3)),
        ("marked_rectangles", grid(5,7,marked=(2,4)), "观察方格图，数出所有包含星号所在小格的长方形（包括正方形）。", 2*4*4*4),
    ]


def render(svg_path: Path, png_path: Path, size: int):
    # Rasterize the deliberately small SVG subset used above. This avoids a
    # browser/Cairo dependency and ensures both variants derive from one scene.
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    scale = size / 720
    root = ET.parse(svg_path).getroot()
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        a = elem.attrib
        if tag == "line":
            xy = tuple(float(a[k]) * scale for k in ("x1", "y1", "x2", "y2"))
            draw.line(xy, fill="#111", width=max(1, round(float(a.get("stroke-width", 4)) * scale)))
        elif tag == "circle":
            x, y, r = (float(a[k]) * scale for k in ("cx", "cy", "r"))
            draw.ellipse((x-r, y-r, x+r, y+r), fill="#111")
        elif tag == "polygon":
            pts = [tuple(float(v) * scale for v in pair.split(",")) for pair in a["points"].split()]
            draw.line(pts + [pts[0]], fill="#111", width=max(1, round(float(a.get("stroke-width", 4)) * scale)))
    image.save(png_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    images = args.output_dir / "images"; images.mkdir(exist_ok=True)
    svg_dir = args.output_dir / "svg_sources"; svg_dir.mkdir(exist_ok=True)
    rows = []
    for idx, (mechanism, body, question, answer) in enumerate(specs(), 1):
        pair_id = f"lrsp_{idx:02d}"
        source = svg_dir / f"{pair_id}.svg"; source.write_text(svg(body), encoding="utf-8")
        for variant, size in (("high", 720), ("low", 144)):
            image = images / f"{pair_id}_{variant}.png"; render(source, image, size)
            rows.append({
                "trace_id": f"{pair_id}_{variant}", "pair_id": pair_id, "variant": variant,
                "mechanism": mechanism, "image": str(image.relative_to(args.output_dir)),
                "question": question, "answer": str(answer),
                "messages": [{"role":"user","content":[{"type":"image","image":str(image)}, {"type":"text","text":question}]}, {"role":"assistant","content":f"\\boxed{{{answer}}}"}],
            })
    (args.output_dir / "probe_pairs.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    manifest = {"version":"v1","pairs":10,"images":20,"high_size":720,"low_size":144,"generation":"same SVG scene per pair; only raster resolution differs","answers_verified_by":"deterministic formulas in generator"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False))


if __name__ == "__main__": main()
