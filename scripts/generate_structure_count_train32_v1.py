#!/usr/bin/env python3
"""Generate 32 verified SFT samples for discrete visual structure counting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from generate_lowres_structure_probe_v1 import (
    fan, grid, marked_square_count, point_set, render, square_count, svg, t_points,
)


def boxed(reason: str, answer: int) -> str:
    return f"<think>\n{reason}\n</think>\n\n最终答案：\\boxed{{{answer}}}"


def build_specs():
    out = []
    for r, c in [(2,4),(4,5),(5,4),(2,7)]:
        ans=(r+1)*math.comb(c+1,2)+(c+1)*math.comb(r+1,2)
        out.append(("grid_segments",{"rows":r,"cols":c},grid(r,c),"数出图中所有不同的水平或竖直线段。",ans,f"横线有{r+1}条，每条{c+1}个端点；竖线有{c+1}条，每条{r+1}个端点。按端点组合计数并相加。"))
    for r,c in [(2,6),(4,4),(5,3),(4,6)]:
        ans=square_count(r,c)
        out.append(("grid_squares",{"rows":r,"cols":c},grid(r,c),"数出图中所有大小的正方形。",ans,f"分别枚举边长1到{min(r,c)}格的正方形，各层位置数相加，避免只数最小方格。"))
    for n in [6,7,9,10]:
        ans=math.comb(n,2)
        out.append(("point_pairs",{"points":n},point_set(n),"观察图中的点。任取两个点作为端点，一共能确定多少条线段？",ans,f"图中共有{n}个点，每条线段由两个不同端点唯一确定，所以计算组合数C({n},2)。"))
    for h,v in [(5,4),(7,3),(5,5),(8,4)]:
        total=h+v-1; ans=math.comb(total,3)-math.comb(h,3)-math.comb(v,3)
        out.append(("t_noncollinear",{"horizontal":h,"vertical":v},t_points(h,v),"任取图中三个不共线的点，一共能组成多少个三角形？",ans,f"先从{total}个点任选3个，再减去横线上{h}点及竖线上{v}点产生的共线三点组。"))
    for rays in [4,5,7,8]:
        ans=math.comb(rays,2)
        out.append(("fan_triangles",{"rays":rays},fan(rays,1),"观察图形，图中共有多少个三角形？",ans,f"所有三角形共用顶点；从底边的{rays}个交点中任选两个作为底边端点，共C({rays},2)个。"))
    for rays,layers in [(5,2),(7,2),(5,3),(7,3)]:
        ans=layers*math.comb(rays,2)
        out.append(("layered_fan",{"rays":rays,"layers":layers},fan(rays,layers),"观察图形，图中共有多少个三角形？",ans,f"每条横线都提供C({rays},2)个以顶点为公共顶点的三角形，共{layers}层，分层相加。"))
    for r,c,rr,cc in [(3,5,2,2),(5,5,3,4),(4,6,3,2),(5,6,2,5)]:
        ans=marked_square_count(r,c,rr,cc)
        out.append(("marked_squares",{"rows":r,"cols":c,"marked":[rr,cc]},grid(r,c,marked=(rr,cc)),"数出所有包含星号所在小格的正方形。",ans,"按正方形边长逐层枚举，并只保留行、列范围都覆盖星号小格的位置。"))
    for r,c,rr,cc in [(4,6,2,2),(6,5,4,3),(4,7,3,5),(6,7,2,6)]:
        ans=rr*(r-rr+1)*cc*(c-cc+1)
        out.append(("marked_rectangles",{"rows":r,"cols":c,"marked":[rr,cc]},grid(r,c,marked=(rr,cc)),"数出所有包含星号所在小格的长方形（包括正方形）。",ans,f"上、下、左、右边界分别有{rr}、{r-rr+1}、{cc}、{c-cc+1}种选择，四项相乘。"))
    assert len(out)==32
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); images=args.output_dir/"images"; images.mkdir(exist_ok=True); svgs=args.output_dir/"svg_sources"; svgs.mkdir(exist_ok=True)
    records=[]; manifest=[]
    for i,(mechanism,params,body,question,answer,reason) in enumerate(build_specs(),1):
        tid=f"struct_train_{i:03d}"; variant="low" if i%2==0 else "high"; size=144 if variant=="low" else 720
        source=svgs/f"{tid}.svg"; source.write_text(svg(body),encoding="utf-8")
        image=images/f"{tid}_{variant}.png"; render(source,image,size)
        assistant=boxed(reason,answer)
        record={"message":[
            {"dialog":"","need_mask":True,"type":"bos"},
            {"dialog":"<|im_start|>user\n","need_mask":True,"type":"user_prefix"},
            {"dialog":f"<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n","need_mask":True,"type":"user_content","image_filenames":[str(image.relative_to(args.output_dir))]},
            {"dialog":"<|im_start|>assistant\n","need_mask":True,"type":"assistant_prefix"},
            {"dialog":assistant+"<|im_end|>\n","need_mask":False,"type":"assistant_content"},
            {"dialog":"<|endoftext|>","need_mask":True,"type":"eos"}],
            "trace_id":tid,"_source_file":"structure_count_train32_v1","_dimension":"结构识别与组合计数","mechanism":mechanism,"variant":variant,"reference_answer":str(answer)}
        records.append(record); manifest.append({"trace_id":tid,"mechanism":mechanism,"variant":variant,"params":params,"answer":answer,"image":str(image.relative_to(args.output_dir))})
    assert sum(x["variant"]=="high" for x in manifest)==16 and sum(x["variant"]=="low" for x in manifest)==16
    (args.output_dir/"train32.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records),encoding="utf-8")
    (args.output_dir/"manifest.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in manifest),encoding="utf-8")
    print(json.dumps({"n":32,"high":16,"low":16,"mechanisms":len(set(x["mechanism"] for x in manifest))},ensure_ascii=False))


if __name__=="__main__": main()
