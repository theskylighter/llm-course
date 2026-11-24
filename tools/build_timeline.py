#!/usr/bin/env python3
"""Generate the Research Papers Journey Excalidraw timeline from a paper list.

Writes web/src/data/research_papers_timeline_elements.json (rendered by ExcalidrawTimeline.tsx)
and content/diagrams/research_papers_timeline.excalidraw (editable on excalidraw.com).
Edit PAPERS / LANES below and re-run: `python3 tools/build_timeline.py`.

Layout (horizontal, sized for a laptop-width content area): lane labels on the left, one
column per distinct year, one lane per category. Papers sharing a lane and year stack
vertically. Arrows come from EDGES and are routed at right angles around the boxes. Box labels are
short (wrapped to two lines when needed) because the column header already gives the year.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# lane id -> (label, stroke, fill)
LANES = {
    "foundation": ("Foundation", "#b45309", "#fef3c7"),
    "moe": ("MoE /\nConditional", "#475569", "#f1f5f9"),
    "encoder": ("Encoder /\nSeq2Seq", "#15803d", "#dcfce7"),
    "decoder": ("Decoder /\nAlignment", "#0f766e", "#ccfbf1"),
    "efficiency": ("Efficiency /\nAttention", "#7e22ce", "#f3e8ff"),
    "position": ("Positional\nEncoding", "#c2410c", "#ffedd5"),
}

# (id, box label, year, lane, link) - order within a lane+year is top-to-bottom order.
PAPERS = [
    ("transformer", "Transformer", 2017, "foundation", "https://arxiv.org/abs/1706.03762"),
    ("adaptive_moe", "Adaptive\nMoE", 1991, "moe", "https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf"),
    ("moe_2017", "Sparse\nMoE", 2017, "moe", "https://arxiv.org/abs/1701.06538"),
    ("glm5", "GLM-5", 2026, "moe", "https://arxiv.org/abs/2602.15763"),
    ("elmo", "ELMo", 2018, "encoder", "https://arxiv.org/abs/1802.05365"),
    ("bert", "BERT", 2018, "encoder", "https://arxiv.org/abs/1810.04805"),
    ("t5", "T5", 2019, "encoder", "https://arxiv.org/abs/1910.10683"),
    ("bart", "BART", 2019, "encoder", "https://arxiv.org/abs/1910.13461"),
    ("gpt1", "GPT-1", 2018, "decoder", "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"),
    ("gpt3", "GPT-3", 2020, "decoder", "https://arxiv.org/abs/2005.14165"),
    ("instructgpt", "Instruct\nGPT", 2022, "decoder", "https://arxiv.org/abs/2203.02155"),
    ("picl", "PICL", 2023, "decoder", "https://arxiv.org/abs/2305.09137"),
    ("dpo", "DPO", 2023, "decoder", "https://arxiv.org/abs/2305.18290"),
    ("it_survey", "IT\nSurvey", 2023, "decoder", "https://arxiv.org/abs/2308.10792"),
    ("lora", "LoRA", 2021, "efficiency", "https://arxiv.org/abs/2106.09685"),
    ("flashattn", "Flash\nAttention", 2022, "efficiency", "https://arxiv.org/abs/2205.14135"),
    ("gqa", "GQA", 2023, "efficiency", "https://arxiv.org/abs/2305.13245"),
    ("swat", "SWAT", 2025, "efficiency", "https://arxiv.org/abs/2502.18845"),
    ("pos_survey", "Position\nSurvey", 2021, "position", "https://arxiv.org/abs/2102.11090"),
    ("rope", "RoPE", 2021, "position", "https://arxiv.org/abs/2104.09864"),
]

# Transitions, each stated in content/lessons/11_6_research_papers_journey.md.
# Same-lane arrows are solid (direct succession); cross-lane ones are dashed (influence).
EDGES = [
    # Transformer family tree: encoder, decoder, encoder-decoder stacks + modifications
    ("transformer", "bert"), ("transformer", "gpt1"), ("transformer", "t5"), ("transformer", "rope"),
    # MoE thread 1991 -> 2017 -> 2026
    ("adaptive_moe", "moe_2017"), ("moe_2017", "glm5"),
    # Contextual fork ELMo -> BERT + GPT-1 (GPT-1 came first); BART = BERT-style encoder + GPT-style decoder
    ("elmo", "bert"), ("elmo", "gpt1"), ("gpt1", "bert"), ("t5", "bart"), ("bert", "bart"), ("gpt1", "bart"),
    # Scale to alignment; ICL shown at scale -> trained explicitly
    ("gpt1", "gpt3"), ("gpt3", "instructgpt"), ("gpt3", "picl"), ("instructgpt", "dpo"), ("dpo", "it_survey"),
    # Efficiency stack; SWAT uses RoPE; efficient attention feeds the modern MoE stack
    ("lora", "flashattn"), ("lora", "gqa"), ("gqa", "swat"), ("rope", "swat"), ("swat", "glm5"),
    ("pos_survey", "rope"),
]

LABEL_W, BOX_W, BOX_H = 116, 92, 46
GAP = 22                    # space between year columns (fits an arrowhead)...
WIDE_GAP = {2018: 48, 2019: 40}  # ...wider where many cross-lane arrows turn
HEAD_H, LANE_PAD, STACK_GAP = 48, 14, 22
BOX_FONT, LABEL_FONT = 16, 15
FONT = 5  # Excalifont (hand-drawn, Excalidraw 0.18 default)
INK, MUTED, RULE = "#1e293b", "#475569", "#cbd5e1"

_seed = 1000


def base(id_, type_, x, y, w, h, **kw):
    global _seed
    _seed += 1
    el = {
        "id": id_, "type": type_, "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None, "seed": _seed, "version": 1,
        "versionNonce": _seed, "isDeleted": False, "boundElements": [], "updated": 1,
        "link": None, "locked": False,
    }
    el.update(kw)
    return el


def text(id_, s, x, y, w, size, color=INK, align="left", valign="top", container=None):
    h = len(s.split("\n")) * size * 1.25
    return base(id_, "text", x, y, w, h, strokeColor=color, text=s, originalText=s, fontSize=size,
                fontFamily=FONT, textAlign=align, verticalAlign=valign, containerId=container,
                lineHeight=1.25, autoResize=True, boundElements=None)


def line(id_, x, y, w, color, style="solid", width=1):
    return base(id_, "line", x, y, w, 0, strokeColor=color, strokeStyle=style, strokeWidth=width,
                points=[[0, 0], [w, 0]], lastCommittedPoint=None, startBinding=None,
                endBinding=None, startArrowhead=None, endArrowhead=None, boundElements=None)


def _hits(seg, rect, pad=3):
    """Does an axis-aligned segment pass through a rectangle (inflated by pad)?"""
    (x0, y0), (x1, y1) = seg
    rx0, ry0 = rect["x"] - pad, rect["y"] - pad
    rx1, ry1 = rect["x"] + rect["width"] + pad, rect["y"] + rect["height"] + pad
    return min(x0, x1) < rx1 and max(x0, x1) > rx0 and min(y0, y1) < ry1 and max(y0, y1) > ry0


MIN_LAST = 16  # Excalidraw shrinks an arrowhead to half its last segment, so keep that segment long


def _overlaps(s1, s2, tol=2, min_len=4):
    """Do two axis-aligned segments run along each other?"""
    (a0, a1), (b0, b1) = s1, s2
    if abs(a0[0] - a1[0]) < 1 and abs(b0[0] - b1[0]) < 1 and abs(a0[0] - b0[0]) < tol:  # both vertical
        lo, hi = max(min(a0[1], a1[1]), min(b0[1], b1[1])), min(max(a0[1], a1[1]), max(b0[1], b1[1]))
        return hi - lo > min_len
    if abs(a0[1] - a1[1]) < 1 and abs(b0[1] - b1[1]) < 1 and abs(a0[1] - b0[1]) < tol:  # both horizontal
        lo, hi = max(min(a0[0], a1[0]), min(b0[0], b1[0])), min(max(a0[0], a1[0]), max(b0[0], b1[0]))
        return hi - lo > min_len
    return False


def route(a, b, boxes, channels, drawn, cross, gap=3):
    """Cheapest right-angle path from box a to box b: clears every other box, ends in a segment long
    enough for a visible arrowhead, and avoids running along arrows already drawn (`drawn` holds
    (src, dst, cross, segment) for those; `cross` marks a cross-lane, dashed arrow)."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
    acx, acy, bcx, bcy = (ax0 + ax1) / 2, (ay0 + ay1) / 2, (bx0 + bx1) / 2, (by0 + by1) / 2
    offs = (9, 14, 20, 26, 32, 40)                             # turn positions in a gutter
    a_out, b_in = (ax1 + gap, acy), (bx0 - gap, bcy)
    a_v = (acx, ay1 + gap) if bcy > acy else (acx, ay0 - gap)   # leave a vertically, toward b
    b_v = (bcx, by0 - gap) if bcy > acy else (bcx, by1 + gap)   # enter b vertically, from a's side
    cands = []
    if abs(acx - bcx) < 1:
        cands.append([a_v, b_v])                                   # same column: straight
    if abs(acy - bcy) < 1:
        cands.append([a_out, b_in])                                # same row: straight
    for gx in sorted({ax1 + k for k in offs} | {bx0 - k for k in offs}):
        if a_out[0] < gx < b_in[0]:
            cands.append([a_out, (gx, acy), (gx, bcy), b_in])      # jog through a column gutter
    cands.append([a_out, (bcx, acy), b_v])                         # across, then into b's top/bottom
    cands.append([a_v, (acx, bcy), b_in])                          # down/up, then across into b
    for cy in channels:                                            # side -> free channel -> side
        cands.append([(acx, ay1 + gap if cy > acy else ay0 - gap), (acx, cy), (bcx, cy),
                      (bcx, by0 - gap if cy < bcy else by1 + gap)])
        for k in offs:
            cands.append([a_out, (ax1 + k, acy), (ax1 + k, cy), (bcx, cy),
                          (bcx, by0 - gap if cy < bcy else by1 + gap)])
    for k in offs:                                                 # around the left of a stack:
        gl = min(ax0, bx0) - k
        cands.append([(ax0 - gap, acy), (gl, acy), (gl, bcy), (bx0 - gap, bcy)])   # into b's left
        for yb in (by1 + 2 * MIN_LAST, by0 - 2 * MIN_LAST):                         # or its far side
            cands.append([(ax0 - gap, acy), (gl, acy), (gl, yb), (bcx, yb),
                          (bcx, by1 + gap if yb > by1 else by0 - gap)])
    others = [r for r in boxes if r is not a and r is not b]

    def segs(pts):
        return list(zip(pts, pts[1:]))

    def ok(pts):
        if any(p == q for p, q in segs(pts)):
            return False
        last = abs(pts[-1][0] - pts[-2][0]) + abs(pts[-1][1] - pts[-2][1])
        return last >= MIN_LAST and not any(_hits(sg, r) for sg in segs(pts) for r in others)

    def cost(pts):
        length = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in segs(pts))
        # Running along another arrow makes two arrows read as one - except that dashed arrows
        # fanning out of (or into) the same box should share a trunk, like a tree.
        clash = sum(0 if (cross and c2 and (s2 == a["id"] or d2 == b["id"])) else 1000
                    for sg in segs(pts) for (s2, d2, c2, other) in drawn if _overlaps(sg, other))
        return length + 60 * (len(pts) - 2) + clash

    good = [c for c in cands if ok(c)]
    if not good:
        raise SystemExit(f"no clear route for {a['id']} -> {b['id']}")
    best = min(good, key=cost)
    drawn += [(a["id"], b["id"], cross, sg) for sg in segs(best)]
    return best


def build():
    years = sorted({p[2] for p in PAPERS})
    col_x, x = {}, LABEL_W
    for y in years:  # left edge of each column's boxes
        x += WIDE_GAP.get(y, GAP)
        col_x[y] = x
        x += BOX_W
    right = x + GAP
    bg, boxes_els, arrows = [], [], []   # z-order: background, arrows, boxes
    rects, lane_of, channels = {}, {}, []

    # Year header + axis.
    for y in years:
        bg.append(text(f"year_{y}", str(y), col_x[y] - 10, 10, BOX_W + 20, 18, MUTED, align="center"))
    bg.append(line("time_axis", LABEL_W, HEAD_H - 10, right - LABEL_W, MUTED, width=2))

    lane_y = HEAD_H
    for lane, (label, stroke, fill) in LANES.items():
        groups = {}
        for p in PAPERS:
            if p[3] == lane:
                groups.setdefault(p[2], []).append(p)
        depth = max(len(g) for g in groups.values())
        lane_h = depth * BOX_H + (depth - 1) * STACK_GAP + 2 * LANE_PAD

        lbl = text(f"lane_{lane}", label, 0, 0, LABEL_W - 12, LABEL_FONT, stroke)
        lbl["y"] = lane_y + (lane_h - lbl["height"]) / 2
        bg.append(lbl)
        channels.append(lane_y + LANE_PAD / 2)
        channels += [lane_y + LANE_PAD + i * (BOX_H + STACK_GAP) - STACK_GAP / 2 for i in range(1, depth)]

        for year in sorted(groups):
            for i, (pid, title, _, _, link) in enumerate(groups[year]):
                # Excalifont is ~0.58em per char; a label that won't fit widens its box evenly
                # into the gutters (fine for a lone box such as Transformer).
                fs = BOX_FONT
                bw = max(BOX_W, round(max(map(len, title.split("\n"))) * fs * 0.58 + 8))
                bx = col_x[year] - (bw - BOX_W) / 2
                by = lane_y + LANE_PAD + i * (BOX_H + STACK_GAP)
                rid, tid = f"rect_{pid}", f"text_{pid}"
                rect = base(rid, "rectangle", bx, by, bw, BOX_H, strokeColor=stroke,
                            backgroundColor=fill, strokeWidth=2, roundness={"type": 3}, link=link,
                            boundElements=[{"type": "text", "id": tid}])
                t = text(tid, title, bx + 4, 0, bw - 8, fs, INK,
                         align="center", valign="middle", container=rid)
                t["y"] = by + (BOX_H - t["height"]) / 2
                boxes_els += [rect, t]
                rects[pid], lane_of[pid] = rect, lane

        lane_y += lane_h
        if lane != list(LANES)[-1]:
            bg.append(line(f"lane_sep_{lane}", 0, lane_y, right, RULE, style="dashed"))
    channels.append(lane_y - LANE_PAD / 2)

    drawn = []
    # Route solid same-lane arrows first so the main storyline gets the straight paths.
    for src, dst in sorted(EDGES, key=lambda e: lane_of[e[0]] != lane_of[e[1]]):
        a, b = rects[src], rects[dst]
        same = lane_of[src] == lane_of[dst]
        pts = route(a, b, list(rects.values()), channels, drawn, cross=not same)
        aid = f"arrow_{src}__{dst}"
        x0, y0 = pts[0]
        rel = [[px - x0, py - y0] for px, py in pts]
        xs, ys = [p[0] for p in rel], [p[1] for p in rel]
        arrows.append(base(aid, "arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
                           strokeColor=LANES[lane_of[src]][1], strokeWidth=2 if same else 1.5,
                           strokeStyle="solid" if same else "dashed", roughness=0 if not same else 1,
                           points=rel, lastCommittedPoint=None, startArrowhead=None, endArrowhead="arrow",
                           startBinding={"elementId": a["id"], "focus": 0, "gap": 3},
                           endBinding={"elementId": b["id"], "focus": 0, "gap": 3},
                           boundElements=None))
        a["boundElements"].append({"type": "arrow", "id": aid})
        b["boundElements"].append({"type": "arrow", "id": aid})

    els = bg + arrows + boxes_els
    # Fractional indices in z-order: a0..az, then b00.. (same scheme as fractional-indexing).
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for i, el in enumerate(els):
        el["index"] = "a" + digits[i] if i < 62 else "b" + digits[(i - 62) // 62] + digits[(i - 62) % 62]
    return els


def main():
    els = build()
    out = ROOT / "web/src/data/research_papers_timeline_elements.json"
    out.write_text(json.dumps(els, indent=2) + "\n")
    scene = ROOT / "content/diagrams/research_papers_timeline.excalidraw"
    doc = json.loads(scene.read_text()) if scene.exists() else {"type": "excalidraw", "version": 2}
    doc.update(source="tools/build_timeline.py", elements=els)
    doc.setdefault("appState", {"viewBackgroundColor": "#ffffff"})
    doc.setdefault("files", {})
    scene.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{len(els)} elements -> {out.relative_to(ROOT)}, {scene.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
