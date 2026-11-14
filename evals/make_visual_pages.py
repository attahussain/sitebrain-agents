"""Generate fictional construction document pages as images for the visual retrieval benchmark.

Every answer in evals/visual_queries.jsonl exists only inside a page image: drawing title blocks,
annotations on a column grid, schedule tables, ticked checklist boxes, a site plan and a progress
chart. Near-identical distractor pages (the same drawing for other levels, the same checklist for
other pours) make retrieval non-trivial, as in a real drawing register.

Deterministic. Writes data/pages/*.png (committed, so the demo needs no generation step) and
evals/visual_queries.jsonl.

Usage:  python evals/make_visual_pages.py
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pages"
QUERIES = Path(__file__).with_name("visual_queries.jsonl")
W, H = 1240, 1754  # A4 at 150 dpi


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _page(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, W - 40, H - 40], outline="black", width=4)
    d.text((80, 70), "AL NOOR TOWER (fictional demo project)", font=_font(28, True), fill="black")
    size = 36
    while _font(size, True).getlength(title) > W - 160 and size > 18:  # keep the title inside the border
        size -= 1
    d.text((80, 115), title, font=_font(size, True), fill="black")
    return img, d


def _title_block(d: ImageDraw.ImageDraw, number: str, rev: str, scale: str, date: str) -> None:
    x0, y0 = W - 560, H - 300
    d.rectangle([x0, y0, W - 60, H - 60], outline="black", width=3)
    rows = [("DRAWING No.", number), ("REVISION", rev), ("SCALE", scale), ("DATE", date)]
    for i, (k, v) in enumerate(rows):
        y = y0 + 15 + i * 55
        d.text((x0 + 20, y), k, font=_font(22), fill="black")
        d.text((x0 + 250, y), v, font=_font(30, True), fill="black")


def drawing(level: int, rev: str, big_col: int, size: str) -> Image.Image:
    img, d = _page(f"STRUCTURAL - LEVEL {level} COLUMN LAYOUT")
    ox, oy, step = 140, 260, 170
    for r in range(3):
        for c in range(6):
            n = r * 6 + c + 1
            x, y = ox + c * step, oy + r * 260
            s = 70 if n == big_col else 44
            d.rectangle([x, y, x + s, y + s], fill="black")
            d.text((x - 10, y + s + 10), f"C{level}-{n:02d}", font=_font(22), fill="black")
    for i in range(7):
        d.line([ox - 30 + i * step, oy - 60, ox - 30 + i * step, oy + 760], fill="gray", width=1)
    d.text((140, 1120), f"NOTE: COLUMN C{level}-{big_col:02d} ENLARGED TO {size} (TRANSFER BEAM SUPPORT)",
           font=_font(26, True), fill="black")
    d.text((140, 1170), "ALL OTHER COLUMNS 500 x 500 UNLESS NOTED", font=_font(24), fill="black")
    _title_block(d, f"S-2{level:02d}", rev, "1:100", "10 OCT 2026")
    return img


def bending_schedule() -> Image.Image:
    img, d = _page("BAR BENDING SCHEDULE BBS-15 (LEVEL 15 COLUMNS)")
    header = ["BAR MARK", "MEMBER", "DIA (mm)", "No.", "LENGTH (mm)", "SHAPE"]
    rows = [["15C01", "C15 main bars", "25", "12", "3650", "00"],
            ["15C02", "C15 links", "10", "38", "2100", "51"],
            ["15C03", "C15-07 main bars", "32", "16", "3650", "00"],
            ["15C04", "C15-07 links", "12", "42", "2700", "51"]]
    xs = [80, 270, 560, 730, 860, 1060]
    y = 260
    for x, h in zip(xs, header):
        d.text((x, y), h, font=_font(24, True), fill="black")
    d.line([70, y + 45, W - 70, y + 45], fill="black", width=3)
    for row in rows:
        y += 80
        for x, v in zip(xs, row):
            d.text((x, y), v, font=_font(26), fill="black")
        d.line([70, y + 50, W - 70, y + 50], fill="gray", width=1)
    return img


def checklist(level: int, slump: int, cubes: bool, inspector: str) -> Image.Image:
    img, d = _page(f"CONCRETE POUR CHECKLIST - LEVEL {level} SLAB")
    items = [("Formwork inspected and approved", True), ("Reinforcement inspected and approved", True),
             ("Cube samples taken", cubes), ("Curing compound applied", True)]
    y = 280
    for text, ok in items:
        d.rectangle([90, y, 130, y + 40], outline="black", width=3)
        if ok:
            d.line([96, y + 20, 108, y + 34], fill="black", width=5)
            d.line([108, y + 34, 126, y + 6], fill="black", width=5)
        d.text((160, y + 2), text, font=_font(28), fill="black")
        y += 90
    d.text((90, y + 40), f"Slump test result: {slump} mm", font=_font(30, True), fill="black")
    d.text((90, y + 100), f"Inspected by: {inspector}", font=_font(30), fill="black")
    return img


def logistics_plan() -> Image.Image:
    img, d = _page("SITE LOGISTICS PLAN - PHASE 2")
    d.rectangle([380, 520, 860, 1100], outline="black", width=5)
    d.text((520, 790), "TOWER", font=_font(36, True), fill="black")
    for label, (x, y) in {"TC-1": (600, 360), "TC-2": (600, 1240), "TC-3": (1000, 800)}.items():
        d.ellipse([x - 40, y - 40, x + 40, y + 40], outline="black", width=5)
        d.text((x - 35, y + 50), label, font=_font(28, True), fill="black")
    d.text((560, 210), "NORTH", font=_font(28, True), fill="black")
    for label, (x, y) in {"GATE 1": (120, 820), "GATE 2": (560, 1420)}.items():
        d.rectangle([x, y, x + 150, y + 50], fill="black")
        d.text((x + 10, y + 60), label, font=_font(26, True), fill="black")
    d.text((100, 1560), "CONCRETE PUMP STATION ADJACENT TO GATE 2", font=_font(26), fill="black")
    return img


def progress_chart() -> Image.Image:
    img, d = _page("STRUCTURE PROGRESS - PLANNED vs ACTUAL SLAB COMPLETION")
    levels = [(12, "02 OCT", "02 OCT"), (13, "08 OCT", "09 OCT"), (14, "14 OCT", "12 OCT"), (15, "20 OCT", "24 OCT")]
    y = 300
    for lvl, plan, actual in levels:
        d.text((80, y), f"LEVEL {lvl}", font=_font(28, True), fill="black")
        d.rectangle([300, y, 300 + 60 * lvl - 300, y + 30], fill="gray")
        d.text((320 + 60 * lvl - 300, y), f"planned {plan}", font=_font(22), fill="black")
        d.rectangle([300, y + 40, 300 + 60 * lvl - 280, y + 70], fill="black")
        d.text((320 + 60 * lvl - 280, y + 40), f"actual {actual}", font=_font(22), fill="black")
        y += 160
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {
        "S-213": drawing(13, "B", 4, "650 x 650"),
        "S-214": drawing(14, "A", 11, "600 x 600"),
        "S-215": drawing(15, "C", 7, "700 x 700"),
        "S-216": drawing(16, "A", 2, "600 x 600"),
        "BBS-15": bending_schedule(),
        "CHK-L13": checklist(13, 95, True, "QA-01"),
        "CHK-L14": checklist(14, 110, True, "QA-03"),
        "CHK-L15": checklist(15, 120, False, "QA-02"),
        "LOG-P2": logistics_plan(),
        "PRG-STR": progress_chart(),
    }
    for page_id, img in pages.items():
        img.save(OUT / f"{page_id}.png", optimize=True)

    # (id, question, page holding the answer, facts the final answer must contain)
    queries = [
        ("v01", "Which drawing shows the Level 15 column layout?", "S-215", ["S-215"]),
        ("v02", "Which column on Level 15 is enlarged to support the transfer beam?", "S-215", ["C15-07"]),
        ("v03", "What size is the enlarged column on Level 13?", "S-213", ["650"]),
        ("v04", "What is the current revision of the Level 14 column layout drawing?", "S-214", ["S-214"]),
        ("v05", "What bar diameter is specified for the C15-07 main bars?", "BBS-15", ["32"]),
        ("v06", "How many links are scheduled for the Level 15 columns?", "BBS-15", ["38"]),
        ("v07", "What slump was recorded on the Level 14 slab pour checklist?", "CHK-L14", ["110"]),
        ("v08", "Which slab pour checklist shows cube samples were not taken?", "CHK-L15", ["15"]),
        ("v09", "Who inspected the Level 13 slab pour?", "CHK-L13", ["QA-01"]),
        ("v10", "Where is tower crane TC-2 positioned on the logistics plan?", "LOG-P2", ["south"]),
        ("v11", "Which gate is next to the concrete pump station?", "LOG-P2", ["Gate 2"]),
        ("v12", "Was the Level 15 slab completed on its planned date?", "PRG-STR", ["24"]),
    ]
    QUERIES.write_text("\n".join(json.dumps({"id": i, "question": q, "expected_pages": [p], "expected_keywords": k})
                                 for i, q, p, k in queries) + "\n", encoding="utf-8")
    print(f"wrote {len(pages)} pages to {OUT} and {len(queries)} queries to {QUERIES}")


if __name__ == "__main__":
    main()
