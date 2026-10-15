import numpy as np
from PIL import Image, ImageDraw

from src.braille_tables import decode_cells, name_to_dots


def reading_order(detections):
    if not detections:
        return []
    med_h = float(np.median([d["height"] for d in detections]))
    rows = []
    for d in sorted(detections, key=lambda d: d["y"]):
        if rows and abs(d["y"] - rows[-1]["y"]) <= 0.5 * med_h:
            rows[-1]["items"].append(d)
            rows[-1]["y"] = float(np.mean([q["y"] for q in rows[-1]["items"]]))
        else:
            rows.append({"y": d["y"], "items": [d]})
    return [sorted(r["items"], key=lambda d: d["x"]) for r in rows]


def detections_to_text(detections):
    lines = []
    for row in reading_order(detections):
        widths = [d["width"] for d in row]
        med_w = float(np.median(widths))
        lefts = sorted(d["x"] for d in row)
        steps = [b - a for a, b in zip(lefts, lefts[1:]) if b - a > 0.5 * med_w]
        pitch = float(np.median(steps)) if steps else med_w * 1.5
        cells = []
        prev_x = None
        for d in row:
            if prev_x is not None:
                gap_cells = int(round((d["x"] - prev_x) / pitch)) - 1
                cells += [None] * min(1, max(0, gap_cells))
            dots = d.get("dots") or name_to_dots(d.get("class", ""))
            cells.append(tuple(dots) if dots else ())
            prev_x = d["x"]
        lines.append(decode_cells([c if c else None for c in cells]))
    return "\n".join(l.strip() for l in lines if l.strip())


def draw_detections(image, detections):
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
    pil = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)
    lw = max(2, int(round(min(pil.size) / 300)))
    for d in detections:
        x1, y1 = d["x"] - d["width"] / 2, d["y"] - d["height"] / 2
        x2, y2 = d["x"] + d["width"] / 2, d["y"] + d["height"] / 2
        conf = float(d.get("confidence", 1.0))
        color = (39, 174, 96) if conf >= 0.6 else ((243, 156, 18) if conf >= 0.35 else (192, 57, 43))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = str(d.get("class", ""))
        if label and ord(label[0]) < 0x2800:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)
    return np.array(pil)
