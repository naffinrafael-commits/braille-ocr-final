from PIL import Image, ImageDraw
from src.braille_tables import text_to_cells


def render_braille(text, dot_radius=10, spacing=1.75, pitch=4.3, line_pitch=7.0, max_cells=20,
                   show_empty=True, ink=(20, 20, 20), empty=(222, 222, 222), background=(255, 255, 255),
                   margin=None, labels=False):
    D = 2 * dot_radius
    s = spacing * D
    P = pitch * D
    L = line_pitch * D
    if margin is None:
        margin = int(2 * D)
    lines = []
    for raw in text.split("\n"):
        row = []
        for word in raw.split(" "):
            wc = text_to_cells(word)
            if not wc:
                continue
            need = len(wc) + (1 if row else 0)
            if row and len(row) + need > max_cells:
                lines.append(row)
                row = []
            if row:
                row.append((" ", ()))
            row.extend(wc)
        lines.append(row)
    lines = [l for l in lines if l] or [[]]
    n_cols = max(1, max(len(l) for l in lines))
    width = int(2 * margin + (n_cols - 1) * P + s + D)
    height = int(2 * margin + (len(lines) - 1) * L + 2 * s + D + (D if labels else 0))
    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    for li, row in enumerate(lines):
        for ci, (label, dots) in enumerate(row):
            x0 = margin + ci * P + dot_radius
            y0 = margin + li * L + dot_radius
            is_blank = len(dots) == 0
            for n in range(1, 7):
                col = 0 if n <= 3 else 1
                rr = (n - 1) % 3
                cx = x0 + col * s
                cy = y0 + rr * s
                if n in dots:
                    color = ink
                elif show_empty and not is_blank:
                    color = empty
                else:
                    continue
                draw.ellipse([cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius], fill=color)
            if labels and not is_blank:
                draw.text((x0, y0 + 2 * s + dot_radius + 2), label, fill=(90, 90, 90))
    return img
