import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from src.braille_tables import text_to_cells, dots_to_code

VOCAB = ("dor agua banheiro medico enfermeira remedio frio calor fome sede familia ajuda urgente falta ar "
         "peito cabeca nausea tontura posicao cama hospital paciente texto braille leito quarto exame "
         "coração ação não está você sábado criança pressão também açúcar então até já médico água "
         "posição família remédio náusea cabeça estômago respiração febre sangue socorro obrigado bom dia "
         "noite tarde preciso quero estou sinto muito pouco agora depois hoje amanhã mãe pai filho "
         "o a os as de do da em no na um uma com por para que se mais menos sim e ou").split()


def random_text(rng, max_words=40):
    words = []
    for _ in range(rng.randint(3, max_words)):
        w = rng.choice(VOCAB)
        if rng.random() < 0.15:
            w = w.capitalize()
        words.append(w)
        if rng.random() < 0.08:
            words.append(str(rng.randint(0, 2026)))
        if rng.random() < 0.08:
            words[-1] += rng.choice([",", ".", "?", "!", ":", ";"])
    return " ".join(words)


def layout(text, s, P, L, max_cells, margin, r, max_side=1400):
    max_cells = max(4, min(max_cells, int((max_side - 2 * margin - s - 2 * r) // P) + 1))
    max_lines = max(1, int((max_side - 2 * margin - 2 * s - 2 * r) // L) + 1)
    lines = []
    row = []
    for word in text.split(" "):
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
    if row:
        lines.append(row)
    lines = lines[:max_lines]
    n_cols = max(len(l) for l in lines)
    W = int(2 * margin + (n_cols - 1) * P + s + 2 * r)
    H = int(2 * margin + (len(lines) - 1) * L + 2 * s + 2 * r)
    cells = []
    for li, row in enumerate(lines):
        for ci, (_, dots) in enumerate(row):
            cells.append((dots, margin + r + ci * P, margin + r + li * L))
    return W, H, cells


def _paper(rng, W, H, base, amp_low, amp_high):
    g = np.random.default_rng(rng.randint(0, 10**9))
    small = g.normal(0, 1, (max(2, H // 48), max(2, W // 48))).astype(np.float32)
    low = np.array(Image.fromarray(small).resize((W, H), Image.BICUBIC))
    low = low / (np.abs(low).max() + 1e-6) * amp_low
    high = np.random.default_rng(rng.randint(0, 10**9)).normal(0, amp_high, (H, W))
    tint = np.array([rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-8, 4)])
    img = base + low[..., None] + high[..., None] + tint
    return np.clip(img, 0, 255).astype(np.uint8)


def _boxes(cells, s, r, k=1.35):
    out = []
    for dots, x, y in cells:
        if not dots:
            continue
        out.append((x - k * r, y - k * r, x + s + k * r, y + 2 * s + k * r, dots_to_code(dots)))
    return out


def render_ink(rng, text):
    r = rng.choice([3, 4, 5, 6, 8, 10, 13, 16])
    D = 2 * r
    s = rng.uniform(1.5, 1.9) * D
    P = s * rng.uniform(2.3, 3.1)
    L = (2 * s + D) * rng.uniform(1.35, 1.9)
    margin = int(rng.uniform(1.5, 4) * D)
    W, H, cells = layout(text, s, P, L, rng.randint(10, 32), margin, r)
    base = rng.uniform(215, 252)
    img = Image.fromarray(_paper(rng, W, H, base, rng.uniform(0, 10), rng.uniform(0, 3)))
    draw = ImageDraw.Draw(img)
    ink = rng.choice([(15, 15, 15), (35, 35, 35), (14, 124, 134), (30, 40, 120), (130, 20, 20), (60, 60, 60)])
    empty = rng.random() < 0.4
    for dots, x, y in cells:
        if not dots:
            continue
        for n in range(1, 7):
            cx = x + (0 if n <= 3 else s)
            cy = y + ((n - 1) % 3) * s
            if n in dots:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
            elif empty:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(int(base) - 30,) * 3)
    return img, _boxes(cells, s, r)


def render_relief(rng, text):
    r = rng.choice([4, 5, 6, 7, 9, 11, 14])
    D = 2 * r
    s = rng.uniform(1.55, 1.85) * D
    P = s * rng.uniform(2.35, 2.9)
    L = (2 * s + D) * rng.uniform(1.4, 1.9)
    margin = int(rng.uniform(1.5, 4) * D)
    W, H, cells = layout(text, s, P, L, rng.randint(10, 32), margin, r)
    base = rng.uniform(165, 235)
    arr = _paper(rng, W, H, base, rng.uniform(4, 16), rng.uniform(1, 5)).astype(np.float32)
    shade = np.zeros((H, W), np.float32)
    theta = rng.uniform(0, 2 * np.pi)
    ox, oy = np.cos(theta) * 0.4 * r, np.sin(theta) * 0.4 * r
    strength = rng.uniform(25, 60)
    def bump(cx, cy, sign):
        x0, x1 = int(max(0, cx - 2 * r)), int(min(W, cx + 2 * r + 1))
        y0, y1 = int(max(0, cy - 2 * r)), int(min(H, cy + 2 * r + 1))
        if x0 >= x1 or y0 >= y1:
            return
        dy, dx = np.ogrid[y0:y1, x0:x1]
        dx = dx - cx
        dy = dy - cy
        rr = np.sqrt(dx ** 2 + dy ** 2) / r
        inside = rr < 1.0
        light = -(dx * ox + dy * oy) / (r * 0.4 * r + 1e-6)
        val = np.where(inside, light * (1 - rr ** 2), 0)
        rim = np.exp(-((rr - 1.05) ** 2) / 0.02) * 0.25
        shade[y0:y1, x0:x1] += sign * (val - rim)

    for dots, x, y in cells:
        for n in dots:
            bump(x + (0 if n <= 3 else s), y + ((n - 1) % 3) * s, 1.0)
    if rng.random() < 0.35:
        dx_v, dy_v = rng.uniform(0.3, 0.7) * P, rng.uniform(0.3, 0.6) * s
        for dots, x, y in cells:
            if dots and rng.random() < 0.8:
                for n in rng.sample(range(1, 7), rng.randint(1, 4)):
                    bump(x + dx_v + (0 if n <= 3 else s), y + dy_v + ((n - 1) % 3) * s, -0.6)
    shade = ndi.gaussian_filter(shade, sigma=max(0.6, 0.15 * r))
    arr = arr + strength * shade[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), _boxes(cells, s, r)


def _homography(src, dst):
    A = []
    b = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1).reshape(3, 3)


def _apply(H, pts):
    pts = np.asarray(pts, float)
    hom = np.c_[pts, np.ones(len(pts))] @ H.T
    return hom[:, :2] / hom[:, 2:3]


def augment(rng, img, boxes, strength=1.0):
    W, H = img.size
    pad = int(0.08 * max(W, H))
    arr = np.array(img)
    border = np.median(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]), axis=0).astype(np.uint8)
    canvas = Image.new("RGB", (W + 2 * pad, H + 2 * pad), tuple(int(v) for v in border))
    canvas.paste(img, (pad, pad))
    boxes = [(x1 + pad, y1 + pad, x2 + pad, y2 + pad, c) for x1, y1, x2, y2, c in boxes]
    CW, CH = canvas.size
    src = [(0, 0), (CW, 0), (CW, CH), (0, CH)]
    ang = np.deg2rad(rng.uniform(-7, 7) * strength)
    jit = 0.06 * strength * min(CW, CH)
    cx, cy = CW / 2, CH / 2
    dst = []
    for x, y in src:
        rx = cx + (x - cx) * np.cos(ang) - (y - cy) * np.sin(ang) + rng.uniform(-jit, jit)
        ry = cy + (x - cx) * np.sin(ang) + (y - cy) * np.cos(ang) + rng.uniform(-jit, jit)
        dst.append((rx, ry))
    Hm = _homography(src, dst)
    inv = np.linalg.inv(Hm)
    coeffs = (inv / inv[2, 2]).flatten()[:8]
    out = canvas.transform((CW, CH), Image.PERSPECTIVE, tuple(coeffs), Image.BILINEAR, fillcolor=tuple(int(v) for v in border))
    new_boxes = []
    for x1, y1, x2, y2, c in boxes:
        pts = _apply(Hm, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        bx1, by1 = pts.min(0)
        bx2, by2 = pts.max(0)
        area = (bx2 - bx1) * (by2 - by1)
        cx1, cy1, cx2, cy2 = max(0, bx1), max(0, by1), min(CW, bx2), min(CH, by2)
        if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.6 * area:
            continue
        new_boxes.append((cx1, cy1, cx2, cy2, c))
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.3)))
    a = np.array(out).astype(np.float32)
    a = a * rng.uniform(0.75, 1.2) + rng.uniform(-25, 25)
    if rng.random() < 0.5:
        a += np.random.default_rng(rng.randint(0, 10**9)).normal(0, rng.uniform(2, 9), a.shape)
    if rng.random() < 0.4:
        gy = np.linspace(rng.uniform(-30, 0), rng.uniform(0, 30), a.shape[0])[:, None, None]
        a += gy
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=rng.randint(45, 95))
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out, new_boxes


def make_sample(rng, style):
    text = random_text(rng)
    if style == "relief":
        img, boxes = render_relief(rng, text)
    else:
        img, boxes = render_ink(rng, text)
    if rng.random() < 0.85:
        img, boxes = augment(rng, img, boxes, strength=rng.uniform(0.3, 1.0))
    return img, boxes
