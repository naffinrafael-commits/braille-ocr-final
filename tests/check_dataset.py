import argparse
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw

from src.braille_tables import CLASS_NAMES, code_to_dots, decode_cells


def label_text(cls):
    dots = code_to_dots(cls + 1)
    ch = decode_cells([dots])
    return ch if ch and ord(ch[0]) < 0x2800 else CLASS_NAMES[cls]


def validate(root):
    problems = Counter()
    per_split = Counter()
    for split in sorted(os.listdir(os.path.join(root, "labels"))):
        ld = os.path.join(root, "labels", split)
        idir = os.path.join(root, "images", split)
        for f in os.listdir(ld):
            per_split[split] += 1
            if not os.path.exists(os.path.join(idir, f[:-4] + ".jpg")):
                problems["imagem_faltando"] += 1
            for line in open(os.path.join(ld, f)).read().split("\n"):
                if not line.strip():
                    continue
                p = line.split()
                if len(p) != 5:
                    problems["linha_invalida"] += 1
                    continue
                c = int(p[0])
                vals = [float(v) for v in p[1:]]
                if not 0 <= c < 63:
                    problems["classe_fora"] += 1
                if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    problems["coordenada_fora"] += 1
    return per_split, problems


def montage(root, out, per_domain=4, seed=1):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for split in os.listdir(os.path.join(root, "images")):
        for f in os.listdir(os.path.join(root, "images", split)):
            groups[f.split("_")[0]].append((split, f))
    tiles = []
    for dom in sorted(groups):
        for split, f in rng.sample(groups[dom], min(per_domain, len(groups[dom]))):
            img = Image.open(os.path.join(root, "images", split, f)).convert("RGB")
            W, H = img.size
            d = ImageDraw.Draw(img)
            lw = max(1, int(min(W, H) / 350))
            for line in open(os.path.join(root, "labels", split, f[:-4] + ".txt")).read().split("\n"):
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = line.split()
                c = int(c)
                cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
                d.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=(230, 30, 30), width=lw)
                d.text((cx - bw / 2 + 1, cy - bh / 2 - 11), label_text(c), fill=(0, 90, 220))
            img.thumbnail((420, 420))
            tiles.append((dom, img))
    cols = per_domain
    rows = (len(tiles) + cols - 1) // cols
    can = Image.new("RGB", (cols * 430, rows * 445), "white")
    dc = ImageDraw.Draw(can)
    for i, (dom, t) in enumerate(tiles):
        x, y = (i % cols) * 430, (i // cols) * 445
        can.paste(t, (x, y + 15))
        dc.text((x + 2, y + 1), dom, fill=(0, 0, 0))
    can.save(out, quality=90)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="conferencia_dataset.jpg")
    a = ap.parse_args()
    per_split, problems = validate(a.data)
    print("Imagens por divisão:", dict(per_split))
    print("Problemas encontrados:", dict(problems) if problems else "nenhum")
    print("Mosaico de conferência:", montage(a.data, a.out))
