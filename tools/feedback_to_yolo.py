import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml
from PIL import Image

from src.braille_tables import CLASS_NAMES, text_to_cells, dots_to_code
from src.feedback import folder
from src.postprocess import reading_order


def build(out_zip=None):
    src = folder()
    out_zip = out_zip or os.path.join(ROOT, "feedback_dataset.zip")
    ok, skipped = [], []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(src)):
            d = os.path.join(src, name)
            if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "correcao.txt")):
                continue
            pred = json.load(open(os.path.join(d, "predicao.json"), encoding="utf-8"))
            text = open(os.path.join(d, "correcao.txt"), encoding="utf-8").read()
            boxes = [b for row in reading_order(pred["deteccoes"]) for b in row]
            cells = [dots for _, dots in text_to_cells(text.replace("\n", " ")) if dots]
            if not boxes or len(boxes) != len(cells):
                skipped.append((name, len(boxes), len(cells)))
                continue
            img = Image.open(os.path.join(d, "imagem.jpg"))
            W, H = img.size
            lines = [f"{dots_to_code(c) - 1} {b['x'] / W:.6f} {b['y'] / H:.6f} {b['width'] / W:.6f} {b['height'] / H:.6f}" for b, c in zip(boxes, cells)]
            z.write(os.path.join(d, "imagem.jpg"), f"feedback/train/images/{name}.jpg")
            z.writestr(f"feedback/train/labels/{name}.txt", "\n".join(lines))
            ok.append(name)
        z.writestr("feedback/data.yaml", yaml.safe_dump({"names": CLASS_NAMES, "nc": 63}, allow_unicode=True))
        z.writestr("feedback/feedback_manifest.json", json.dumps({"aproveitadas": ok, "ignoradas": skipped}, ensure_ascii=False, indent=2))
    return out_zip, ok, skipped


if __name__ == "__main__":
    path, ok, skipped = build()
    print(f"{len(ok)} correções convertidas, {len(skipped)} ignoradas (número de células diferente do texto). Arquivo: {path}")
