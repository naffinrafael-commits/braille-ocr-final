import json
import os
import time

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def folder():
    import yaml
    with open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
        rel = yaml.safe_load(f).get("feedback", {}).get("folder", "data/feedback")
    path = os.path.join(ROOT, rel)
    os.makedirs(path, exist_ok=True)
    return path


def save(original_image, result, corrected_text):
    d = os.path.join(folder(), time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    Image.fromarray(original_image).convert("RGB").save(os.path.join(d, "imagem.jpg"), quality=95)
    dets = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in det.items() if k in ("x", "y", "width", "height", "confidence", "dots", "class")}
            for det in result.get("detections", [])]
    with open(os.path.join(d, "predicao.json"), "w", encoding="utf-8") as f:
        json.dump({"metodo": result.get("method"), "texto_lido": result.get("text"), "deteccoes": dets}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(d, "correcao.txt"), "w", encoding="utf-8") as f:
        f.write(corrected_text)
    return d


def count():
    p = folder()
    return sum(1 for x in os.listdir(p) if os.path.isdir(os.path.join(p, x)))
