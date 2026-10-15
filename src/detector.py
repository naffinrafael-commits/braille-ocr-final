import base64
import io
import os
from functools import lru_cache

import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from PIL import Image

from src.braille_grid import decode as grid_decode
from src.braille_tables import name_to_dots, decode_cells, dots_to_unicode, code_to_dots

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODES = {"yolo11": "YOLO11n (1 etapa)", "yolo26": "YOLO26n (1 etapa)"}


def cfg():
    with open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


def to_pil(image):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, image)
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr[..., :3].astype(np.uint8))


def weight_path(key):
    rel = cfg()["model"].get(key)
    if not rel:
        return None
    override = os.getenv("BRAILLE_WEIGHTS_DIR")
    if override:
        p = os.path.join(override, os.path.basename(rel))
        return p if os.path.exists(p) else None
    p = os.path.join(ROOT, rel)
    return p if os.path.exists(p) else None


@lru_cache(maxsize=4)
def _yolo(path):
    from ultralytics import YOLO
    return YOLO(path)


@lru_cache(maxsize=2)
def _classifier(path):
    from src.two_stage import load_classifier
    return load_classifier(path, device="cpu")


def status():
    out = {}
    items = (("yolo11", "YOLO11n"), ("yolo26", "YOLO26n"), ("cell_detector", "Detector de células"), ("cell_classifier", "Classificador MobileNetV2"))
    for key, label_ in items:
        p = weight_path(key)
        if p is None:
            out[label_] = (False, "arquivo não encontrado")
        else:
            out[label_] = (True, f"{os.path.relpath(p, ROOT)} ({os.path.getsize(p) / 1e6:.1f} MB)")
    return out


def available(mode):
    if mode in YOLO_MODES:
        return weight_path(mode) is not None
    if mode == "duas_etapas":
        return weight_path("cell_detector") is not None and weight_path("cell_classifier") is not None
    return True


def _imgsz(img):
    c = cfg()["model"]
    return c["img_size_large"] if max(img.size) > c["large_above"] else c["img_size"]


def label(dots):
    if not dots:
        return "?"
    ch = decode_cells([dots])
    return ch if ch else dots_to_unicode(dots)


def _predict(model, img):
    c = cfg()["model"]
    return model(np.asarray(img)[..., ::-1].copy(), conf=c["confidence"], iou=c["iou"], imgsz=_imgsz(img), max_det=c["max_det"], verbose=False)[0]


def _det(x1, y1, x2, y2, conf, dots, source):
    return {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "width": x2 - x1, "height": y2 - y1,
            "confidence": float(conf), "dots": dots, "class": label(dots), "source": source}


def _result(dets, method):
    q = float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
    return {"text": None, "detections": dets, "quality": round(q, 3), "method": method}


def detect_yolo(image, which="yolo11"):
    img = to_pil(image)
    path = weight_path(which)
    if path is None:
        raise FileNotFoundError(f"pesos de {which} não encontrados em train/weights/")
    r = _predict(_yolo(path), img)
    dets = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append(_det(x1, y1, x2, y2, b.conf[0], name_to_dots(r.names[int(b.cls[0])]), which))
    return _result(dets, which)


def detect_two_stage(image):
    from src.two_stage import classify, crop_boxes
    img = to_pil(image)
    dp, cp = weight_path("cell_detector"), weight_path("cell_classifier")
    if dp is None or cp is None:
        raise FileNotFoundError("cell_detector.pt ou cell_classifier.pt não encontrado em train/weights/")
    r = _predict(_yolo(dp), img)
    boxes = [tuple(b.xyxy[0].tolist()) + (float(b.conf[0]),) for b in r.boxes]
    model, codes, size = _classifier(cp)
    preds = classify(model, codes, size, crop_boxes(img, boxes))
    dets = [_det(x1, y1, x2, y2, cd * p, code_to_dots(code), "duas_etapas") for (x1, y1, x2, y2, cd), (code, p) in zip(boxes, preds)]
    return _result(dets, "duas_etapas")


def detect_roboflow(image):
    c = cfg()
    key = os.getenv("ROBOFLOW_API_KEY", "")
    if not key:
        raise ValueError("ROBOFLOW_API_KEY não definida no arquivo .env")
    buf = io.BytesIO()
    to_pil(image).save(buf, format="JPEG")
    url = f"{c['roboflow']['api_url']}/{c['roboflow']['model_id']}?api_key={key}&name=image.jpg"
    resp = requests.post(url, data=base64.b64encode(buf.getvalue()).decode(), headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    resp.raise_for_status()
    dets = [_det(p["x"] - p["width"] / 2, p["y"] - p["height"] / 2, p["x"] + p["width"] / 2, p["y"] + p["height"] / 2,
                 p["confidence"], name_to_dots(p["class"]), "roboflow") for p in resp.json().get("predictions", [])]
    return _result(dets, "roboflow")


def detect_grid(image):
    res = grid_decode(np.asarray(to_pil(image)))
    for d in res["detections"]:
        d.setdefault("dots", None)
    res["method"] = "grade"
    return res


def detect(image, mode="auto"):
    if mode == "grade":
        return detect_grid(image)
    if mode in YOLO_MODES:
        return detect_yolo(image, mode)
    if mode == "duas_etapas":
        return detect_two_stage(image)
    if mode == "roboflow":
        return detect_roboflow(image)
    notes = []
    grid = None
    try:
        grid = detect_grid(image)
    except Exception as e:
        notes.append(f"leitura por grade falhou: {e}")
    c = cfg()["auto"]
    if grid and grid.get("n_cells", 0) > 0 and grid.get("quality", 0) >= c["grid_min_quality"]:
        grid["notes"] = notes
        return grid
    for m in c.get("photo_order", ["yolo11", "yolo26", "duas_etapas"]):
        if not available(m):
            continue
        try:
            res = detect(image, m)
            if res["detections"]:
                res["notes"] = notes + [f"imagem tratada como foto: usado {m}"]
                return res
        except Exception as e:
            notes.append(f"{m} falhou: {e}")
    grid = grid or {"text": "", "detections": [], "quality": 0.0, "method": "grade", "n_cells": 0}
    grid["notes"] = notes
    return grid
