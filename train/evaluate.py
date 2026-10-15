import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

from src.braille_tables import name_to_dots, dots_to_code
from src.postprocess import reading_order
from src.text_metrics import levenshtein

DOMAINS = ["roboflow", "angelina", "sintetico", "mosaico"]
IOU_NMS = 0.45


def hmean(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def yolo_domain_metrics(weights, data_dir, imgsz=640, single_cls=False):
    from ultralytics import YOLO
    model = YOLO(weights)
    out = {}
    for d in DOMAINS:
        y = os.path.join(data_dir, f"data_test_{d}.yaml")
        folder = os.path.join(data_dir, "images", f"test_{d}")
        if not os.path.exists(y) or not os.path.isdir(folder) or not os.listdir(folder):
            continue
        m = model.val(data=y, split="val", imgsz=imgsz, batch=16, iou=IOU_NMS, max_det=1500, plots=False, verbose=False, single_cls=single_cls)
        p, r = float(m.box.mp), float(m.box.mr)
        out[d] = {"precisao": round(p, 4), "revocacao": round(r, 4), "hmean": round(hmean(p, r), 4),
                  "map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4)}
    return out


def load_gt(label_path, W, H):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path).read().split("\n"):
            p = line.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(p[0]), *[float(v) for v in p[1:]]
                boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, c + 1, 1.0))
    return boxes


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.5):
    used = set()
    tp_cls = tp_box = 0
    for p in sorted(preds, key=lambda t: -t[5]):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= thr:
            used.add(bj)
            tp_box += 1
            tp_cls += int(p[4] == gts[bj][4])
    return tp_cls, tp_box


def sequence(boxes):
    dets = [{"x": (b[0] + b[2]) / 2, "y": (b[1] + b[3]) / 2, "width": b[2] - b[0], "height": b[3] - b[1], "code": b[4]} for b in boxes]
    return [d["code"] for row in reading_order(dets) for d in row]


def predict_yolo(model, img, imgsz, conf, device):
    r = model(np.asarray(img)[..., ::-1].copy(), imgsz=imgsz, conf=conf, iou=IOU_NMS, max_det=1500, verbose=False, device=device)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dots = name_to_dots(r.names[int(b.cls[0])])
        out.append((x1, y1, x2, y2, dots_to_code(dots) if dots else 0, float(b.conf[0])))
    return out


def predict_two_stage(det, clf, img, imgsz, conf, device):
    from src.two_stage import classify, crop_boxes
    model, codes, size = clf
    r = det(np.asarray(img)[..., ::-1].copy(), imgsz=imgsz, conf=conf, iou=IOU_NMS, max_det=1500, verbose=False, device=device)[0]
    boxes = [tuple(b.xyxy[0].tolist()) + (float(b.conf[0]),) for b in r.boxes]
    cls = classify(model, codes, size, crop_boxes(img, boxes), device="cuda" if device == 0 else "cpu")
    return [(b[0], b[1], b[2], b[3], c, b[4] * p) for b, (c, p) in zip(boxes, cls)]


def predict_grid(img):
    from src.braille_grid import decode
    out = []
    for d in decode(np.asarray(img))["detections"]:
        if d.get("dots"):
            out.append((d["x"] - d["width"] / 2, d["y"] - d["height"] / 2, d["x"] + d["width"] / 2, d["y"] + d["height"] / 2,
                        dots_to_code(d["dots"]), float(d["confidence"])))
    return out


def end_to_end(data_dir, fn, max_images=150):
    out = {}
    for d in DOMAINS:
        imgs = sorted(glob.glob(os.path.join(data_dir, "images", f"test_{d}", "*.jpg")))[:max_images]
        if not imgs:
            continue
        tp_c = tp_b = n_pred = n_gt = 0
        edits = ref_len = 0
        t = 0.0
        for ip in imgs:
            img = Image.open(ip).convert("RGB")
            gts = load_gt(ip.replace("/images/", "/labels/")[:-4] + ".txt", *img.size)
            t0 = time.time()
            preds = fn(img)
            t += time.time() - t0
            a, b = match(preds, gts)
            tp_c += a
            tp_b += b
            n_pred += len(preds)
            n_gt += len(gts)
            if gts:
                edits += levenshtein(sequence(gts), sequence(preds))
                ref_len += len(gts)
        p, r = tp_c / max(1, n_pred), tp_c / max(1, n_gt)
        out[d] = {"precisao_celula": round(p, 4), "revocacao_celula": round(r, 4), "hmean_celula": round(hmean(p, r), 4),
                  "deteccao_revocacao": round(tp_b / max(1, n_gt), 4), "cer": round(edits / max(1, ref_len), 4),
                  "fps": round(len(imgs) / max(t, 1e-6), 2), "imagens": len(imgs)}
    return out


def fps_cpu(fn, data_dir, n=10):
    imgs = sorted(glob.glob(os.path.join(data_dir, "images", "test_roboflow", "*.jpg")))[:n]
    imgs = imgs or sorted(glob.glob(os.path.join(data_dir, "images", "val", "*.jpg")))[:n]
    if not imgs:
        return None
    ims = [Image.open(p).convert("RGB") for p in imgs]
    fn(ims[0])
    t0 = time.time()
    for im in ims:
        fn(im)
    return round(len(ims) / (time.time() - t0), 2)


def table(rows, cols, title):
    lines = [f"### {title}", "", "| Domínio | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for d, m in rows.items():
        lines.append(f"| {d} | " + " | ".join(str(m.get(k, "")) for _, k in cols) + " |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--yolo26")
    ap.add_argument("--single416")
    ap.add_argument("--cell")
    ap.add_argument("--cls")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    res = {}
    singles = [("yolo11n_640", a.single, a.imgsz), ("yolo26n_640", a.yolo26, a.imgsz), ("yolo11n_416", a.single416, 416)]
    for name, w, sz in singles:
        if not w or not os.path.exists(w):
            continue
        print("Avaliando", name, flush=True)
        model, model_cpu = YOLO(w), YOLO(w)
        res[name] = {"yolo": yolo_domain_metrics(w, a.data, sz),
                     "ponta_a_ponta": end_to_end(a.data, lambda im, m=model, s=sz: predict_yolo(m, im, s, a.conf, device)),
                     "fps_cpu": fps_cpu(lambda im, m=model_cpu, s=sz: predict_yolo(m, im, s, a.conf, "cpu"), a.data)}
    if a.cell and a.cls and os.path.exists(a.cell) and os.path.exists(a.cls):
        print("Avaliando 2 etapas", flush=True)
        from src.two_stage import load_classifier
        det, det_cpu = YOLO(a.cell), YOLO(a.cell)
        clf = load_classifier(a.cls, device="cuda" if device == 0 else "cpu")
        clf_cpu = load_classifier(a.cls, device="cpu")
        res["duas_etapas"] = {"deteccao_celulas": yolo_domain_metrics(a.cell, a.data, a.imgsz, single_cls=True),
                              "ponta_a_ponta": end_to_end(a.data, lambda im: predict_two_stage(det, clf, im, a.imgsz, a.conf, device)),
                              "fps_cpu": fps_cpu(lambda im: predict_two_stage(det_cpu, clf_cpu, im, a.imgsz, a.conf, "cpu"), a.data)}
    print("Avaliando leitura por grade", flush=True)
    res["grade_classica"] = {"ponta_a_ponta": end_to_end(a.data, predict_grid), "fps_cpu": fps_cpu(predict_grid, a.data)}
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    ycols = [("Precisão", "precisao"), ("Revocação", "revocacao"), ("Hmean", "hmean"), ("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map50_95")]
    ecols = [("Precisão", "precisao_celula"), ("Revocação", "revocacao_celula"), ("Hmean", "hmean_celula"),
             ("Detecção sem classe", "deteccao_revocacao"), ("CER", "cer"), ("FPS GPU", "fps")]
    titles = {"yolo11n_640": "YOLO11n, 640 px", "yolo26n_640": "YOLO26n, 640 px", "yolo11n_416": "YOLO11n, 416 px",
              "duas_etapas": "2 etapas (detector de células + MobileNetV2)", "grade_classica": "Leitura por grade (método clássico)"}
    lines = ["# Resultados", "",
             "Métricas ponta a ponta: uma célula conta como acerto quando a caixa coincide com a anotada (IoU de pelo menos 0,5) "
             "e o padrão de pontos está correto. CER: erros de caracteres na sequência lida, em relação à anotada (0 é perfeito).", ""]
    for k, v in res.items():
        if "yolo" in v:
            lines += table(v["yolo"], ycols, titles[k] + ": métricas do YOLO")
        lines += table(v["ponta_a_ponta"], ecols, titles[k] + ": leitura ponta a ponta")
    lines += ["### Velocidade em CPU (imagens por segundo)", ""] + [f"- {titles[k]}: {v.get('fps_cpu')}" for k, v in res.items()]
    with open(os.path.join(a.out, "RELATORIO_RESULTADOS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
