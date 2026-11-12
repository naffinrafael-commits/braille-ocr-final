import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

W = os.path.join(ROOT, "train", "weights")
rows = []
for name, f, size in (("YOLO11n 63 classes", "best.pt", 640), ("YOLO26n 63 classes", "yolo26.pt", 640), ("Detector de celas (classe única)", "cell_detector.pt", 640)):
    p = os.path.join(W, f)
    if not os.path.exists(p):
        continue
    m = YOLO(p)
    m.fuse()
    net = m.model
    params = sum(t.numel() for t in net.parameters())
    flops = get_flops(net, size)
    rows.append((name, f"{params:,}".replace(",", "."), f"{flops:.1f}".replace(".", ","), f"{size} px"))
p = os.path.join(W, "cell_classifier.pt")
if os.path.exists(p):
    from src.two_stage import load_classifier
    from torch.utils.flop_counter import FlopCounterMode
    model, codes, size = load_classifier(p)
    params = sum(t.numel() for t in model.parameters())
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 3, size[0], size[1]))
    rows.append(("MobileNetV2 (classificador)", f"{params:,}".replace(",", "."), f"{fc.get_total_flops() / 1e9:.2f}".replace(".", ","), f"{size[0]} × {size[1]} px"))
lines = ["| Modelo | Parâmetros | GFLOPs | Entrada |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
text = "\n".join(lines)
print(text)
with open(os.path.join(W, "MODEL_INFO.md"), "w", encoding="utf-8") as f:
    f.write(text + "\n")
