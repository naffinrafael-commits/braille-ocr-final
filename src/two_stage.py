import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(n_classes, pretrained=False):
    import torch
    from torchvision.models import mobilenet_v2
    weights = None
    if pretrained:
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    try:
        model = mobilenet_v2(weights=weights)
    except Exception:
        model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, n_classes)
    return model


def load_classifier(path, device="cpu"):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ck["codes"]))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, list(ck["codes"]), tuple(ck.get("size", (96, 64)))


def crop_boxes(img, boxes, pad=0.08):
    out = []
    W, H = img.size
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        w, h = x2 - x1, y2 - y1
        out.append(img.crop((int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)), int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h)))))
    return out


def classify(model, codes, size, crops, device="cpu", batch=256):
    import torch
    if not crops:
        return []
    h, w = size
    results = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            arr = np.stack([(np.asarray(c.convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255.0 - MEAN) / STD for c in crops[i:i + batch]])
            x = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
            prob = torch.softmax(model(x), dim=1)
            conf, idx = prob.max(1)
            results += [(codes[int(k)], float(p)) for k, p in zip(idx.cpu(), conf.cpu())]
    return results
