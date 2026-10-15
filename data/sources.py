import hashlib
import io
import re
import zipfile

import numpy as np
import yaml
from PIL import Image

from src.braille_tables import name_to_dots, letter_to_dots, dots_to_code, code_to_dots

_ZIPS = {}


def zf(path):
    z = _ZIPS.get(path)
    if z is None:
        z = zipfile.ZipFile(path)
        _ZIPS[path] = z
    return z


MULTI_CONS = "bcdfghjklmnpqrstvwxyz"
MULTI_VOW = "aiueo"
AUG_SPLIT = re.compile(r"_(bright|exp|hue|sat|contrast|blur|noise|rot|flip|gamma)", re.I)


def split_of(group, val=10, test=10):
    h = int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % 100
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def open_image(data):
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def detect_kinds(names):
    kinds = set()
    if any(n.endswith("data.yaml") for n in names) and any("/labels/" in n for n in names):
        kinds.add("yolo")
    if any("AngelinaDataset" in n for n in names):
        kinds.add("angelina")
    if any(re.search(r"Braille Dataset/[a-z]\d", n) for n in names):
        kinds.add("kaggle")
    if any("Image Classification" in n for n in names):
        kinds.add("classification")
    if any("/Grid/" in n for n in names):
        kinds.add("multi")
    if any(n.endswith("feedback_manifest.json") for n in names):
        kinds.add("feedback")
    return kinds


def read_yolo(zpath, domain="roboflow", force_split=None):
    z = zf(zpath)
    names = z.namelist()
    yamls = [n for n in names if n.endswith("data.yaml")]
    seen = set()
    for ypath in yamls:
        prefix = ypath[: -len("data.yaml")]
        cfg = yaml.safe_load(z.read(ypath))
        cls_names = cfg.get("names", [])
        if isinstance(cls_names, dict):
            cls_names = [cls_names[k] for k in sorted(cls_names)]
        mapping = {i: name_to_dots(n) for i, n in enumerate(cls_names)}
        for member in names:
            if member.startswith(prefix) and "/images/" in member and member.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                data = z.read(member)
                digest = hashlib.md5(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                stem = member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                lbl = member.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                rows = z.read(lbl).decode("utf-8", "ignore").split("\n") if lbl in names else []
                group = domain + ":" + stem.split(".rf.")[0].replace("_jpg", "").replace("_png", "")
                yield {"zip": zpath, "member": member, "rows": rows, "map": mapping, "group": group,
                       "split": force_split or split_of(group), "domain": domain, "kind": "yolo"}


def load_yolo_sample(s):
    z = zf(s["zip"])
    img = open_image(z.read(s["member"]))
    W, H = img.size
    boxes = []
    for row in s["rows"]:
        p = row.split()
        if len(p) < 5:
            continue
        dots = s["map"].get(int(float(p[0])))
        if not dots:
            continue
        cx, cy, w, h = [float(v) for v in p[1:5]]
        boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, dots_to_code(dots)))
    return img, boxes


def read_angelina(zpath):
    z = zf(zpath)
    names = set(z.namelist())
    roots = [n[: -len("books/train.txt")] for n in names if n.endswith("AngelinaDataset-master/books/train.txt")]
    if not roots:
        return
    root = roots[0]
    plan = [("books/train.txt", "train"), ("books/val.txt", "val"), ("handwritten/train.txt", "train"),
            ("handwritten/val.txt", "val"), ("uploaded/test.txt", "test"), ("not_braille/train.txt", "train")]
    for lst, split in plan:
        path = root + lst
        if path not in names:
            continue
        coll = lst.split("/")[0]
        for line in z.read(path).decode("utf-8", "ignore").split():
            rel = line.strip().replace("\\", "/")
            member = root + coll + "/" + rel
            if member not in names:
                continue
            csv = member.rsplit(".", 1)[0] + ".csv"
            yield {"zip": zpath, "member": member, "csv": csv if csv in names else None, "split": split,
                   "domain": "angelina" if coll != "not_braille" else "negativo", "group": "ang:" + rel, "kind": "angelina"}


def load_angelina_sample(s):
    z = zf(s["zip"])
    img = open_image(z.read(s["member"]))
    W, H = img.size
    boxes = []
    if s["csv"] and s["domain"] == "angelina":
        for row in z.read(s["csv"]).decode("utf-8", "ignore").split("\n"):
            p = row.strip().split(";")
            if len(p) < 5:
                continue
            try:
                l, t, r, b = [float(v) for v in p[:4]]
                code = int(float(p[4]))
            except ValueError:
                continue
            if code <= 0 or code > 63:
                continue
            boxes.append((l * W, t * H, r * W, b * H, code))
    return img, boxes


def _grid_squares(arr):
    r, g, b = [arr[..., k].astype(int) for k in range(3)]
    line = ((r > 170) & (g > 170) & (b < 140)) | ((r > 90) & (b > 90) & (g < 80))

    def segs(prof):
        free = prof < 0.25
        out = []
        start = None
        for i, f in enumerate(free):
            if f and start is None:
                start = i
            if not f and start is not None:
                if i - start > 15:
                    out.append((start, i))
                start = None
        if start is not None and len(free) - start > 15:
            out.append((start, len(free)))
        return out

    rs, cs = segs(line.mean(1)), segs(line.mean(0))
    if len(rs) != 3 or len(cs) != 4:
        return None
    w = sorted(b - a for a, b in cs)[1]
    cs = [(max(a, b - w), b) if b - a > 1.2 * w else (a, b) for a, b in cs]
    return rs, cs


def _clean_lines(crop):
    a = crop.astype(np.float32)
    mx = a.max(-1)
    mn = a.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    mask = sat > 0.18
    if mask.mean() > 0.6:
        return crop
    fill = np.median(a[~mask], axis=0) if (~mask).any() else a.reshape(-1, 3).mean(0)
    a[mask] = fill
    return np.clip(a, 0, 255).astype(np.uint8)


def read_multi(zpath):
    z = zf(zpath)
    for member in z.namelist():
        if "/Grid/" not in member or member.endswith("/"):
            continue
        folder = member.split("/Grid/")[1].split("/")[0]
        if not folder.isdigit():
            continue
        idx = int(folder)
        if idx >= len(MULTI_CONS) * len(MULTI_VOW):
            continue
        left, right = MULTI_CONS[idx // 5], MULTI_VOW[idx % 5]
        stem = member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        group = "multi:" + folder + ":" + AUG_SPLIT.split(stem)[0].strip()
        yield {"zip": zpath, "member": member, "labels": (letter_to_dots(left), letter_to_dots(right)),
               "group": group, "split": split_of(group), "domain": "multi", "kind": "multi"}


def load_multi_crops(s):
    z = zf(s["zip"])
    arr = np.array(open_image(z.read(s["member"])))
    H, W = arr.shape[:2]
    sq = _grid_squares(arr)
    precise = sq is not None
    if precise:
        rs, cs = sq
        rects = [(cs[0][0], rs[0][0], cs[1][1], rs[2][1]), (cs[2][0], rs[0][0], cs[3][1], rs[2][1])]
    else:
        rects = split_pair(arr)
    out = []
    for k, ((x1, y1, x2, y2), dots) in enumerate(zip(rects, s["labels"])):
        if not precise and k == 1 and not two_columns(dots):
            continue
        crop = _clean_lines(arr[y1:y2, x1:x2])
        if crop.size and dots:
            out.append((Image.fromarray(crop), dots_to_code(dots)))
    return out


def read_kaggle(zpath):
    z = zf(zpath)
    for member in z.namelist():
        m = re.search(r"Braille Dataset/([a-z])(\d+\.JPG\d+)", member)
        if not m or not member.lower().endswith((".jpg", ".png")):
            continue
        group = "kag:" + m.group(1) + m.group(2)
        yield {"zip": zpath, "member": member, "code": dots_to_code(letter_to_dots(m.group(1))), "group": group,
               "split": split_of(group), "domain": "kaggle", "kind": "crop"}


def read_classification(zpath):
    z = zf(zpath)
    for member in z.namelist():
        if "Image Classification" not in member or not member.lower().endswith((".png", ".jpg")):
            continue
        letter = member.rsplit("/", 2)[-2]
        dots = letter_to_dots(letter)
        if not dots or len(letter) != 1:
            continue
        group = "cls:" + member.rsplit("/", 1)[-1]
        yield {"zip": zpath, "member": member, "code": dots_to_code(dots), "group": group,
               "split": split_of(group), "domain": "classificacao", "kind": "crop"}


def load_crop(s):
    z = zf(s["zip"])
    return open_image(z.read(s["member"])), s["code"]


def read_feedback(zpath):
    yield from read_yolo(zpath, domain="feedback", force_split="train")


def _energy_cols(arr):
    from scipy import ndimage as ndi
    g = ndi.gaussian_filter(np.asarray(arr, np.float32).mean(-1), 1.0)
    e = np.abs(ndi.sobel(g, 0)) + np.abs(ndi.sobel(g, 1))
    return ndi.uniform_filter1d(e.mean(0), size=max(3, arr.shape[1] // 25))


def _trim(arr, x0, x1, prof):
    seg = prof[x0:x1]
    if seg.size == 0:
        return x0, x1
    thr = seg.min() + 0.35 * (seg.max() - seg.min())
    idx = np.nonzero(seg > thr)[0]
    if idx.size == 0:
        return x0, x1
    pad = max(2, (x1 - x0) // 12)
    return max(x0, x0 + idx[0] - pad), min(x1, x0 + idx[-1] + 1 + pad)


def split_pair(arr):
    H, W = arr.shape[:2]
    prof = _energy_cols(arr)
    a, b = int(W * 0.35), int(W * 0.65)
    if b <= a:
        return [(0, 0, W // 2, H), (W // 2, 0, W, H)]
    seg = prof[a:b]
    low = np.nonzero(seg <= seg.min() + 0.25 * (seg.max() - seg.min()))[0]
    cut = a + int((low[0] + low[-1]) // 2) if low.size else a + int(np.argmin(seg))
    return [(0, 0, cut, H), (cut, 0, W, H)]


def two_columns(dots):
    return bool(set(dots) & {1, 2, 3}) and bool(set(dots) & {4, 5, 6})


def read_multi_cropped(zpath):
    z = zf(zpath)
    for member in z.namelist():
        if "/cropped_images/" not in member or member.endswith("/"):
            continue
        name = member.rsplit("/", 1)[-1]
        syl = name.split("_")[0].lower()
        if len(syl) != 2 or syl[0] not in MULTI_CONS or syl[1] not in MULTI_VOW:
            continue
        group = "mcrop:" + name.split("_jpg.rf.")[0]
        yield {"zip": zpath, "member": member, "labels": (letter_to_dots(syl[0]), letter_to_dots(syl[1])),
               "group": group, "split": split_of(group), "domain": "multi_recorte", "kind": "multi_cropped"}


def load_multi_cropped(s):
    z = zf(s["zip"])
    img = open_image(z.read(s["member"]))
    halves = split_pair(np.array(img))
    return [(img.crop(r), dots_to_code(d)) for k, (r, d) in enumerate(zip(halves, s["labels"])) if d and (k == 0 or two_columns(d))]
