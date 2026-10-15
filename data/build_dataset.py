import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

from data import sources as S
from data.synth import make_sample, augment, _paper
from src.braille_tables import CLASS_NAMES, code_to_dots

TEST_DOMAINS = ["roboflow", "angelina", "sintetico", "mosaico"]


class Writer:
    def __init__(self, out):
        self.out = out
        self.count = Counter()
        self.cells = Counter()
        self.class_counts = Counter()
        self.n = 0
        self.skipped = 0
        for d in ["train", "val"] + [f"test_{t}" for t in TEST_DOMAINS]:
            os.makedirs(os.path.join(out, "images", d), exist_ok=True)
            os.makedirs(os.path.join(out, "labels", d), exist_ok=True)

    def add(self, img, boxes, split_dir, domain):
        W, H = img.size
        lines = []
        for x1, y1, x2, y2, code in boxes:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2 or not 1 <= code <= 63:
                continue
            lines.append(f"{code - 1} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            self.class_counts[code] += 1
        self.n += 1
        name = f"{domain}_{self.n:07d}"
        img.save(os.path.join(self.out, "images", split_dir, name + ".jpg"), quality=92)
        with open(os.path.join(self.out, "labels", split_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        self.count[(domain, split_dir)] += 1
        self.cells[(domain, split_dir)] += len(lines)


def split_dir(split, domain):
    if split == "test":
        return "test_" + (domain if domain in TEST_DOMAINS else "roboflow")
    return split


def limit_size(img, boxes, max_side):
    W, H = img.size
    f = min(1.0, max_side / max(W, H))
    if f < 1.0:
        img = img.resize((int(W * f), int(H * f)), Image.BILINEAR)
        boxes = [(x1 * f, y1 * f, x2 * f, y2 * f, c) for x1, y1, x2, y2, c in boxes]
    return img, boxes


def tiles(img, boxes, tile_above):
    W, H = img.size
    if max(W, H) <= tile_above:
        return [(img, boxes)]
    out = []
    tw, th = int(W * 0.58), int(H * 0.58)
    for ox in (0, W - tw):
        for oy in (0, H - th):
            sub = []
            for x1, y1, x2, y2, c in boxes:
                ix1, iy1, ix2, iy2 = max(x1, ox), max(y1, oy), min(x2, ox + tw), min(y2, oy + th)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) >= 0.7 * (x2 - x1) * (y2 - y1):
                    sub.append((ix1 - ox, iy1 - oy, ix2 - ox, iy2 - oy, c))
            out.append((img.crop((ox, oy, ox + tw, oy + th)), sub))
    return out


def _synth_job(args):
    seed, style = args
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 32))
    return make_sample(rng, style)


def mosaic_page(rng, pool):
    h = rng.randint(30, 110)
    rows = rng.randint(2, 8)
    per_row = rng.randint(4, 14)
    items = []
    for _ in range(rows):
        row = []
        while len(row) < per_row:
            if row and rng.random() < 0.15:
                row.append(None)
                continue
            row.append(rng.choice(pool))
        items.append(row)
    sizes = []
    for row in items:
        sz = []
        for it in row:
            if it is None:
                sz.append(int(h * 0.7))
            else:
                im = it[0]
                sz.append(max(4, int(im.width * h / max(1, im.height))))
        sizes.append(sz)
    gap = int(h * rng.uniform(0.15, 0.45))
    lgap = int(h * rng.uniform(0.3, 0.8))
    margin = int(h * rng.uniform(0.4, 1.2))
    W = 2 * margin + max(sum(sz) + gap * (len(sz) - 1) for sz in sizes)
    Ht = 2 * margin + rows * h + (rows - 1) * lgap
    ref = np.array(rng.choice(pool)[0].resize((8, 8))).reshape(-1, 3)
    base = float(np.median(ref))
    page = Image.fromarray(_paper(rng, W, Ht, base, rng.uniform(2, 10), rng.uniform(1, 4)))
    boxes = []
    y = margin
    for row, sz in zip(items, sizes):
        x = margin
        for it, w in zip(row, sz):
            if it is not None:
                page.paste(it[0].resize((w, h), Image.BILINEAR), (x, y))
                boxes.append((x, y, x + w, y + h, it[1]))
            x += w + gap
        y += h + lgap
    if rng.random() < 0.7:
        page, boxes = augment(rng, page, boxes, strength=rng.uniform(0.2, 0.8))
    return page, boxes


def save_crop(out, split, code, source, idx, img):
    d = os.path.join(out, "cls", split, f"{code:02d}")
    os.makedirs(d, exist_ok=True)
    img.convert("RGB").resize((64, 96), Image.BILINEAR).save(os.path.join(d, f"{source}_{idx:07d}.jpg"), quality=92)


def crop_box(img, b, pad=0.08):
    x1, y1, x2, y2, c = b
    w, h = x2 - x1, y2 - y1
    return img.crop((int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)), int(min(img.width, x2 + pad * w)), int(min(img.height, y2 + pad * h))))


def write_yaml(out):
    names = {i: n for i, n in enumerate(CLASS_NAMES)}
    import yaml
    base = {"path": out, "train": "images/train", "val": "images/val",
            "test": [f"images/test_{t}" for t in TEST_DOMAINS], "nc": 63, "names": names}
    with open(os.path.join(out, "data.yaml"), "w") as f:
        yaml.safe_dump(base, f, allow_unicode=True, sort_keys=False)
    for t in TEST_DOMAINS:
        cfg = dict(base, val=f"images/test_{t}", test=f"images/test_{t}")
        with open(os.path.join(out, f"data_test_{t}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def report(out, w, crop_stats, elapsed, sources_found):
    stats = {
        "imagens": {f"{d}|{s}": n for (d, s), n in sorted(w.count.items())},
        "celulas": {f"{d}|{s}": n for (d, s), n in sorted(w.cells.items())},
        "celulas_por_classe": {CLASS_NAMES[c - 1]: n for c, n in sorted(w.class_counts.items())},
        "recortes_classificador": {f"{a}|{b}": n for (a, b), n in sorted(crop_stats.items())},
        "fontes": sources_found,
        "tempo_segundos": round(elapsed, 1),
    }
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    by_domain = defaultdict(lambda: Counter())
    for (d, s), n in w.count.items():
        by_domain[d][s] += n
    lines = ["# Relatório do dataset", "", "| Domínio | Treino | Validação | Teste | Células anotadas |", "|---|---|---|---|---|"]
    for d, c in sorted(by_domain.items()):
        test = sum(v for k, v in c.items() if k.startswith("test"))
        cells = sum(n for (dd, _), n in w.cells.items() if dd == d)
        lines.append(f"| {d} | {c['train']} | {c['val']} | {test} | {cells} |")
    tot_i = sum(w.count.values())
    tot_c = sum(w.cells.values())
    lines += [f"| **Total** | | | | **{tot_c}** em **{tot_i}** imagens |", "",
              f"Classes com exemplos: {len(w.class_counts)} de 63", "",
              f"Blocos do Angelina descartados por anotação parcial: {w.skipped}", "",
              f"Recortes para o classificador: {sum(crop_stats.values())}", ""]
    with open(os.path.join(out, "RELATORIO_DATASET.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        codes = list(range(1, 64))
        vals = [w.class_counts.get(c, 0) for c in codes]
        plt.figure(figsize=(14, 4))
        plt.bar(range(63), vals, color="#0E7C86")
        plt.xticks(range(63), [CLASS_NAMES[c - 1] for c in codes], rotation=90, fontsize=6)
        plt.ylabel("Células anotadas")
        plt.title("Distribuição das 63 classes (padrões de pontos)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "class_distribution.png"), dpi=130)
        plt.close()
    except Exception:
        pass
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", type=int, default=1500)
    ap.add_argument("--mosaic", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--tile-above", type=int, default=1300)
    ap.add_argument("--crops-per-class", type=int, default=300)
    ap.add_argument("--min-cells-angelina", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    rng = random.Random(a.seed)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    w = Writer(a.out)
    zips = sorted(os.path.join(a.raw, f) for f in os.listdir(a.raw) if f.lower().endswith(".zip"))
    det, crops_src, found = [], [], {}
    for zp in zips:
        kinds = S.detect_kinds(zipfile.ZipFile(zp).namelist())
        found[os.path.basename(zp)] = sorted(kinds)
        if "feedback" in kinds:
            det += list(S.read_feedback(zp))
        elif "yolo" in kinds:
            det += list(S.read_yolo(zp))
        if "angelina" in kinds:
            det += list(S.read_angelina(zp))
        if "multi" in kinds:
            crops_src += list(S.read_multi(zp)) + list(S.read_multi_cropped(zp))
        if "kaggle" in kinds:
            crops_src += list(S.read_kaggle(zp))
        if "classification" in kinds:
            crops_src += list(S.read_classification(zp))
    print("Fontes encontradas:", json.dumps(found, ensure_ascii=False))
    crop_stats = Counter()
    per_class = Counter()

    def maybe_crop(split, code, source, img):
        key = (split, code, source)
        cap = a.crops_per_class if split == "train" else max(20, a.crops_per_class // 5)
        if per_class[key] >= cap:
            return
        per_class[key] += 1
        crop_stats[(source, split)] += 1
        save_crop(a.out, split, code, source, sum(crop_stats.values()), img)

    for i, s in enumerate(det):
        if s["kind"] == "angelina":
            img, boxes = S.load_angelina_sample(s)
        else:
            img, boxes = S.load_yolo_sample(s)
        img, boxes = limit_size(img, boxes, a.max_side)
        sd = split_dir(s["split"], s["domain"])
        cls_split = "test" if s["split"] == "test" else s["split"]
        for b in rng.sample(boxes, min(len(boxes), 40)):
            maybe_crop(cls_split, b[4], s["domain"], crop_box(img, b))
        for tim, tb in tiles(img, boxes, a.tile_above):
            if s["domain"] == "angelina" and len(tb) < a.min_cells_angelina:
                w.skipped += 1
                continue
            w.add(tim, tb, sd, s["domain"])
        if i % 200 == 0:
            print(f"  detecção {i}/{len(det)}", flush=True)
    pools = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(crops_src):
        if s["kind"] == "multi":
            items = S.load_multi_crops(s)
        elif s["kind"] == "multi_cropped":
            items = S.load_multi_cropped(s)
        else:
            items = [S.load_crop(s)]
        for im, code in items:
            pools[s["split"]][s["domain"]].append((im, code))
            maybe_crop(s["split"], code, s["domain"], im)
        if i % 1000 == 0:
            import resource
            print(f"  recortes {i}/{len(crops_src)} mem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}MB", flush=True)
    jobs = [(a.seed * 1000003 + i, "relief" if i % 2 else "ink") for i in range(a.synthetic)]
    if jobs:
        with mp.get_context("spawn").Pool(a.workers) as p:
            for i, (img, boxes) in enumerate(p.imap(_synth_job, jobs, chunksize=8)):
                r = i % 20
                split = "test" if r == 0 else ("val" if r == 1 else "train")
                w.add(img, boxes, split_dir(split, "sintetico"), "sintetico")
                for b in random.Random(i).sample(boxes, min(len(boxes), 3)):
                    maybe_crop(split, b[4], "sintetico", crop_box(img, b))
                if i % 500 == 0:
                    print(f"  sintéticas {i}/{len(jobs)}")
    weights = {"multi": 0.45, "multi_recorte": 0.25, "kaggle": 0.15, "classificacao": 0.15}
    for i in range(a.mosaic):
        r = i % 20
        split = "test" if r == 0 else ("val" if r in (1, 2) else "train")
        avail = {k: v for k, v in pools[split].items() if v}
        if not avail:
            continue
        keys = list(avail)
        src = rng.choices(keys, weights=[weights.get(k, 0.1) for k in keys])[0]
        page, boxes = mosaic_page(rng, avail[src])
        w.add(page, boxes, split_dir(split, "mosaico"), "mosaico")
        if i % 500 == 0:
            print(f"  mosaicos {i}/{a.mosaic}")
    write_yaml(a.out)
    report(a.out, w, crop_stats, time.time() - t0, found)


if __name__ == "__main__":
    main()
