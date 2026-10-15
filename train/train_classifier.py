import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.two_stage import build_model, MEAN, STD


def loaders(root, size, batch, workers):
    h, w = size
    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.4, 0.4, 0.3, 0.05),
        transforms.RandomGrayscale(0.3),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD.tolist()),
    ])
    eval_tf = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist())])
    sets = {}
    for split, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
        path = os.path.join(root, split)
        if os.path.isdir(path):
            sets[split] = datasets.ImageFolder(path, transform=tf)
    classes = sets["train"].classes
    for s in sets.values():
        s.class_to_idx = {c: i for i, c in enumerate(classes)}
        s.samples = [(p, classes.index(os.path.basename(os.path.dirname(p)))) for p, _ in s.samples if os.path.basename(os.path.dirname(p)) in classes]
        s.targets = [t for _, t in s.samples]
    dl = {k: DataLoader(v, batch_size=batch, shuffle=(k == "train"), num_workers=workers, pin_memory=True) for k, v in sets.items()}
    return sets, dl, [int(c) for c in classes]


def evaluate(model, dl, device, dataset=None):
    model.eval()
    correct = total = 0
    per_source = Counter()
    per_source_ok = Counter()
    offset = 0
    with torch.no_grad():
        for x, y in dl:
            pred = model(x.to(device)).argmax(1).cpu()
            ok = (pred == y)
            correct += int(ok.sum())
            total += len(y)
            if dataset is not None:
                for j in range(len(y)):
                    src = os.path.basename(dataset.samples[offset + j][0]).rsplit("_", 1)[0]
                    per_source[src] += 1
                    per_source_ok[src] += int(ok[j])
            offset += len(y)
    acc = correct / max(1, total)
    by_src = {k: round(per_source_ok[k] / v, 4) for k, v in per_source.items()}
    return acc, by_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    size = (96, 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets, dl, codes = loaders(a.data, size, a.batch, a.workers)
    counts = Counter(sets["train"].targets)
    weights = torch.tensor([1.0 / max(1, counts.get(i, 1)) ** 0.5 for i in range(len(codes))], dtype=torch.float32)
    weights = weights / weights.mean()
    model = build_model(len(codes), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    best = -1.0
    history = []
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        for x, y in dl["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        val_acc, _ = evaluate(model, dl["val"], device) if "val" in dl else (0.0, {})
        history.append({"epoca": ep + 1, "acc_val": round(val_acc, 4), "seg": round(time.time() - t0, 1)})
        print(history[-1], flush=True)
        if val_acc > best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "codes": codes, "size": size}, a.out)
    ck = torch.load(a.out, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    result = {"acc_val_melhor": round(best, 4), "historico": history, "classes": len(codes)}
    if "test" in dl:
        acc, by_src = evaluate(model, dl["test"], device, sets["test"])
        result.update({"acc_teste": round(acc, 4), "acc_teste_por_fonte": by_src})
    with open(os.path.splitext(a.out)[0] + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "historico"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
