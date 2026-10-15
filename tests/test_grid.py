import io
import os
import random
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageFilter
from src.braille_render import render_braille
from src.braille_grid import decode

WORDS = ("dor agua banheiro medico enfermeira remedio frio calor fome sede familia ajuda urgente "
         "falta de ar peito cabeca nausea tontura posicao cama hospital paciente braille texto "
         "coração ação não está você sábado criança pressão também açúcar então até já").split()


def norm(t):
    return " ".join(t.split())


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def make_case(rng):
    n = rng.randint(1, 6)
    words = [rng.choice(WORDS) for _ in range(n)]
    if rng.random() < 0.3:
        words[0] = words[0].capitalize()
    if rng.random() < 0.25:
        words.append(str(rng.randint(1, 999)))
    text = " ".join(words)
    if rng.random() < 0.3:
        text += rng.choice([".", "?", "!", ","])
    style = {
        "dot_radius": rng.choice([2, 3, 4, 6, 9, 14, 20]),
        "spacing": rng.uniform(1.5, 1.9),
        "pitch": 0.0,
        "line_pitch": rng.uniform(6.5, 8.0),
        "max_cells": rng.randint(8, 25),
        "show_empty": rng.random() < 0.5,
        "ink": rng.choice([(0, 0, 0), (20, 20, 20), (14, 124, 134), (40, 40, 120), (150, 20, 20)]),
    }
    style["pitch"] = style["spacing"] * rng.uniform(2.3, 3.2)
    img = render_braille(text, **style)
    tags = []
    if rng.random() < 0.4:
        q = rng.choice([50, 70, 90])
        img = jpeg(img, q)
        tags.append(f"jpeg{q}")
    if rng.random() < 0.15 and style["dot_radius"] >= 4:
        img = img.filter(ImageFilter.GaussianBlur(0.8))
        tags.append("blur")
    if rng.random() < 0.1:
        img = Image.eval(img, lambda v: 255 - v)
        tags.append("invert")
    return text, img, style, tags


def run(n=300, seed=7):
    rng = random.Random(seed)
    exact = 0
    chars_ok = 0
    chars_total = 0
    fails = []
    for i in range(n):
        text, img, style, tags = make_case(rng)
        out = decode(img)["text"].replace("\n", " ")
        a, b = norm(text), norm(out)
        if a == b:
            exact += 1
        else:
            fails.append((text, out, style["dot_radius"], tags))
        import difflib
        sm = difflib.SequenceMatcher(None, a, b)
        chars_ok += sum(m.size for m in sm.get_matching_blocks())
        chars_total += max(len(a), len(b))
    print(f"Casos: {n}")
    print(f"Acerto exato da frase: {exact}/{n} = {100 * exact / n:.1f}%")
    print(f"Acerto por caractere: {100 * chars_ok / chars_total:.2f}%")
    for f in fails[:15]:
        print("FALHA:", f)
    return exact / n


if __name__ == "__main__":
    run()
