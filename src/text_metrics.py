import re
import unicodedata


def normalize(text, accents=True):
    t = " ".join(str(text).lower().split())
    if not accents:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t


def levenshtein(a, b):
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis, accents=True):
    r, h = normalize(reference, accents), normalize(hypothesis, accents)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def contains_reference(reference, hypothesis):
    r = re.sub(r"\s+", "", normalize(reference))
    h = re.sub(r"\s+", "", normalize(hypothesis))
    return bool(r) and r in h
