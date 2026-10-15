import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image

from src.detector import available, detect, to_pil
from src.postprocess import detections_to_text
from src.text_metrics import cer, contains_reference, normalize

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")
MODES = ["auto", "grade", "yolo11", "yolo26", "duas_etapas"]


def read_sheet(name, data):
    rows = []
    if name.lower().endswith((".xlsx", ".xls")):
        import pandas as pd
        df = pd.read_excel(io.BytesIO(data))
        for _, r in df.iterrows():
            rows.append((str(r.iloc[0]).strip(), str(r.iloc[1]).strip()))
        return rows
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = ";" if text.count(";") >= text.count(",") else ","
    for r in csv.reader(io.StringIO(text), delimiter=sep):
        if len(r) >= 2 and r[0].strip():
            rows.append((r[0].strip(), r[1].strip()))
    return rows


def load(zip_path):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if not n.endswith("/") and "__MACOSX" not in n]
    sheets = [n for n in names if n.lower().endswith((".csv", ".xlsx", ".xls"))]
    if not sheets:
        raise ValueError("Nenhuma planilha (.csv ou .xlsx) encontrada no ZIP")
    rows = read_sheet(sheets[0], z.read(sheets[0]))
    by_name = {os.path.basename(n).lower(): n for n in names if n.lower().endswith(IMG_EXT)}
    by_stem = {os.path.splitext(k)[0]: v for k, v in by_name.items()}
    items, missing = [], []
    for fname, ref in rows:
        key = os.path.basename(fname).lower()
        if key in ("arquivo", "file", "nome") or not ref:
            continue
        member = by_name.get(key) or by_stem.get(os.path.splitext(key)[0])
        if member is None:
            missing.append(fname)
            continue
        items.append((os.path.basename(member), ref, z.read(member)))
    return items, missing


def read_text(img, mode):
    res = detect(img, mode)
    text = res.get("text")
    if text is None:
        text = detections_to_text(res.get("detections", []))
    return text.replace("\n", " "), res.get("method", mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "weights"))
    ap.add_argument("--modes", default=",".join(MODES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items, missing = load(a.zip)
    modes = [m for m in a.modes.split(",") if m and available(m)]
    print(f"{len(items)} fotos com texto de referência; modos: {', '.join(modes)}")
    if missing:
        print("Fotos citadas na planilha e não encontradas no ZIP:", ", ".join(missing))
    per_photo = []
    summary = {}
    for m in modes:
        cers, cers_na, exact, cont, t = [], [], 0, 0, 0.0
        for i, (fname, ref, data) in enumerate(items):
            img = to_pil(Image.open(io.BytesIO(data)))
            t0 = time.time()
            try:
                hyp, used = read_text(img, m)
            except Exception as e:
                hyp, used = "", f"erro: {e}"
            t += time.time() - t0
            c, cn = cer(ref, hyp), cer(ref, hyp, accents=False)
            cers.append(c)
            cers_na.append(cn)
            exact += int(normalize(ref) == normalize(hyp))
            cont += int(contains_reference(ref, hyp))
            if len(per_photo) <= i:
                per_photo.append({"arquivo": fname, "referencia": ref})
            per_photo[i][m] = {"lido": hyp, "cer": round(c, 4), "metodo": used}
        n = max(1, len(items))
        summary[m] = {"cer_medio": round(sum(cers) / n, 4), "cer_sem_acento": round(sum(cers_na) / n, 4),
                      "leituras_exatas": round(exact / n, 4), "nome_contido": round(cont / n, 4), "segundos_por_foto": round(t / n, 3)}
        print(m, summary[m])
    with open(os.path.join(a.out, "teste_portugues.json"), "w", encoding="utf-8") as f:
        json.dump({"resumo": summary, "fotos": per_photo, "nao_encontradas": missing}, f, ensure_ascii=False, indent=2)
    lines = [f"Avaliação em {len(items)} fotos reais de Braille em português, comparando a leitura com o texto digitado pelo grupo.", "",
             "| Modo | CER | CER sem acentos | Leituras exatas | Texto encontrado na leitura | Segundos por foto |", "|---|---|---|---|---|---|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {100 * s['cer_medio']:.1f}% | {100 * s['cer_sem_acento']:.1f}% | {100 * s['leituras_exatas']:.0f}% | {100 * s['nome_contido']:.0f}% | {s['segundos_por_foto']} |")
    lines += ["", "CER: proporção de caracteres que precisariam ser corrigidos (0% é perfeito). "
              "Texto encontrado: o texto de referência aparece dentro da leitura, mesmo com símbolos extras ao redor.", "",
              "| Foto | Referência | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for p in per_photo:
        lines.append(f"| {p['arquivo']} | {p['referencia']} | " + " | ".join(p.get(m, {}).get("lido", "") for m in modes) + " |")
    with open(os.path.join(a.out, "RELATORIO_TESTE_PORTUGUES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Relatório:", os.path.join(a.out, "RELATORIO_TESTE_PORTUGUES.md"))


if __name__ == "__main__":
    main()
