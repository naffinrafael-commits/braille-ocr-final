import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


cells = [
    md("# ⠿ TG Braille OCR: treino final",
       "**FATEC Cotia, Ciência de Dados.** Rode as células de cima para baixo, uma de cada vez.",
       "",
       "| Etapa | O que faz | Tempo aproximado (T4) |",
       "|---|---|---|",
       "| 1 a 3 | Confere GPU, Drive e arquivos | 3 min |",
       "| 4 | Monta o dataset unificado com todas as bases | 15 min |",
       "| 5 | Treina o YOLO11n de 63 classes (modelo principal) | 70 min |",
       "| 6 | Treina o YOLO26n (comparação de arquitetura) | 70 min |",
       "| 7 | Teste de resolução 416 px | 30 min |",
       "| 8 | Abordagem em 2 etapas (detector + MobileNetV2) | 60 min |",
       "| 9 | Avaliação comparativa por domínio | 15 min |",
       "| 10 | Teste com Braille em português (caixas de remédio) | 5 min |",
       "| 11 | Empacota os pesos para o Codespaces | 1 min |",
       "",
       "**Sessão 1:** etapas 1 a 5. **Sessão 2 (outro dia):** rode tudo de novo; o que já terminou é pulado automaticamente.",
       "Tudo fica salvo no Drive. Se a sessão cair, rode de novo do começo: o treino continua da última época salva."),
    md("## 1. GPU"),
    code("""
import torch
print("GPU disponível:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ATENÇÃO: ative a GPU em Ambiente de execução > Alterar o tipo de ambiente de execução > T4 GPU")
"""),
    md("## 2. Configuração", "Só mude se quiser economizar tempo de GPU (por exemplo, `RUN_416 = False`)."),
    code("""
BASE = "/content/drive/MyDrive/braille-ocr"
EPOCHS_MAIN = 60
EPOCHS_YOLO26 = 60
EPOCHS_416 = 30
EPOCHS_CELL = 40
EPOCHS_CLS = 15
RUN_YOLO26 = True
RUN_416 = True
RUN_TWO_STAGE = True
N_SYNTH = 1500
N_MOSAIC = 1500
DATA = "/content/dataset_final"
PROJECT = "/content/braille-ocr-final"
"""),
    md("## 3. Drive e arquivos",
       "Na pasta `braille-ocr` do Drive devem estar o `braille-ocr-final.zip` e a subpasta `datasets` com os ZIPs das bases.",
       "Na primeira execução, o Colab pede permissão para acessar o Drive: clique em **Conectar** e escolha sua conta."),
    code("""
import os, glob, shutil, zipfile
from google.colab import drive
drive.mount("/content/drive")
RUNS = f"{BASE}/runs"
RESULTS = f"{BASE}/results"
for p in (RUNS, RESULTS, f"{BASE}/datasets"):
    os.makedirs(p, exist_ok=True)
proj_zip = f"{BASE}/braille-ocr-final.zip"
raw = sorted(glob.glob(f"{BASE}/datasets/*.zip"))
pt_tests = [z for z in raw if "teste_portugues" in os.path.basename(z).lower()]
print("Projeto:", "OK" if os.path.exists(proj_zip) else "FALTANDO braille-ocr-final.zip em " + BASE)
print("Arquivos em datasets:")
for z in raw:
    print("  -", os.path.basename(z), f"({os.path.getsize(z) / 1e6:.0f} MB)")
if not raw:
    print("  NENHUM: suba os ZIPs das bases em", f"{BASE}/datasets")
print("Teste em português:", os.path.basename(pt_tests[0]) if pt_tests else "ainda não enviado (opcional nesta sessão)")
"""),
    code("""
!pip install -q ultralytics pillow-heif
if os.path.exists(PROJECT):
    shutil.rmtree(PROJECT)
with zipfile.ZipFile(proj_zip) as z:
    z.extractall("/content")
%cd {PROJECT}
import sys
sys.path.insert(0, PROJECT)
print("Projeto pronto em", PROJECT)
"""),
    md("## 4. Dataset unificado",
       "Converte Roboflow, Angelina, Multi Braille, Kaggle, Classification e correções salvas pelo app para 63 classes (padrões de pontos), "
       "agrupa fotos repetidas para não haver vazamento entre treino e teste, gera imagens sintéticas e páginas montadas com recortes reais. "
       "O ZIP do teste em português é ignorado aqui de propósito: ele só é usado na etapa 10."),
    code("""
if not os.path.exists(f"{DATA}/data.yaml"):
    !python data/build_dataset.py --raw "{BASE}/datasets" --out {DATA} --synthetic {N_SYNTH} --mosaic {N_MOSAIC} --workers 2
!python tests/check_dataset.py --data {DATA} --out {DATA}/conferencia_dataset.jpg
for f in ("RELATORIO_DATASET.md", "stats.json", "class_distribution.png", "conferencia_dataset.jpg"):
    if os.path.exists(f"{DATA}/{f}"):
        shutil.copy2(f"{DATA}/{f}", RESULTS)
from IPython.display import Markdown, Image as IImage, display
display(Markdown(open(f"{DATA}/RELATORIO_DATASET.md", encoding="utf-8").read()))
display(IImage(f"{DATA}/class_distribution.png"))
"""),
    md("Confira o mosaico abaixo: cada caixa vermelha deve envolver uma célula e a letra em azul deve corresponder aos pontos. "
       "**Na primeira vez, pare aqui e mande os prints da tabela e do mosaico antes de seguir.**"),
    code("""
display(IImage(f"{DATA}/conferencia_dataset.jpg", width=1100))
"""),
    md("## 5. Modelo principal: YOLO11n, 63 classes",
       "Hiperparâmetros fixos para comparações justas: SGD, lr0 = 0,01, paciência 30. "
       "`fliplr = 0` é obrigatório: espelhar a imagem troca o ponto 1 pelo 4 e muda a letra."),
    code("""
from ultralytics import YOLO

def train_or_resume(name, base_weights, **kw):
    run_dir = f"{RUNS}/{name}"
    done, best, last = f"{run_dir}/DONE", f"{run_dir}/weights/best.pt", f"{run_dir}/weights/last.pt"
    if os.path.exists(done):
        print(name, "já treinado:", best)
        return best
    if os.path.exists(last):
        print("Retomando", name)
        try:
            YOLO(last).train(resume=True)
        except Exception as e:
            print("Retomada encerrada:", e)
    else:
        YOLO(base_weights).train(project=RUNS, name=name, exist_ok=True, **kw)
    open(done, "w").write("ok")
    return best

COMMON = dict(data=f"{DATA}/data.yaml", batch=16, patience=30, optimizer="SGD", lr0=0.01, momentum=0.937,
              weight_decay=0.0005, cos_lr=True, close_mosaic=10, fliplr=0.0, flipud=0.0, degrees=5.0,
              iou=0.45, max_det=1500, plots=True, device=0, workers=2, seed=42)
BEST_MAIN = train_or_resume("yolo11n_63", "yolo11n.pt", epochs=EPOCHS_MAIN, imgsz=640, **COMMON)
"""),
    md("## 6. Comparação de arquitetura: YOLO26n", "Mesmo dataset e mesmos hiperparâmetros do YOLO11n, mudando só a arquitetura."),
    code("""
BEST_26 = None
if RUN_YOLO26:
    try:
        BEST_26 = train_or_resume("yolo26n_63", "yolo26n.pt", epochs=EPOCHS_YOLO26, imgsz=640, **COMMON)
    except Exception as e:
        print("YOLO26n indisponível nesta versão da Ultralytics, seguindo sem ele:", e)
"""),
    md("## 7. Teste metodológico de resolução (416 px)"),
    code("""
BEST_416 = train_or_resume("yolo11n_63_416", "yolo11n.pt", epochs=EPOCHS_416, imgsz=416, **COMMON) if RUN_416 else None
"""),
    md("## 8. Abordagem em 2 etapas", "Etapa 1: detector de células (classe única). Etapa 2: MobileNetV2 classifica cada recorte entre os 63 padrões."),
    code("""
BEST_CELL, CLS = None, f"{RESULTS}/cell_classifier.pt"
if RUN_TWO_STAGE:
    BEST_CELL = train_or_resume("detector_celulas", "yolo11n.pt", epochs=EPOCHS_CELL, imgsz=640, single_cls=True, **COMMON)
    if not os.path.exists(CLS):
        !python train/train_classifier.py --data {DATA}/cls --out "{CLS}" --epochs {EPOCHS_CLS}
    print(open(CLS.replace(".pt", "_metrics.json"), encoding="utf-8").read())
"""),
    md("## 9. Avaliação comparativa por domínio",
       "Compara YOLO11n, YOLO26n, 416 px, 2 etapas e o método clássico por grade em cada conjunto de teste."),
    code("""
args = f'--data {DATA} --single "{BEST_MAIN}" --out "{RESULTS}/avaliacao"'
if BEST_26 and os.path.exists(BEST_26):
    args += f' --yolo26 "{BEST_26}"'
if BEST_416 and os.path.exists(BEST_416):
    args += f' --single416 "{BEST_416}"'
if BEST_CELL and os.path.exists(CLS):
    args += f' --cell "{BEST_CELL}" --cls "{CLS}"'
!python train/evaluate.py {args}
display(Markdown(open(f"{RESULTS}/avaliacao/RELATORIO_RESULTADOS.md", encoding="utf-8").read()))
"""),
    md("## 11. Pacote de pesos para o Codespaces", "(A etapa 10 vem logo abaixo e pode ser rodada depois, quando o teste em português estiver pronto.)"),
    code("""
def pack():
    root = "/content/pesos"
    w = f"{root}/train/weights"
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(w)
    copies = [(BEST_MAIN, "best.pt"), (BEST_26, "yolo26.pt"), (BEST_CELL, "cell_detector.pt"), (CLS, "cell_classifier.pt")]
    for src, dst in copies:
        if src and os.path.exists(src):
            shutil.copy2(src, f"{w}/{dst}")
    for f in ("results.png", "confusion_matrix_normalized.png", "results.csv"):
        if os.path.exists(f"{RUNS}/yolo11n_63/{f}"):
            shutil.copy2(f"{RUNS}/yolo11n_63/{f}", w)
    extras = ("avaliacao/metrics.json", "avaliacao/RELATORIO_RESULTADOS.md", "RELATORIO_DATASET.md", "class_distribution.png",
              "conferencia_dataset.jpg", "stats.json", "teste_portugues.json", "RELATORIO_TESTE_PORTUGUES.md", "cell_classifier_metrics.json")
    for f in extras:
        if os.path.exists(f"{RESULTS}/{f}"):
            shutil.copy2(f"{RESULTS}/{f}", w)
    shutil.make_archive(f"{BASE}/pesos_para_codespaces", "zip", root)
    print("Pronto:", f"{BASE}/pesos_para_codespaces.zip")
    print(sorted(os.listdir(w)))

pack()
"""),
    md("## 10. Teste com Braille em português (caixas de remédio)",
       "Coloque o `teste_portugues.zip` (fotos + planilha) em `braille-ocr/datasets` e rode esta célula. "
       "Ela atualiza o pacote de pesos com o relatório do teste. Pode ser feita numa sessão separada: rode antes as células 1 a 3 e a 5 a 8 (todas são puladas porque já terminaram)."),
    code("""
pt_tests = [z for z in glob.glob(f"{BASE}/datasets/*.zip") if "teste_portugues" in os.path.basename(z).lower()]
if not pt_tests:
    print("Envie o teste_portugues.zip para", f"{BASE}/datasets")
else:
    staging = "/content/pesos_teste/train/weights"
    os.makedirs(staging, exist_ok=True)
    for src, dst in [(BEST_MAIN, "best.pt"), (BEST_26, "yolo26.pt"), (BEST_CELL, "cell_detector.pt"), (CLS, "cell_classifier.pt")]:
        if src and os.path.exists(src):
            shutil.copy2(src, f"{staging}/{dst}")
    os.environ["BRAILLE_WEIGHTS_DIR"] = staging
    !python tools/teste_portugues.py --zip "{pt_tests[0]}" --out "{RESULTS}"
    display(Markdown(open(f"{RESULTS}/RELATORIO_TESTE_PORTUGUES.md", encoding="utf-8").read()))
    pack()
"""),
    md("## Próximo passo",
       "Baixe `pesos_para_codespaces.zip` da pasta `braille-ocr` do Drive e siga o guia do README para colocá-lo no Codespaces.",
       "",
       "Para melhorar o modelo depois: gere o `feedback_dataset.zip` na aba Melhoria contínua do app, coloque em `braille-ocr/datasets`, "
       "apague no Drive as pastas `runs/yolo11n_63` e `results/avaliacao` e rode este notebook de novo."),
]

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": cells}
out = os.path.join(ROOT, "notebooks", "TG_Braille_Final.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(out)
