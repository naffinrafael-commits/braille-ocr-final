# ⠿ Braille OCR: versão final

Reconhecimento óptico de Braille em língua portuguesa a partir de fotografias. TG em Ciência de Dados, FATEC Cotia, 2026.

## Como o sistema lê

| Tipo de imagem | Método |
|---|---|
| Braille digital ou escaneado (pontos de tinta) | Leitura por grade |
| Foto de Braille em relevo | YOLO11n de 63 classes (com YOLO26n e 2 etapas como alternativas) |

O modo **Automático** tenta a grade e, se a confiança ficar baixa, trata a imagem como foto.

## Passo a passo resumido

1. **Drive:** pasta `braille-ocr` com `braille-ocr-final.zip`; subpasta `datasets` com os ZIPs das bases
2. **Colab:** upload de `notebooks/TG_Braille_Final.ipynb`, GPU T4, rodar as células em ordem (duas sessões)
3. **Codespaces:** extrair o projeto e o `pesos_para_codespaces.zip`, instalar e rodar:
```bash
unzip -o braille-ocr-final.zip
cd braille-ocr-final
unzip -o ../pesos_para_codespaces.zip
pip install -r requirements.txt
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-python-headless
streamlit run app.py
```
4. **Teste em português:** fotos das caixas de remédio + planilha (`tools/modelo_planilha_teste_portugues.csv`) num ZIP chamado `teste_portugues.zip`, em `braille-ocr/datasets`; rodar a etapa 10 do notebook
5. **Melhoria contínua:** corrigir leituras no app, gerar o pacote na aba Melhoria contínua e retreinar

## Estrutura

```
app.py                         interface (4 abas)
src/braille_grid.py            leitura por grade (método clássico)
src/braille_tables.py          Grafia Braille portuguesa e 63 classes
src/detector.py                modo automático, YOLO11n, YOLO26n, 2 etapas, Roboflow
src/two_stage.py               classificador MobileNetV2
src/postprocess.py             caixas para texto na ordem de leitura
src/text_metrics.py            CER e comparação de textos
src/feedback.py                correções salvas pelo app
data/sources.py                leitores de cada base
data/synth.py                  imagens sintéticas (tinta e relevo)
data/build_dataset.py          dataset unificado
train/train_classifier.py      treino do MobileNetV2
train/evaluate.py              avaliação comparativa por domínio
tools/teste_portugues.py       avaliação nas fotos em português
tools/feedback_to_yolo.py      correções viram dados de treino
notebooks/TG_Braille_Final.ipynb
tests/test_grid.py             300 testes da leitura por grade
tests/check_dataset.py         validação e conferência visual dos rótulos
docs/METODOLOGIA_DADOS.md      texto de metodologia para a monografia
```

## Equipe

Flávio Vieira de Araujo · Rafael Naffin de Almeida · Renan Kaique Fidelis do Amaral · Ricardo Henrique Guimarães
Orientadora: Profª. Micaella Garcia Fontes Gatti
