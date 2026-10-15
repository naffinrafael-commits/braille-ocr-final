import io
import os
import sys

import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.braille_render import render_braille
from src.braille_tables import text_to_cells, dots_to_unicode

st.set_page_config(page_title="Braille OCR", page_icon="⠿", layout="wide")
st.markdown(
    "<style>.txt{font-size:1.6rem;font-weight:600;padding:1rem 1.2rem;background:#eef6f6;border-left:6px solid #0E7C86;"
    "border-radius:8px;white-space:pre-wrap;margin:.4rem 0}.bu{font-size:2.4rem;letter-spacing:6px;padding:1rem;"
    "background:#f6f8f8;border-radius:8px;word-break:break-all}</style>",
    unsafe_allow_html=True,
)
WEIGHTS = os.path.join(ROOT, "train", "weights")
MODES = {
    "auto": "Automático (recomendado)",
    "grade": "Leitura por grade (imagem digital ou escaneada)",
    "yolo11": "YOLO11n, 1 etapa (fotos)",
    "yolo26": "YOLO26n, 1 etapa (fotos)",
    "duas_etapas": "Detector + MobileNetV2, 2 etapas (fotos)",
    "roboflow": "API Roboflow (comparação)",
}
METHOD = {"grade": "Grade", "yolo11": "YOLO11n", "yolo26": "YOLO26n", "duas_etapas": "2 etapas", "roboflow": "Roboflow"}


def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except Exception:
        st.image(img, caption=caption, use_container_width=True)


def big_button(label, key):
    try:
        return st.button(label, type="primary", key=key, width="stretch")
    except Exception:
        return st.button(label, type="primary", key=key, use_container_width=True)


@st.cache_data(show_spinner=False)
def model_status():
    try:
        from src.detector import status
        return status()
    except Exception as e:
        return {"Modelos": (False, str(e))}


def available(mode):
    try:
        from src.detector import available as av
        return av(mode)
    except Exception:
        return mode in ("auto", "grade")


def show_md(path, title=None):
    if os.path.exists(path):
        if title:
            st.markdown(f"#### {title}")
        st.markdown(open(path, encoding="utf-8").read())
        return True
    return False


st.title("⠿ Braille OCR")
st.caption("Reconhecimento óptico de Braille em língua portuguesa a partir de fotografias")

with st.sidebar:
    st.header("Configurações")
    modes = [m for m in MODES if m in ("auto", "grade", "roboflow") or available(m)]
    mode = st.selectbox("Modo de leitura", modes, format_func=lambda m: MODES[m])
    st.markdown("---")
    st.markdown("**Modelos encontrados**")
    for name, (ok, info) in model_status().items():
        (st.success if ok else st.warning)(f"{name}: {info}")

tab_ocr, tab_txt, tab_perf, tab_loop = st.tabs(["📷 Imagem → Texto", "⌨️ Texto → Braille", "📊 Desempenho", "🔁 Melhoria contínua"])

with tab_ocr:
    col1, col2 = st.columns(2)
    image = None
    with col1:
        st.subheader("Entrada")
        source = st.radio("Origem da imagem", ["Enviar arquivo", "Tirar foto"], horizontal=True)
        if source == "Enviar arquivo":
            up = st.file_uploader("Imagem com Braille", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if up is not None:
                image = Image.open(up)
        else:
            shot = st.camera_input("Aponte a câmera para o texto em Braille")
            if shot is not None:
                image = Image.open(shot)
        if image is not None:
            show_image(image, "Imagem recebida")
            if big_button("Ler Braille", "run"):
                with st.spinner("Lendo o Braille..."):
                    try:
                        from src.pipeline import run as run_pipeline
                        st.session_state["res"] = run_pipeline(image, mode=mode)
                        st.session_state.pop("saved", None)
                    except Exception as e:
                        st.session_state["res"] = None
                        st.error("Não foi possível analisar esta imagem.")
                        st.exception(e)
    with col2:
        st.subheader("Resultado")
        res = st.session_state.get("res")
        if res:
            text = res["text"] or ""
            if text.strip():
                st.markdown(f'<div class="txt">{text}</div>', unsafe_allow_html=True)
            else:
                st.info("Nenhum texto em Braille foi reconhecido nesta imagem.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Método", METHOD.get(res["method"], res["method"]))
            c2.metric("Confiança", f"{100 * float(res['quality']):.0f}%")
            c3.metric("Células", len(res["detections"]))
            if float(res["quality"]) < 0.6 and text.strip():
                st.warning("Confiança baixa: confira o texto. Pouca luz, inclinação ou desfoque reduzem a precisão.")
            for n in res.get("notes", []):
                st.caption(n)
            fixed = st.text_area("Corrija aqui se a leitura tiver erros", text, height=110, key="fixed")
            if fixed.strip() and fixed != text and st.button("Salvar correção para o próximo treino"):
                from src.feedback import save
                st.session_state["saved"] = save(res["original_image"], res, fixed)
            if st.session_state.get("saved"):
                st.success("Correção salva. Ela entra no próximo treino pela aba Melhoria contínua.")
            show_image(res["annotated_image"], "Verde: confiança alta · laranja: média · vermelho: baixa")
            with st.expander("Detalhes técnicos"):
                st.json({"metodo": res["method"], "qualidade": res["quality"], "parametros": res["params"]})
        else:
            st.markdown('<div style="padding:2rem;text-align:center;color:#999;border:2px dashed #ddd;border-radius:10px">'
                        '<div style="font-size:3rem">⠿</div><div>Aguardando imagem</div></div>', unsafe_allow_html=True)

with tab_txt:
    st.subheader("Converter texto para Braille")
    texto = st.text_area("Texto", placeholder="Ex.: Dipirona 500 mg", max_chars=300, height=100, key="txt")
    if texto.strip():
        cells = text_to_cells(texto)
        st.markdown(f'<div class="bu">{"".join(dots_to_unicode(d) if d else " " for _, d in cells)}</div>', unsafe_allow_html=True)
        img = render_braille(texto, dot_radius=10, labels=True)
        show_image(img, "Células Braille (escuro = ponto em relevo)")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("Baixar imagem PNG", buf.getvalue(), file_name="braille.png", mime="image/png")
        if st.button("Testar ida e volta (ler a imagem gerada)"):
            from src.pipeline import run as run_pipeline
            lido = " ".join(run_pipeline(render_braille(texto, dot_radius=10), mode="grade")["text"].split())
            if lido == " ".join(texto.split()):
                st.success(f"Leitura idêntica ao original: {lido}")
            else:
                st.error(f"Diferença encontrada. Lido: {lido}")
        with st.expander("Teclas do teclado Braille (F D S / J K L)"):
            keymap = {1: "F", 2: "D", 3: "S", 4: "J", 5: "K", 6: "L"}
            for lab, dots in cells[:40]:
                if dots:
                    st.markdown(f"**{lab}** → {' + '.join(keymap[d] for d in dots)}")

with tab_perf:
    st.subheader("Desempenho do modelo")
    found = show_md(os.path.join(WEIGHTS, "RELATORIO_RESULTADOS.md"))
    found = show_md(os.path.join(WEIGHTS, "RELATORIO_TESTE_PORTUGUES.md"), "Teste com Braille em português") or found
    if not found:
        st.info("Os resultados aparecem aqui depois do treino no Colab (pasta train/weights).")
    p = os.path.join(WEIGHTS, "RELATORIO_DATASET.md")
    if os.path.exists(p):
        with st.expander("Composição do dataset"):
            st.markdown(open(p, encoding="utf-8").read())
    for f, cap in (("class_distribution.png", "Distribuição das classes"), ("results.png", "Curvas de treino do YOLO11n"),
                   ("confusion_matrix_normalized.png", "Matriz de confusão do YOLO11n"), ("conferencia_dataset.jpg", "Conferência dos rótulos")):
        p = os.path.join(WEIGHTS, f)
        if os.path.exists(p):
            with st.expander(cap):
                show_image(Image.open(p))

with tab_loop:
    st.subheader("Melhoria contínua")
    st.markdown("Cada correção salva vira dado de treino. Gere o pacote, envie para a pasta "
                "`braille-ocr/datasets` do Drive e rode o notebook de novo: o modelo aprende com os erros que cometeu.")
    try:
        from src.feedback import count
        st.metric("Correções salvas", count())
    except Exception:
        st.metric("Correções salvas", 0)
    if st.button("Gerar pacote de correções"):
        from tools.feedback_to_yolo import build
        path, ok, skipped = build()
        st.success(f"{len(ok)} correções convertidas, {len(skipped)} ignoradas (quantidade de células diferente do texto).")
        with open(path, "rb") as f:
            st.download_button("Baixar feedback_dataset.zip", f.read(), file_name="feedback_dataset.zip", mime="application/zip")
