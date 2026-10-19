"""
Pré-Impressão Gráfica & Orçamentos
----------------------------------
Lê PDFs de engenharia/arquitetura, extrai o tamanho real da folha (CropBox/MediaBox),
calcula a cobertura de tinta por página e gera o orçamento pela tabela de preços.

Executar:
    pip install streamlit pymupdf numpy pandas
    streamlit run app.py
"""

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orçamento Gráfico • Pré-Impressão", page_icon="🖨️", layout="wide")

PT_TO_MM = 25.4 / 72.0

# ----------------------------------------------------------------------------
# TABELA DE PREÇOS (edite aqui se os valores mudarem)
# Chaves de tamanho: A0, A1, A2, A3 e ML (Metro Linear)
# ----------------------------------------------------------------------------
FORMATOS_MM = {  # (lado curto, lado longo)
    "A0": (841, 1189),
    "A1": (594, 841),
    "A2": (420, 594),
    "A3": (297, 420),
    "A4": (210, 297),
}

PLOTAGEM = {
    "PB - Linhas":     {"A0": 8.90,  "A1": 4.90,  "A2": 3.90,  "A3": 2.00, "ML": 8.00},
    "Color - Linhas":  {"A0": 12.90, "A1": 6.90,  "A2": 4.90,  "A3": 4.00, "ML": 12.00},
    "Color - Até 50%": {"A0": 25.00, "A1": 15.00, "A2": 10.00, "A3": 4.50, "ML": 23.00},
    "Color/PB - 80%":  {"A0": 40.00, "A1": 30.00, "A2": 20.00, "A3": 5.50, "ML": 35.00},
    "Color/PB - 100%": {"A0": 60.00, "A1": 45.00, "A2": 35.00, "A3": 5.50, "ML": 55.00},
}
ADICIONAL_90G = {"A0": 10.00, "A1": 7.00, "A2": 5.00, "ML": 12.00}

COPIA = {
    "PB - Linhas":    {"A0": 12.00, "A1": 6.00,  "A2": 4.00, "ML": 10.00},
    "Color - Linhas": {"A0": 16.00, "A1": 10.00, "A2": 7.00, "ML": 12.00},
}

DIGITALIZACAO = {"A0": 25.00, "A1": 20.00, "A2": 15.00}

FORA_PADRAO = "Fora de Padrão (Metro Linear)"


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    return f"{v:.1f}".replace(".", ",") + "%"


def num(v: float, casas: int = 2) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def classificar_formato(w_mm: float, h_mm: float, tol: float):
    """Retorna 'A0'..'A4' (independe da orientação) ou None se fora de padrão."""
    curto, longo = sorted((w_mm, h_mm))
    for nome, (c, l) in FORMATOS_MM.items():
        if abs(curto - c) <= tol and abs(longo - l) <= tol:
            return nome
    return None


def df_para_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    linhas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(linhas)


# ----------------------------------------------------------------------------
# Análise do PDF (dimensões + cobertura de tinta) — cacheada
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analisar_pdf(dados: bytes, max_px: int, branco: int, croma: int, frac_cor: float):
    doc = fitz.open(stream=dados, filetype="pdf")
    if doc.needs_pass:
        raise ValueError("PDF protegido por senha.")

    paginas = []
    for i, page in enumerate(doc, start=1):
        # page.rect = CropBox (limitada à MediaBox) já com a rotação aplicada
        r = page.rect
        w_mm, h_mm = r.width * PT_TO_MM, r.height * PT_TO_MM

        # Renderização em baixa resolução só para medir a área impressa
        lado_longo_pt = max(r.width, r.height)
        zoom = min(max_px / lado_longo_pt, 150 / 72)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False
        )
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        minimo = arr.min(axis=2)
        maximo = arr.max(axis=2)

        # Pixel "com tinta" = qualquer canal abaixo do limiar de branco
        cobertura = float((minimo < branco).mean() * 100)
        # Pixel "colorido" = diferença grande entre canais (cinza/preto não conta)
        colorido = float(((maximo.astype(np.int16) - minimo) > croma).mean() * 100)

        paginas.append(
            {
                "pagina": i,
                "w_mm": w_mm,
                "h_mm": h_mm,
                "cobertura": cobertura,
                "pct_colorido": colorido,
                "colorido": colorido >= frac_cor,
            }
        )
    doc.close()
    return paginas


# ----------------------------------------------------------------------------
# Precificação
# ----------------------------------------------------------------------------
def categoria_plotagem(cob, colorido, lim_linhas):
    if cob <= lim_linhas:
        return "Color - Linhas" if colorido else "PB - Linhas"
    if cob <= 50:
        return "Color - Até 50%"
    if cob <= 80:
        return "Color/PB - 80%"
    return "Color/PB - 100%"


def precificar(pg, servico, papel_90g, qtd, lim_linhas, tol, a4_como_a3):
    """Retorna (formato_label, categoria_label, valor, aviso|None)."""
    fmt = classificar_formato(pg["w_mm"], pg["h_mm"], tol)
    chave = fmt
    obs_a4 = ""

    if fmt is None:
        chave = "ML"
    elif fmt == "A4":
        if a4_como_a3:
            chave, obs_a4 = "A3", " (A4 cobrado como A3)"
        else:
            return fmt, "A4 – sem preço na tabela", 0.0, "A4 não existe na tabela de preços."

    formato_label = fmt if fmt else FORA_PADRAO
    metros = max(pg["w_mm"], pg["h_mm"]) / 1000.0  # metro linear = lado maior
    ml_txt = f" ({num(metros)} m)" if chave == "ML" else ""
    tam_txt = f"Metro Linear{ml_txt}" if chave == "ML" else chave

    # ---- Digitalização: preço fixo por tamanho
    if servico == "Digitalização":
        if chave not in DIGITALIZACAO:
            return formato_label, f"{tam_txt} – sem preço de digitalização", 0.0, \
                f"Digitalização não tem preço para {tam_txt}."
        return formato_label, f"{tam_txt} - PB/Color{obs_a4}", DIGITALIZACAO[chave] * qtd, None

    # ---- Cópia: só tabela de linhas
    if servico == "Cópia":
        if pg["cobertura"] > lim_linhas:
            return formato_label, f"{tam_txt} – cobertura acima de 'Linhas'", 0.0, \
                "Cópia só tem preço para plantas de linhas; cobertura acima do limite."
        cat = "Color - Linhas" if pg["colorido"] else "PB - Linhas"
        tabela = COPIA[cat]
        if chave not in tabela:
            return formato_label, f"{tam_txt} – sem preço de cópia", 0.0, \
                f"Cópia não tem preço para {tam_txt}."
        base = tabela[chave] * (metros if chave == "ML" else 1)
        return formato_label, f"{tam_txt} - {cat}{obs_a4}", base * qtd, None

    # ---- Plotagem
    cat = categoria_plotagem(pg["cobertura"], pg["colorido"], lim_linhas)
    base = PLOTAGEM[cat][chave] * (metros if chave == "ML" else 1)
    extra = ""
    if papel_90g:
        add = ADICIONAL_90G.get(chave, 0.0) * (metros if chave == "ML" else 1)
        base += add
        extra = " + 90g"
    return formato_label, f"{tam_txt} - {cat}{extra}{obs_a4}", base * qtd, None


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("🖨️ Pré-Impressão & Orçamento Gráfico")
st.caption("Envie PDFs de engenharia/arquitetura: o app lê o tamanho real da folha, mede a cobertura de tinta e orça pela sua tabela.")

with st.sidebar:
    st.header("⚙️ Orçamento")
    servico = st.radio("Serviço", ["Plotagem", "Cópia", "Digitalização"])
    papel_90g = False
    if servico == "Plotagem":
        papel_90g = st.radio("Papel", ["Sulfite 75g", "Sulfite 90g"]) == "Sulfite 90g"
    qtd = st.number_input("Quantidade de vias", min_value=1, value=1, step=1)
    a4_como_a3 = st.checkbox(
        "Cobrar A4 pelo preço do A3",
        value=False,
        help="A tabela não possui preço para A4. Se desmarcado, A4 fica com R$ 0,00 e um aviso.",
    )

    with st.expander("Ajustes avançados"):
        tol = st.slider("Tolerância de formato (mm)", 1, 10, 4)
        lim_linhas = st.slider("Até quantos % de tinta é 'Linhas'", 1, 40, 15)
        branco = st.slider("Limiar de branco (0–255)", 200, 255, 245,
                           help="Pixel com todos os canais ≥ este valor é considerado papel (0% de tinta).")
        croma = st.slider("Sensibilidade de cor", 5, 100, 30,
                          help="Diferença mínima entre canais RGB para o pixel contar como colorido.")
        frac_cor = st.slider("Área colorida mínima p/ cobrar Color (%)", 0.0, 5.0, 0.1, 0.05)
        max_px = st.select_slider("Resolução da análise (px no lado maior)",
                                  options=[1500, 2000, 3000, 4000, 6000], value=3000)

arquivos = st.file_uploader("Envie um ou mais PDFs", type=["pdf"], accept_multiple_files=True)

if not arquivos:
    st.info("⬆️ Faça o upload de um PDF para gerar o relatório.")
    st.stop()

total_geral, paginas_geral, avisos_geral = 0.0, 0, []

for arq in arquivos:
    st.divider()
    st.subheader(f"📄 {arq.name}")

    try:
        with st.spinner("Lendo metadados e calculando cobertura de tinta..."):
            paginas = analisar_pdf(arq.getvalue(), max_px, branco, croma, frac_cor)
    except Exception as e:  # PDF corrompido, protegido etc.
        st.error(f"Não foi possível analisar este arquivo: {e}")
        continue

    linhas, valores, formatos = [], [], []
    for pg in paginas:
        fmt_label, cat_label, valor, aviso = precificar(
            pg, servico, papel_90g, int(qtd), lim_linhas, tol, a4_como_a3
        )
        if aviso:
            avisos_geral.append(f"{arq.name} – pág {pg['pagina']}: {aviso}")
        valores.append(valor)
        formatos.append(fmt_label)
        linhas.append(
            {
                "Página": f"Pág {pg['pagina']}",
                "Dimensões (mm)": f"{pg['w_mm']:.0f} x {pg['h_mm']:.0f} mm",
                "Formato (A0, A1...)": fmt_label,
                "Cobertura de Tinta (%)": pct(pg["cobertura"]),
                "Cor": "Color" if pg["colorido"] else "PB",
                "Categoria de Cobrança": cat_label,
                "Valor (R$)": brl(valor),
            }
        )

    df = pd.DataFrame(linhas)
    st.dataframe(df, hide_index=True)

    subtotal = float(sum(valores))
    total_geral += subtotal
    paginas_geral += len(paginas)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de páginas analisadas", len(paginas))
    c2.metric("Valor Total do Orçamento", brl(subtotal))
    c3.metric("Serviço", f"{servico} × {int(qtd)}")

    resumo = (
        pd.DataFrame({"Formato": formatos, "Valor": valores})
        .groupby("Formato")
        .agg(Páginas=("Valor", "size"), Subtotal=("Valor", "sum"))
        .reset_index()
    )
    resumo["Subtotal"] = resumo["Subtotal"].map(brl)
    with st.expander("Resumo por formato"):
        st.dataframe(resumo, hide_index=True)

    md = (
        df_para_markdown(df)
        + f"\n\n**Resumo do Orçamento**\n- Total de páginas analisadas: {len(paginas)}"
        + f"\n- Valor Total do Orçamento: {brl(subtotal)}"
    )
    b1, b2 = st.columns(2)
    b1.download_button(
        "⬇️ Baixar CSV",
        df.to_csv(sep=";", index=False).encode("utf-8-sig"),
        file_name=f"orcamento_{arq.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
        key=f"csv_{arq.name}",
    )
    with b2.expander("Copiar em Markdown"):
        st.code(md, language="markdown")

if avisos_geral:
    st.warning("**Itens sem preço na tabela (valor R$ 0,00):**\n\n" + "\n".join(f"- {a}" for a in avisos_geral))

if len(arquivos) > 1:
    st.divider()
    st.subheader("🧾 Total Geral")
    g1, g2 = st.columns(2)
    g1.metric("Total de páginas analisadas", paginas_geral)
    g2.metric("Valor Total do Orçamento", brl(total_geral))

with st.expander("ℹ️ Como o cálculo é feito"):
    st.markdown(
        """
- **Dimensões:** vêm dos metadados da página (CropBox, limitada pela MediaBox, com rotação aplicada). Nada dentro do desenho é lido.
- **Formato:** compara os lados (em qualquer orientação) com A0–A4 usando a tolerância configurada; senão, *Fora de Padrão (Metro Linear)*.
- **Cobertura:** a página é renderizada em baixa resolução; pixels brancos = 0%, todo o resto (linhas, cores, preenchimentos) é somado.
- **Cor × PB:** a página é *Color* se a área de pixels coloridos passar do mínimo configurado.
- **Faixas (Plotagem):** Linhas (até o limite configurado) → Até 50% → 80% → 100%. PB acima de "Linhas" usa as linhas Color/PB da tabela.
- **Metro Linear:** cobrado pelo **lado maior** da folha, em metros. O adicional de 90g também é por metro.
- **Cópia** só tem preço para plantas de linhas; **Digitalização** é preço fixo por tamanho (A0–A2).
        """
    )
