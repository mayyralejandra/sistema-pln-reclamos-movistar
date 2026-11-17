"""Sistema Inteligente de Detección y Clasificación de Reclamos Ciudadanos
contra Movistar Perú – Aplicación Streamlit.

Apartados: 1) Análisis de reclamos (carga CSV, preprocesamiento, exploración)
           2) Analizar un reclamo (clasificación, sentimiento, tópico)
"""
import io
import re
import textwrap
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import modelos as m

BASE = Path(__file__).parent
CSV_DEFECTO = BASE / "data" / "reclamos_con_sentimiento.csv"
AZUL, AZUL_OSC, VERDE = "#019DF4", "#0B2739", "#5CB615"
COLORES_SENT = {"Negativo": "#E5484D", "Neutro": "#9CA3AF", "Positivo": VERDE}
NINGUNA = "(ninguna)"

st.set_page_config(page_title="Reclamos Movistar Perú", page_icon="📡", layout="wide")
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    [data-testid="stSidebar"] {{ background-color: #F3F7FB; }}
    .tarjeta {{ border: 1px solid #E3E8EF; border-radius: 12px; padding: 14px 16px; background: #FFFFFF; height: 100%; }}
    .tarjeta h4 {{ margin: 0 0 4px 0; color: {AZUL_OSC}; font-size: 1rem; }}
    .tarjeta p {{ margin: 0; color: #4B5563; font-size: 0.88rem; }}
    .resultado {{ border-left: 6px solid {AZUL}; background: #F3F7FB; border-radius: 8px; padding: 14px 18px; }}
    .resultado small {{ color: #6B7280; text-transform: uppercase; letter-spacing: .04em; }}
    .resultado h3 {{ margin: 2px 0 0 0; color: {AZUL_OSC}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def envolver(txt, ancho=38):
    return "<br>".join(textwrap.wrap(str(txt), ancho))


# ======================================================== Preprocesamiento
# Funciones tomadas de Persona_A_Continuacion_146_Reclamos_Colab.ipynb (sección 6).
NEGACIONES = {"no", "nunca", "sin", "ni", "tampoco", "jamás"}


def limpiar_basico(texto):
    """NFKC + minúsculas + expresiones regulares (URL, correo, menciones, números, símbolos)."""
    texto = unicodedata.normalize("NFKC", str(texto)).lower()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " ", texto)
    texto = re.sub(r"(?<!\w)@[\w_]+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


@st.cache_resource(show_spinner=False)
def cargar_spacy():
    """spaCy + es_core_news_sm (el mismo del notebook de la Persona A).
    Devuelve (nlp, stopwords) o (None, mensaje_de_error)."""
    try:
        import spacy
        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
        if "lemmatizer" not in nlp.pipe_names:
            return None, "El modelo de spaCy no incluye lematizador."
        return nlp, set(nlp.Defaults.stop_words) - NEGACIONES
    except Exception as e:  # spaCy no instalado, modelo ausente, etc.
        return None, f"{type(e).__name__}: {e}"


def preprocesar(texto, nlp, stopwords):
    """Devuelve las etapas del preprocesamiento para un texto."""
    limpio = limpiar_basico(texto)
    etapas = {"original": str(texto), "limpio": limpio}
    if nlp is None:
        etapas["tokens"] = limpio.split()  # tokenización simple por espacios
        etapas["sin_stopwords"] = None
        etapas["lematizado"] = None
        return etapas
    doc = nlp(limpio)
    etapas["tokens"] = [t.text for t in doc]
    etapas["sin_stopwords"] = [t.text for t in doc
                               if t.text.isalpha() and len(t.text) >= 2 and t.text not in stopwords]
    lemas = []
    for t in doc:  # misma lógica del notebook: se filtra sobre el lema
        lema = t.lemma_.lower().strip()
        if lema and lema.isalpha() and len(lema) >= 2 and lema not in stopwords:
            lemas.append(lema)
    etapas["lematizado"] = lemas
    return etapas


# ================================================================== Datos
@st.cache_data(show_spinner=False)
def leer_csv(contenido: bytes):
    """Lee un CSV probando codificaciones y separadores. Devuelve (df, error)."""
    if not contenido or not contenido.strip():
        return None, "El archivo está vacío."
    ultimo = None
    for enc in ("utf-8-sig", "latin-1"):
        validos = {}
        for sep in (",", ";", "\t", "|"):
            try:
                validos[sep] = pd.read_csv(io.BytesIO(contenido), sep=sep, encoding=enc)
            except Exception as e:
                ultimo = e
        # Preferir el separador que produce varias columnas; si ninguno, aceptar una sola columna con ","
        elegido = next((d for d in validos.values() if d.shape[1] > 1), validos.get(","))
        if elegido is not None:
            if elegido.shape[1] == 0 or elegido.empty:
                return None, "El archivo no contiene filas de datos."
            elegido.columns = [str(c).strip() for c in elegido.columns]
            return elegido, None
    return None, f"No se pudo leer el archivo como CSV ({type(ultimo).__name__}: {ultimo})."


def adivinar_columna(df, candidatas):
    low = {c.lower(): c for c in df.columns}
    for c in candidatas:
        if c in low:
            return low[c]
    return None


def adivinar_con_datos(df, candidatas):
    """Como adivinar_columna, pero ignora columnas completamente vacías."""
    c = adivinar_columna(df, candidatas)
    return c if c is not None and df[c].notna().any() else None


def adivinar_texto(df):
    c = adivinar_columna(df, ["texto", "reclamo", "reclamos", "comentario", "descripcion", "descripción", "text"])
    if c:
        return c
    obj = [x for x in df.columns if df[x].dtype == object or str(df[x].dtype).startswith("str")]
    if obj:
        return max(obj, key=lambda x: df[x].astype(str).str.len().mean())
    return df.columns[0]


def validar_textos(serie):
    """Devuelve (serie_limpia, mensaje_error)."""
    s = serie.fillna("").astype(str).str.strip()
    util = s.str.contains(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{3,}", regex=True)
    if util.sum() == 0:
        return None, "La columna elegida no contiene texto utilizable (se necesitan palabras con letras)."
    return s, None


@st.cache_data(show_spinner=False)
def predecir_lote(textos: tuple):
    """Categoría (con probabilidad) y tópico dominante para cada texto (modelos del equipo)."""
    vec, clf = m.cargar_clasificador()
    proba = clf.predict_proba(vec.transform(list(textos)))
    cats = clf.classes_[proba.argmax(axis=1)]
    tfidf, nmf = m.cargar_topicos()
    W = nmf.transform(tfidf.transform(list(textos)))
    top = [int(w.argmax()) + 1 if w.sum() > 0 else None for w in W]
    return list(cats), list(proba.max(axis=1)), top


def calcular_sentimientos(textos, barra):
    """RoBERTuito por lotes. Guarda resultados en session_state (no repite trabajo)."""
    cache = st.session_state.setdefault("sent_pred", {})
    pendientes = [t for t in dict.fromkeys(textos) if t not in cache]
    analyzer = m.cargar_sentimiento()
    for i in range(0, len(pendientes), 16):
        lote = pendientes[i:i + 16]
        res = analyzer.predict(lote)
        for t, r in zip(lote, res):
            cache[t] = m.SENT_MAP.get(r.output, r.output)
        barra.progress(min(1.0, (i + len(lote)) / max(1, len(pendientes))))


# ============================================================== Sidebar
st.sidebar.title("📡 Reclamos Movistar")
pagina = st.sidebar.radio("Navegación", ["Análisis de reclamos", "Analizar un reclamo"])
st.sidebar.caption("Curso de Procesamiento de Lenguaje Natural")

st.title("Sistema Inteligente de Detección y Clasificación de Reclamos Ciudadanos contra Movistar Perú")
st.write("Carga reclamos en CSV, observa su preprocesamiento y explora los resultados de tres técnicas de PLN: "
         "clasificación de texto (TF-IDF), análisis de sentimientos (RoBERTuito) y modelado de tópicos (NMF).")
st.divider()

# ====================================================== APARTADO 1
if pagina == "Análisis de reclamos":

    # ---------- A. Carga de archivos
    st.header("A. Carga de datos")
    archivo = st.file_uploader("Carga un archivo CSV con reclamos (opcional)", type=["csv"],
                               help="Si no cargas nada se usa el CSV original del proyecto.")
    if archivo is not None:
        df_raw, err = leer_csv(archivo.getvalue())
        origen = archivo.name
        if err:
            st.error(f"{err} Se muestra el CSV original del proyecto mientras tanto.")
            archivo = None
    if archivo is None:
        try:
            df_raw, err = leer_csv(CSV_DEFECTO.read_bytes())
        except Exception as e:
            df_raw, err = None, str(e)
        origen = "reclamos_con_sentimiento.csv (CSV original del proyecto)"
        if err:
            st.error(f"No se pudo cargar el CSV original: {err}")
            st.stop()
    st.success(f"Archivo en uso: **{origen}** · {len(df_raw)} filas · {df_raw.shape[1]} columnas")

    with st.expander("Vista previa y columnas disponibles", expanded=True):
        st.dataframe(df_raw.head(10), width="stretch", hide_index=True)
        st.caption("Columnas: " + ", ".join(f"`{c}`" for c in df_raw.columns))

    cols = list(df_raw.columns)
    guess_t = adivinar_texto(df_raw)
    guess_c = adivinar_con_datos(df_raw, ["categoria", "categoría", "materia"])
    guess_s = adivinar_con_datos(df_raw, ["sentimiento", "sentiment"])
    s1, s2, s3 = st.columns(3)
    col_texto = s1.selectbox("Columna con los reclamos", cols, index=cols.index(guess_t))
    col_cat = s2.selectbox("Columna de categoría (si existe)", [NINGUNA] + cols,
                           index=(cols.index(guess_c) + 1) if guess_c else 0)
    col_sent = s3.selectbox("Columna de sentimiento (si existe)", [NINGUNA] + cols,
                            index=(cols.index(guess_s) + 1) if guess_s else 0)

    textos_ok, err = validar_textos(df_raw[col_texto])
    if err:
        st.error(err + " Elige otra columna o carga otro archivo.")
        st.stop()
    validos = textos_ok.str.contains(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{3,}", regex=True)
    if (~validos).any():
        st.warning(f"{int((~validos).sum())} filas sin texto utilizable fueron excluidas del análisis.")

    datos = pd.DataFrame({"texto": textos_ok[validos]})
    datos.insert(0, "fila", datos.index + 1)
    datos = datos.reset_index(drop=True)
    idx = textos_ok[validos].index

    # Categoría
    if col_cat != NINGUNA:
        datos["categoria"] = df_raw.loc[idx, col_cat].astype(str).where(df_raw.loc[idx, col_cat].notna()).values
        nota_cat = "Categoría: columna del archivo."
    else:
        try:
            with st.spinner("Clasificando con TF-IDF + Regresión Logística…"):
                cats, confs, tops = predecir_lote(tuple(datos["texto"]))
            datos["categoria"] = cats
            datos["confianza"] = [round(c, 3) for c in confs]
            nota_cat = "⚠️ El archivo no tiene categorías: se muestran **predicciones del clasificador**, no etiquetas reales."
        except Exception as e:
            st.error(f"No se pudo ejecutar el clasificador: {e}")
            nota_cat = "Categorías no disponibles."
    # Tópico (siempre con el modelo NMF)
    try:
        _, _, tops = predecir_lote(tuple(datos["texto"]))
        datos["tópico"] = tops
    except Exception as e:
        st.error(f"No se pudo ejecutar el modelo de tópicos: {e}")
    # Sentimiento
    if col_sent != NINGUNA:
        datos["sentimiento"] = df_raw.loc[idx, col_sent].astype(str).where(df_raw.loc[idx, col_sent].notna()).values
        nota_sent = "Sentimiento: columna del archivo."
        faltan_sent = False
    else:
        pred = st.session_state.get("sent_pred", {})
        datos["sentimiento"] = datos["texto"].map(pred)
        faltan_sent = datos["sentimiento"].isna().any()
        nota_sent = "⚠️ El archivo no tiene sentimientos: se usan **predicciones de RoBERTuito** cuando las calculas."
    st.caption(nota_cat + "  " + nota_sent)

    if col_sent == NINGUNA and faltan_sent:
        n_falta = int(datos["sentimiento"].isna().sum())
        st.info(f"Sentimientos no disponibles para {n_falta} reclamos. Puedes calcularlos con el modelo "
                "pysentimiento/RoBERTuito (la primera vez descarga el modelo y puede tardar).")
        if st.button(f"Calcular sentimientos ({n_falta} reclamos)"):
            try:
                barra = st.progress(0.0, text="Analizando sentimientos…")
                calcular_sentimientos(list(datos["texto"]), barra)
                barra.empty()
                st.rerun()
            except Exception as e:
                st.error("El modelo de sentimientos no está disponible en este entorno; no se generan "
                         f"predicciones. Detalle: {type(e).__name__}: {e}")

    st.divider()

    # ---------- B. Preprocesamiento
    st.header("B. Preprocesamiento textual")
    nlp, stopwords = cargar_spacy()
    if nlp is None:
        st.warning("spaCy / es_core_news_sm no está disponible, por lo que **no se pueden mostrar tokenización "
                   "con spaCy, stopwords ni lematización**. Solo se muestran limpieza con expresiones regulares "
                   "y tokenización simple. Para habilitarlo: `pip install spacy` y "
                   "`python -m spacy download es_core_news_sm`. Detalle: " + str(stopwords))
    else:
        st.caption("Pipeline del equipo (Persona A): normalización Unicode NFKC, minúsculas y expresiones regulares "
                   "→ tokenización → eliminación de stopwords (se conservan las negaciones: "
                   + ", ".join(sorted(NEGACIONES)) + ") → lematización con spaCy. No se implementó stemming.")

    if len(datos) > 1:
        n_ej = st.slider("Cantidad de reclamos a mostrar", 1, min(10, len(datos)), min(5, len(datos)))
    else:
        n_ej = 1
    muestra = datos["texto"].head(n_ej)
    filas = []
    for t in muestra:
        e = preprocesar(t, nlp, stopwords)
        filas.append({
            "Texto original": e["original"],
            "Texto limpio (regex + minúsculas)": e["limpio"],
            "Tokens": " | ".join(e["tokens"]),
            "Sin stopwords": " | ".join(e["sin_stopwords"]) if e["sin_stopwords"] is not None else "—",
            "Normalizado / lematizado": " ".join(e["lematizado"]) if e["lematizado"] is not None else "—",
        })
    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)

    st.markdown("**Probar con un reclamo (elige uno de los cargados)**")
    sel = st.selectbox("Reclamo", range(len(datos)), format_func=lambda i: f"{datos.loc[i, 'fila']}. {datos.loc[i, 'texto'][:90]}")
    e = preprocesar(datos.loc[sel, "texto"], nlp, stopwords)
    b1, b2, b3 = st.columns(3)
    b1.markdown("**1. Original**"); b1.write(e["original"])
    b2.markdown("**2. Limpio** (URLs, correos, menciones, números y símbolos eliminados)"); b2.write(e["limpio"])
    if e["lematizado"] is not None:
        b3.markdown("**3. Normalizado** (sin stopwords, lematizado)"); b3.write(" ".join(e["lematizado"]))
        quitadas = len(e["tokens"]) - len(e["lematizado"])
        st.caption(f"Tokens: {len(e['tokens'])} → {len(e['lematizado'])} tras eliminar stopwords y lematizar "
                   f"({quitadas} descartados).")
    else:
        b3.markdown("**3. Normalizado**"); b3.info("Requiere spaCy.")
    st.caption("Nota: el clasificador y el modelo de tópicos del equipo se entrenaron sobre el texto original, "
               "por lo que este preprocesamiento se muestra con fines demostrativos y no alimenta esos modelos.")

    st.divider()

    # ---------- C. Análisis exploratorio
    st.header("C. Análisis exploratorio")
    vc_sent = datos["sentimiento"].dropna().value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reclamos analizados", len(datos))
    k2.metric("Categorías", datos["categoria"].nunique() if "categoria" in datos else "—")
    if len(vc_sent):
        k3.metric(f"Sentimiento {vc_sent.index[0].lower()}", f"{vc_sent.iloc[0]} ({vc_sent.iloc[0] / vc_sent.sum():.0%})")
    else:
        k3.metric("Sentimientos", "No calculados")
    k4.metric("Palabras por reclamo (prom.)", f"{datos['texto'].str.split().str.len().mean():.1f}")

    g1, g2 = st.columns([3, 2])
    with g1:
        etiqueta_cat = "Reclamos por categoría" + ("" if col_cat != NINGUNA else " (predichas por el modelo)")
        st.subheader(etiqueta_cat)
        if "categoria" in datos and datos["categoria"].notna().any():
            cat = datos["categoria"].value_counts().sort_values().reset_index()
            cat.columns = ["categoria", "reclamos"]
            cat["etiqueta"] = cat["categoria"].map(envolver)
            fig = px.bar(cat, x="reclamos", y="etiqueta", orientation="h", text="reclamos",
                         color_discrete_sequence=[AZUL], hover_name="categoria")
            fig.update_traces(hovertemplate="%{hovertext}<br>Reclamos: %{x}<extra></extra>")
            fig.update_layout(height=max(380, 40 * len(cat)), yaxis_title=None, xaxis_title="Cantidad de reclamos",
                              plot_bgcolor="white", margin=dict(l=0, r=10, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Categorías no disponibles.")
    with g2:
        st.subheader("Distribución de sentimientos")
        if len(vc_sent):
            sent = vc_sent.reset_index(); sent.columns = ["sentimiento", "reclamos"]
            fig = px.pie(sent, names="sentimiento", values="reclamos", hole=0.5,
                         color="sentimiento", color_discrete_map=COLORES_SENT)
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Sentimientos no disponibles: el archivo no los incluye. Usa el botón «Calcular sentimientos» de la sección A.")

    st.subheader("Relación entre categoría y sentimiento")
    if "categoria" in datos and len(vc_sent):
        cs = datos.dropna(subset=["categoria", "sentimiento"]).groupby(["categoria", "sentimiento"]).size().reset_index(name="reclamos")
        cs["etiqueta"] = cs["categoria"].map(lambda t: envolver(t, 45))
        fig = px.bar(cs, x="reclamos", y="etiqueta", color="sentimiento", orientation="h",
                     color_discrete_map=COLORES_SENT, hover_name="categoria")
        fig.update_layout(height=max(420, 45 * cs["categoria"].nunique()), yaxis_title=None,
                          xaxis_title="Cantidad de reclamos", plot_bgcolor="white", legend_title_text="",
                          margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Se necesitan categorías y sentimientos para este gráfico.")

    # Evolución temporal (solo si hay fechas suficientes)
    col_fecha = adivinar_columna(df_raw, ["fecha", "date", "fecha_publicacion", "fecha_publicación"])
    if col_fecha:
        fechas = pd.to_datetime(df_raw.loc[idx, col_fecha], errors="coerce").dropna()
        if len(fechas) >= 30 and fechas.dt.to_period("M").nunique() >= 3:
            st.subheader("Evolución temporal de los reclamos")
            ev = fechas.dt.to_period("M").dt.to_timestamp().value_counts().sort_index().reset_index()
            ev.columns = ["mes", "reclamos"]
            fig = px.line(ev, x="mes", y="reclamos", markers=True, color_discrete_sequence=[AZUL])
            fig.update_layout(plot_bgcolor="white", height=320, margin=dict(l=0, r=10, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption(f"Evolución temporal no incluida: solo {len(fechas)} reclamos tienen fecha válida "
                       "(se requieren al menos 30 en 3 meses distintos).")

    # Tópicos NMF
    st.subheader("Modelado de tópicos (NMF)")
    try:
        palabras = m.top_palabras_topicos(10)
        conteo = datos["tópico"].value_counts() if "tópico" in datos else pd.Series(dtype=int)
        for fila in (range(1, 4), range(4, 6)):
            cs_ = st.columns(3)
            for i, col in zip(fila, cs_):
                col.markdown(
                    f"<div class='tarjeta'><h4>Tópico {i}</h4><p><b>{m.TOPIC_LABELS[i]}</b></p>"
                    f"<p style='margin-top:6px'>{', '.join(palabras[i - 1])}</p>"
                    f"<p style='margin-top:6px; color:{AZUL}'><b>{int(conteo.get(i, 0))} reclamos en el archivo</b></p></div>",
                    unsafe_allow_html=True)
        if len(conteo):
            d = pd.DataFrame({"tópico": [f"T{i}" for i in range(1, 6)], "reclamos": [int(conteo.get(i, 0)) for i in range(1, 6)]})
            fig = px.bar(d, x="tópico", y="reclamos", text="reclamos", color_discrete_sequence=[AZUL])
            fig.update_layout(height=280, plot_bgcolor="white", xaxis_title=None, margin=dict(l=0, r=10, t=20, b=0))
            st.plotly_chart(fig, width="stretch")
        sin_top = int(datos["tópico"].isna().sum()) if "tópico" in datos else 0
        if sin_top:
            st.caption(f"{sin_top} reclamos no contienen términos del vocabulario de tópicos y no se asignaron.")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo de tópicos: {e}")

    # Tabla
    st.subheader("Tabla de reclamos")
    f1, f2, f3 = st.columns([2, 1, 2])
    cat_sel = f1.selectbox("Categoría", ["Todas"] + (sorted(datos["categoria"].dropna().unique()) if "categoria" in datos else []))
    sen_sel = f2.selectbox("Sentimiento", ["Todos"] + sorted(datos["sentimiento"].dropna().unique()))
    busca = f3.text_input("Buscar en el texto")
    t = datos
    if cat_sel != "Todas":
        t = t[t["categoria"] == cat_sel]
    if sen_sel != "Todos":
        t = t[t["sentimiento"] == sen_sel]
    if busca:
        t = t[t["texto"].str.contains(busca, case=False, na=False, regex=False)]
    mostrar = [c for c in ["fila", "texto", "categoria", "confianza", "sentimiento", "tópico"] if c in t.columns]
    st.caption(f"{len(t)} de {len(datos)} reclamos")
    st.dataframe(t[mostrar], width="stretch", hide_index=True, height=380,
                 column_config={"texto": st.column_config.TextColumn("texto", width="large")})

# ====================================================== APARTADO 2
else:
    st.header("Analizar un reclamo")
    texto = st.text_area("Escribe el reclamo", height=150,
                         placeholder="Ej.: Me cobraron un paquete de datos que nunca pedí y no me lo quieren anular.")
    if st.button("Analizar reclamo", type="primary"):
        if len(texto.strip()) < 10:
            st.warning("Escribe un reclamo de al menos 10 caracteres.")
            st.stop()
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("##### Clasificación de texto")
            try:
                top = m.predecir_categoria(texto)
                st.markdown(f"<div class='resultado'><small>Categoría predicha</small><h3>{top[0][0]}</h3></div>",
                            unsafe_allow_html=True)
                st.caption(f"Probabilidad del clasificador: {top[0][1]:.0%}")
                d = pd.DataFrame(top, columns=["categoria", "probabilidad"])
                d["etiqueta"] = d["categoria"].map(lambda x: envolver(x, 30))
                fig = px.bar(d[::-1], x="probabilidad", y="etiqueta", orientation="h",
                             text=d[::-1]["probabilidad"].map("{:.0%}".format), color_discrete_sequence=[AZUL])
                fig.update_layout(height=260, yaxis_title=None, xaxis_title=None, plot_bgcolor="white",
                                  xaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=0, r=10, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Clasificador no disponible: {e}")

        with r2:
            st.markdown("##### Análisis de sentimientos")
            try:
                with st.spinner("Cargando modelo de sentimientos (RoBERTuito)…"):
                    etiqueta, probas = m.predecir_sentimiento(texto)
                st.markdown(f"<div class='resultado'><small>Sentimiento identificado</small><h3>{etiqueta}</h3></div>",
                            unsafe_allow_html=True)
                d = pd.DataFrame({"sentimiento": list(probas), "probabilidad": list(probas.values())})
                fig = px.bar(d, x="sentimiento", y="probabilidad", color="sentimiento", color_discrete_map=COLORES_SENT,
                             text=d["probabilidad"].map("{:.0%}".format))
                fig.update_layout(height=260, showlegend=False, xaxis_title=None, yaxis_title=None, plot_bgcolor="white",
                                  yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=0, r=10, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error("El modelo de sentimientos (pysentimiento/RoBERTuito) no está disponible en este entorno; "
                         f"no se muestra predicción. Detalle: {type(e).__name__}: {e}")

        with r3:
            st.markdown("##### Modelado de tópicos")
            try:
                num, pesos = m.predecir_topico(texto)
                if num is None:
                    st.info("El texto no contiene términos del vocabulario de tópicos; no se puede asignar un tópico.")
                else:
                    st.markdown(f"<div class='resultado'><small>Tópico relacionado</small>"
                                f"<h3>Tópico {num}: {m.TOPIC_LABELS[num]}</h3></div>", unsafe_allow_html=True)
                    st.caption("Palabras principales: " + ", ".join(m.top_palabras_topicos(10)[num - 1]))
                    d = pd.DataFrame({"tópico": [f"T{i}" for i in range(1, 6)], "peso": pesos})
                    fig = px.bar(d, x="tópico", y="peso", color_discrete_sequence=[AZUL], text=d["peso"].map("{:.0%}".format))
                    fig.update_layout(height=260, xaxis_title=None, yaxis_title=None, plot_bgcolor="white",
                                      yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=0, r=10, t=10, b=0))
                    st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Modelo de tópicos no disponible: {e}")

        try:
            met = pd.read_csv(BASE / "data" / "comparacion_modelos.csv", index_col=0)
            st.caption(f"Nota: el clasificador (Regresión Logística) obtuvo accuracy {met.loc['Regresión Logística', 'acc']:.3f} "
                       f"y F1 ponderado {met.loc['Regresión Logística', 'f1']:.3f} sobre 30 reclamos de prueba; "
                       "es una base pequeña, por lo que las predicciones son orientativas.")
        except Exception:
            pass
