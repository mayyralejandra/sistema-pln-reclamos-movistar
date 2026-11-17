"""Carga de modelos entrenados por el equipo y funciones de predicción.

No se reentrena nada: solo se usan los .pkl entregados.
Los modelos de clasificación y tópicos se entrenaron sobre la columna `texto`
(en el notebook, texto_limpio = texto), por eso aquí se predice sobre el texto crudo.
"""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

# Interpretación breve de cada tópico NMF (basada en sus palabras clave
# y en las categorías que más se concentran en cada uno).
TOPIC_LABELS = {
    1: "Planes, migración y cobros inesperados",
    2: "Portabilidad y cambio de operador",
    3: "Internet y cable: fallas y soporte técnico",
    4: "Paquetes, recargas y cargos no autorizados",
    5: "Bajas solicitadas y cobros que continúan",
}

SENT_MAP = {"POS": "Positivo", "NEU": "Neutro", "NEG": "Negativo"}


@st.cache_resource(show_spinner=False)
def cargar_clasificador():
    """Devuelve (vectorizador, clasificador) o lanza excepción si falla."""
    vec = joblib.load(MODELS_DIR / "vectorizador_tfidf.pkl")
    clf = joblib.load(MODELS_DIR / "clasificador.pkl")
    return vec, clf


@st.cache_resource(show_spinner=False)
def cargar_topicos():
    """Devuelve (tfidf_topicos, nmf) o lanza excepción si falla."""
    tfidf = joblib.load(MODELS_DIR / "tfidf_topicos.pkl")
    nmf = joblib.load(MODELS_DIR / "nmf_topicos.pkl")
    return tfidf, nmf


@st.cache_resource(show_spinner=False)
def cargar_sentimiento():
    """Carga RoBERTuito (pysentimiento). La primera vez descarga el modelo."""
    from pysentimiento import create_analyzer
    return create_analyzer(task="sentiment", lang="es")


def top_palabras_topicos(n=10):
    """Lista de las n palabras con más peso de cada tópico (orden: tópico 1..5)."""
    tfidf, nmf = cargar_topicos()
    terminos = tfidf.get_feature_names_out()
    return [[terminos[j] for j in comp.argsort()[::-1][:n]] for comp in nmf.components_]


def predecir_categoria(texto, top_k=3):
    vec, clf = cargar_clasificador()
    proba = clf.predict_proba(vec.transform([texto]))[0]
    orden = np.argsort(proba)[::-1][:top_k]
    return [(clf.classes_[i], float(proba[i])) for i in orden]


def predecir_topico(texto):
    """Devuelve (numero_topico, pesos_normalizados) o (None, None) si el texto
    no contiene ningún término del vocabulario de tópicos."""
    tfidf, nmf = cargar_topicos()
    w = nmf.transform(tfidf.transform([texto]))[0]
    if w.sum() == 0:
        return None, None
    return int(w.argmax()) + 1, w / w.sum()


def predecir_sentimiento(texto):
    """Devuelve (etiqueta, {etiqueta: probabilidad})."""
    analyzer = cargar_sentimiento()
    res = analyzer.predict(texto)
    probas = {SENT_MAP.get(k, k): float(v) for k, v in res.probas.items()}
    return SENT_MAP.get(res.output, res.output), probas
