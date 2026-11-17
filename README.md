# Sistema Inteligente de Detección y Clasificación de Reclamos Ciudadanos contra Movistar Perú

Aplicación Streamlit del curso de Procesamiento de Lenguaje Natural (Persona C).

**Apartados**
1. **Dashboard de reclamos**: indicadores, reclamos por categoría, distribución de sentimientos, sentimiento por categoría, 5 temas NMF y tabla filtrable.
2. **Analizar un reclamo**: categoría predicha (TF-IDF + Regresión Logística), sentimiento (pysentimiento/RoBERTuito) y tópico NMF más relacionado.

## Estructura
```
movistar-reclamos-app/
├── app.py                  # aplicación principal
├── modelos.py              # carga de modelos y predicciones
├── requirements.txt
├── .streamlit/config.toml  # tema claro (fondo blanco)
├── data/
│   ├── reclamos_con_sentimiento.csv
│   └── comparacion_modelos.csv
└── models/
    ├── clasificador.pkl
    ├── vectorizador_tfidf.pkl
    ├── tfidf_topicos.pkl
    └── nmf_topicos.pkl
```

## Ejecución local
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Se recomienda Python 3.12. La primera vez que se analiza un reclamo se descarga el modelo
RoBERTuito (~500 MB) desde Hugging Face; requiere internet.

## Notas importantes
- Los `.pkl` se entrenaron con **scikit-learn 1.6.1**; por eso está fijado en `requirements.txt`. No lo cambies.
- Los modelos se entrenaron sobre el texto crudo (`texto`), no sobre el lematizado.
- La base tiene 146 reclamos (132 sintéticos y 14 de Reddit pendientes de verificación). Los resultados son de práctica.
- No se incluye evolución temporal porque solo 14 de 146 reclamos tienen fecha.

## Publicar en GitHub
1. Crea una cuenta en https://github.com y pulsa **New repository**.
2. Nombre: `movistar-reclamos-app`, visibilidad **Public**, sin README ni .gitignore (ya existen). Pulsa **Create repository**.
3. En la carpeta del proyecto:
```bash
git init
git add .
git commit -m "App Streamlit reclamos Movistar"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/movistar-reclamos-app.git
git push -u origin main
```
(Sin Git: en la página del repo, **Add file → Upload files**, arrastra el contenido de la carpeta, incluidas `data/`, `models/` y `.streamlit/`, y pulsa **Commit changes**.)

## Desplegar en Streamlit Community Cloud
1. Entra a https://share.streamlit.io e inicia sesión con GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository: `TU_USUARIO/movistar-reclamos-app` · Branch: `main` · Main file path: `app.py`.
4. **Advanced settings → Python version: 3.12**.
5. **Deploy**. La primera construcción tarda varios minutos (instala torch y transformers).
6. Copia el enlace `https://<nombre>.streamlit.app` para tu exposición.

## Solución de problemas
| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError` | Falta una dependencia | Añádela a `requirements.txt` y haz push |
| Aviso `InconsistentVersionWarning` o predicciones raras | scikit-learn distinto a 1.6.1 | Verifica `scikit-learn==1.6.1` |
| App se reinicia / "Resource limits" al analizar | RoBERTuito + torch exceden ~1 GB de RAM | Ver abajo |
| Fallo al construir por torch | Versión incompatible con Python | Usa Python 3.12 en Advanced settings |
| `FileNotFoundError` en data/ o models/ | Carpetas no subidas | Verifica que existan en GitHub con esos nombres exactos |
| Cambios no se ven | Caché | Menú ⋮ → **Reboot app** |

**Límite de memoria:** el plan gratuito ofrece ~1 GB de RAM. Si el modelo de sentimientos no cabe,
la app muestra un mensaje de error en esa columna (sin inventar resultados) y sigue prediciendo categoría y tópico.
En ese caso ejecuta la demo localmente o despliega en Hugging Face Spaces (16 GB gratis).
