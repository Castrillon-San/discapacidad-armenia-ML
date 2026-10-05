"""
Aplicación Streamlit: estimación de la categoría de discapacidad probable
a partir del perfil sociodemográfico y de aseguramiento (Armenia, Quindío).

Archivos necesarios en la misma carpeta:
- modelo_discapacidad.joblib  (pipeline completo serializado desde Colab)
- metadatos_modelo.json       (opciones de las variables, métricas y versiones)
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

CARPETA = Path(__file__).parent

# Nombres legibles de las variables para el formulario
ETIQUETAS = {
    "edad": "Edad (años)",
    "vigencia": "Año de la certificación",
    "genero": "Género",
    "tipo_documento": "Tipo de documento",
    "comuna": "Comuna de residencia",
    "eps": "EPS",
    "regimen": "Régimen de salud",
    "condicion_victima": "Víctima del conflicto armado",
}

st.set_page_config(page_title="Categoría de discapacidad probable", page_icon="🧭", layout="centered")


# ---------------------------------------------------------------------------
# Carga del modelo y los metadatos (se cachean para no recargarlos en cada clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(CARPETA / "modelo_discapacidad.joblib")


@st.cache_data
def cargar_metadatos():
    with open(CARPETA / "metadatos_modelo.json", encoding="utf-8") as f:
        return json.load(f)


try:
    modelo = cargar_modelo()
    meta = cargar_metadatos()
except FileNotFoundError as e:
    st.error(f"No se encontró el archivo {Path(e.filename).name}. "
             "Súbelo al repositorio en la misma carpeta que app.py.")
    st.stop()

PREDICTORAS = meta["predictoras"]
CATEGORICAS = meta["variables_categoricas"]
EDAD_MIN, EDAD_MAX = meta["rango_edad"]
VIGENCIAS = sorted(set(meta["vigencias_entrenamiento"]) | {max(meta["vigencias_entrenamiento"]) + 1})
TIENE_PROBABILIDADES = hasattr(modelo, "predict_proba")


def predecir(datos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la clase predicha y, si el modelo lo permite, las probabilidades por clase."""
    datos = datos[PREDICTORAS].copy()
    datos["edad"] = pd.to_numeric(datos["edad"], errors="coerce").replace(999, np.nan)  # 999 = no reporta
    datos["vigencia"] = pd.to_numeric(datos["vigencia"], errors="coerce")
    resultado = pd.DataFrame({"categoria_estimada": modelo.predict(datos)}, index=datos.index)
    if TIENE_PROBABILIDADES:
        proba = pd.DataFrame(modelo.predict_proba(datos), columns=modelo.classes_, index=datos.index)
        resultado["probabilidad"] = proba.max(axis=1).round(3)
        resultado = pd.concat([resultado, proba.round(3).add_prefix("p_")], axis=1)
    return resultado


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Categoría de discapacidad probable")
st.write(
    "Estima la categoría de discapacidad más probable para un perfil de persona certificada en "
    "Armenia (Quindío), según su edad, lugar de residencia y aseguramiento en salud. "
    "Sirve para planear servicios por grupos de población."
)
st.caption(
    "El resultado es una estimación estadística con precisión moderada. "
    "No reemplaza la valoración del equipo de certificación ni debe usarse para decisiones sobre una persona."
)

tab_individual, tab_lote, tab_modelo = st.tabs(["Un perfil", "Varios perfiles (CSV)", "Sobre el modelo"])

# ---------------------------------------------------------------------------
# Pestaña 1: un perfil
# ---------------------------------------------------------------------------
with tab_individual:
    with st.form("perfil"):
        col1, col2 = st.columns(2)
        with col1:
            sin_edad = st.checkbox("Edad no reportada")
            edad = st.number_input(ETIQUETAS["edad"], min_value=EDAD_MIN, max_value=EDAD_MAX,
                                   value=35, step=1, disabled=sin_edad)
        with col2:
            vigencia = st.selectbox(ETIQUETAS["vigencia"], VIGENCIAS, index=len(VIGENCIAS) - 1)

        valores_cat = {}
        columnas = st.columns(2)
        for i, var in enumerate(CATEGORICAS):
            with columnas[i % 2]:
                valores_cat[var] = st.selectbox(ETIQUETAS.get(var, var), meta["opciones"][var])

        enviado = st.form_submit_button("Estimar categoría", type="primary", width="stretch")

    if enviado:
        registro = pd.DataFrame([{
            "edad": np.nan if sin_edad else edad,
            "vigencia": vigencia,
            **valores_cat,
        }])
        resultado = predecir(registro).iloc[0]

        st.subheader(f"Categoría estimada: {resultado['categoria_estimada']}")

        if TIENE_PROBABILIDADES:
            proba = pd.DataFrame({
                "Categoría": modelo.classes_,
                "Probabilidad": [resultado[f"p_{c}"] for c in modelo.classes_],
            })
            grafico = (
                alt.Chart(proba)
                .mark_bar()
                .encode(
                    x=alt.X("Probabilidad:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Categoría:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum["Categoría"] == resultado["categoria_estimada"],
                        alt.value("#2E6B5E"), alt.value("#B8C9C4")),
                    tooltip=["Categoría", alt.Tooltip("Probabilidad:Q", format=".1%")],
                )
                .properties(height=200)
            )
            st.altair_chart(grafico, width="stretch")

            segunda = proba.sort_values("Probabilidad", ascending=False).iloc[1]
            diferencia = resultado["probabilidad"] - segunda["Probabilidad"]
            if diferencia < 0.10:
                st.info(f"La estimación es poco concluyente: {segunda['Categoría']} tiene una probabilidad "
                        f"muy cercana ({segunda['Probabilidad']:.0%} frente a {resultado['probabilidad']:.0%}).")

# ---------------------------------------------------------------------------
# Pestaña 2: varios perfiles desde un CSV
# ---------------------------------------------------------------------------
with tab_lote:
    st.write("Sube un archivo CSV con una fila por persona y estas columnas:")
    st.code(", ".join(PREDICTORAS), language=None)

    plantilla = pd.DataFrame([{
        "edad": 35, "vigencia": VIGENCIAS[-1],
        **{v: meta["opciones"][v][0] for v in CATEGORICAS},
    }])[PREDICTORAS]
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8-sig"),
                       file_name="plantilla_perfiles.csv", mime="text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No fue posible leer el archivo como CSV separado por comas ({e}).")
            st.stop()

        faltantes = [c for c in PREDICTORAS if c not in datos.columns]
        if faltantes:
            st.error("Al archivo le faltan estas columnas: " + ", ".join(faltantes))
        else:
            salida = pd.concat([datos, predecir(datos)], axis=1)
            st.success(f"Se estimaron {len(salida)} perfiles.")
            st.dataframe(salida, width="stretch", hide_index=True)
            st.bar_chart(salida["categoria_estimada"].value_counts())
            st.download_button("Descargar resultados", salida.to_csv(index=False).encode("utf-8-sig"),
                               file_name="perfiles_estimados.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Pestaña 3: información del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    st.write(f"**Modelo seleccionado:** {meta['modelo']}")
    m = meta["metricas_prueba"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.2f}")
    c2.metric("Precision macro", f"{m['precision_macro']:.2f}")
    c3.metric("Recall macro", f"{m['recall_macro']:.2f}")
    c4.metric("F1 macro", f"{m['f1_macro']:.2f}")
    st.caption("Métricas calculadas sobre el 30 % de los datos reservado para prueba.")

    st.write("**Hiperparámetros ajustados**")
    st.json({k.replace("modelo__", ""): v for k, v in meta["hiperparametros"].items()})

    st.write("**Datos y metodología**")
    st.write(
        "Conjunto \"Personas con discapacidad certificadas en Armenia\" (Datos Abiertos Colombia), "
        f"vigencias {min(meta['vigencias_entrenamiento'])} a {max(meta['vigencias_entrenamiento'])}. "
        "El proyecto sigue la metodología CRISP-DM. Las categorías visual, auditiva y sordoceguera "
        "se agrupan como sensorial."
    )
    st.write("**Limitaciones**")
    st.write(
        "Las variables disponibles no incluyen información clínica, por lo que la capacidad predictiva es "
        "moderada. El modelo describe a la población certificada y puede no representar a quienes no han "
        "accedido a la certificación. Las estimaciones para años posteriores a los datos suponen que la "
        "tendencia reciente se mantiene."
    )
