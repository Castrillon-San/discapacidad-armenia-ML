# Categoría de discapacidad probable en Armenia (Quindío)

Proyecto integrador de analítica predictiva desarrollado con la metodología CRISP-DM. El modelo estima la categoría de discapacidad más probable de una persona certificada a partir de su perfil sociodemográfico y de aseguramiento en salud, como apoyo a la planeación de servicios por grupos de población.

- **Aplicación en línea:** [URL_DE_LA_APP](https://URL_DE_LA_APP.streamlit.app)
- **Cuaderno de Google Colab:** [enlace al cuaderno](URL_DEL_COLAB)

## Datos

Conjunto "Personas con discapacidad certificadas en Armenia", publicado en el portal Datos Abiertos Colombia (datos.gov.co). Contiene 3.181 registros de las vigencias 2021 a 2025.

## Problema y solución

| Elemento | Descripción |
|---|---|
| Tipo de problema | Clasificación multiclase supervisada |
| Variable objetivo | Categoría de discapacidad: física, intelectual, múltiple, psicosocial y sensorial (visual, auditiva y sordoceguera agrupadas) |
| Predictoras | Edad, año de certificación, género, comuna, EPS, régimen de salud y condición de víctima |
| Modelos evaluados | Regresión logística, KNN, árbol de decisión, SVM, Naive Bayes, votación, *bagging* y *boosting* |
| Validación | Partición estratificada 70/30 y validación cruzada de 5 pliegues sobre el entrenamiento |
| Métrica principal | F1 macro, por el desbalance entre clases |

El *pipeline* serializado incluye imputación, escalado, codificación *one-hot*, balanceo por sobremuestreo aleatorio y el clasificador seleccionado.

## Estructura del repositorio

```
├── app.py                       Aplicación Streamlit
├── modelo_discapacidad.joblib   Pipeline completo entrenado
├── metadatos_modelo.json        Opciones de las variables, métricas y versiones
├── requirements.txt             Dependencias con versiones fijas
├── .streamlit/config.toml       Tema visual de la aplicación
└── README.md
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitaciones

Las variables disponibles no incluyen información clínica, por lo que la capacidad predictiva es moderada. El modelo describe a la población certificada y no necesariamente a toda la población con discapacidad del municipio. Sus resultados no reemplazan la valoración del equipo de certificación ni deben usarse para decisiones sobre personas individuales.

## Autor

Reis. Proyecto integrador CRISP-DM, 2026.
