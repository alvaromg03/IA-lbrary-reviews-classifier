# IA-lbrary-reviews-classifier

## Descripción

Este proyecto implementa un clasificador de sentimientos para reseñas de productos utilizando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático. El objetivo es predecir si una reseña es positiva o negativa a partir de su texto.

El flujo principal del proyecto es:
1. Limpieza y balanceo de datos desde una base de datos SQLite.
2. Entrenamiento de modelos de clasificación de texto (Naive Bayes, Regresión Logística o SVM).
3. Evaluación del modelo y guardado del pipeline entrenado.
4. Carga del modelo y predicción de sentimientos en nuevas reseñas.

## Estructura del proyecto

## Uso

### 1. Limpieza y preparación de datos

Ejecuta `data_cleaning.py` para extraer, limpiar y balancear los datos desde la base de datos SQLite y guardarlos en `data/def_data.csv`.

```sh
python [data_cleaning.py](http://_vscodecontentref_/6)

2. Entrenamiento y evaluación del modelo
Ejecuta App.py para entrenar el modelo, evaluarlo y guardar el pipeline entrenado. El script también realiza predicciones de ejemplo sobre nuevas reseñas.

python [App.py](http://_vscodecontentref_/8)

3. Predicción en nuevas reseñas
Puedes modificar la variable nuevas en App.py para predecir el sentimiento de cualquier texto.

Modelos soportados
Naive Bayes (naive_bayes)
Regresión Logística (logistic)
SVM (svm)
Puedes cambiar el modelo a entrenar modificando el parámetro "model" en el diccionario params de App.py.

Requisitos
Python 3.x
pandas
scikit-learn
joblib
Instala las dependencias con:

pip install -r [requirements.txt](http://_vscodecontentref_/9)

Notas
El dataset limpio se guarda en data/def_data.csv usando el separador ¬.
El modelo entrenado se guarda en la carpeta model/ como sentiment_model_pipeline.pkl.
El pipeline incluye vectorización TF-IDF y el clasificador seleccionado.