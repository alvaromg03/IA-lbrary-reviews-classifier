import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

class TextClassifier:
    def __init__(self):
        self.pipeline = None

    def train_model(self, data, trainParams):
        # Parámetros
        test_size = trainParams.get("test_size", 0.2)
        model_type = trainParams.get("model", "naive_bayes")

        # Separar características y etiquetas
        X = data["Text"]
        y = data["Sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Vectorizador
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

        # Selección del modelo
        if model_type == "naive_bayes":
            classifier = MultinomialNB()
        elif model_type == "logistic":
            classifier = LogisticRegression(max_iter=1000)
        elif model_type == "svm":
            classifier = LinearSVC()
        else:
            raise ValueError("Modelo no soportado")

        # Pipeline
        self.pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", classifier)
        ])

        # Entrenamiento
        self.pipeline.fit(X_train, y_train)

        # Predicción
        y_pred = self.pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        return cm, y_test, y_pred

    def save_model(self, modelName, save_path="."):

       os.makedirs(save_path, exist_ok=True)

       file_name = f"{modelName}_pipeline.pkl"
       full_path = os.path.join(save_path, file_name)

       joblib.dump(self.pipeline, full_path)
       return full_path

    def load_model(self, modelName, save_path="."):
       file_name = f"{modelName}_pipeline.pkl"
       full_path = os.path.join(save_path, file_name)

       self.pipeline = joblib.load(full_path)
       return full_path

    def test_model(self, data):
        if self.pipeline is None:
            raise ValueError("El modelo no está entrenado o cargado.")
        return self.pipeline.predict(data["Text"])