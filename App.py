import pandas as pd
from AIlibrary import TextClassifier
from sklearn.metrics import classification_report, accuracy_score
classifier = TextClassifier() # Create an instance of the class

csv_path4 = './data/def_data.csv'
csv_path5 = './model'

data = pd.read_csv(csv_path4, sep="¬", engine='python')

print("Datos cargados correctamente.")

params = {
    "model": "logistic",     # opciones: "naive_bayes", "logistic", "svm"
    "test_size": 0.2
}

print("Entrenando modelo...")
cm, y_test, y_pred = classifier.train_model(data, params)
print("Modelo entrenado correctamente.")


print("📊 Matriz de confusión:\n", cm)
print("\n📈 Métricas de clasificación:\n")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")


output_path = classifier.save_model("sentiment_model", save_path=csv_path5)
print(f"✅ Modelo guardado en: {output_path}")

classifier.load_model("sentiment_model", save_path = csv_path5)
print("📂 Modelo cargado.")


nuevas = pd.DataFrame({
    "Text": [
        "I love this product. It exceeded all my expectations!",
        "Worst thing I ever bought. Don’t recommend.",
        "It's okay, nothing special."
    ]
})

preds = classifier.test_model(nuevas)

print("\n🔍 Predicciones en nuevas reseñas:")
for txt, p in zip(nuevas["Text"], preds):
    print(f"\nRESEÑA: {txt}\n→ PREDICCIÓN: {p}")