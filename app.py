@app.route("/")
def home():
    return "API de predicción funcionando correctamente 🚀" 

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Ruta base donde está el app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo y vectorizador con ruta absoluta
modelo = joblib.load(os.path.join(BASE_DIR, "modelo_svc2.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer_tfidf_svc2.pkl"))

# ==============================
# 1. Inicializar la aplicación
# ==============================
app = Flask(__name__)

# ==============================
# 3. Ruta de prueba
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'texto' not in data:
            return jsonify({"error": "Falta el campo 'texto' en el JSON"}), 400

        texto = data['texto']
        X = vectorizer.transform([texto])
        pred = modelo.predict(X)[0]

        # Mapear predicción de texto
        if pred.lower() == "negative":
            sentimiento = "NEGATIVO 😞"
        elif pred.lower() == "neutral":
            sentimiento = "NEUTRO 😐"
        elif pred.lower() == "positive":
            sentimiento = "POSITIVO 😊"
        else:
            sentimiento = "DESCONOCIDO"

        return jsonify({
            "texto": texto,
            "sentimiento": sentimiento,
            "valor": pred
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 5. Ejecutar servidor
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port=8000)
