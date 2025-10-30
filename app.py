import os
import joblib
from flask import Flask, request, jsonify

# ==============================
# 1. Inicializar la aplicación
# ==============================
app = Flask(__name__)

# ==============================
# 2. Cargar modelo y vectorizador
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    modelo = joblib.load(os.path.join(BASE_DIR, "modelo_svc2.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer_tfidf_svc2.pkl"))
    print("✅ Modelo y vectorizador cargados correctamente.")
except Exception as e:
    print("❌ Error cargando modelo/vectorizador:", e)

# ==============================
# 3. Ruta raíz de prueba
# ==============================
@app.route("/")
def home():
    return "API de predicción funcionando correctamente 🚀"

# ==============================
# 4. Ruta de predicción
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validación del input
        if not data or "texto" not in data:
            return jsonify({"error": "Falta el campo 'texto' en el JSON"}), 400

        texto = data["texto"]

        # Transformar texto con el vectorizador entrenado
        X = vectorizer.transform([texto])
        pred = modelo.predict(X)[0]

        # Mapear sentimiento
        if pred.lower() == "negative":
            sentimiento = "NEGATIVO 😞"
        elif pred.lower() == "neutral":
            sentimiento = "NEUTRO 😐"
        elif pred.lower() == "positive":
            sentimiento = "POSITIVO 😊"
        else:
            sentimiento = "DESCONOCIDO 🤔"

        return jsonify({
            "texto": texto,
            "sentimiento": sentimiento,
            "valor": pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 5. Ejecutar localmente
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# ==============================
# 5. Ejecutar servidor
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port=8000)
