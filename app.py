import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

try:
    modelo = joblib.load("modelo_svc2.pkl")
    vectorizer = joblib.load("vectorizer_tfidf_svc2.pkl")
    print("✅ Modelo y vectorizador cargados correctamente.")
except Exception as e:
    print("❌ Error cargando modelo/vectorizador:", e)

@app.route("/")
def home():
    return "API de predicción funcionando correctamente 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "texto" not in data:
            return jsonify({"error": "Falta el campo 'texto' en el JSON"}), 400

        texto = data["texto"]
        X = vectorizer.transform([texto])
        pred = modelo.predict(X)[0]

        if pred.lower() == "negative":
            sentimiento = "NEGATIVO 😞"
        elif pred.lower() == "neutral":
            sentimiento = "NEUTRO 😐"
        elif pred.lower() == "positive":
            sentimiento = "POSITIVO 😊"
        else:
            sentimiento = "DESCONOCIDO"

        return jsonify({"texto": texto, "sentimiento": sentimiento, "valor": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
