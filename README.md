##Proyecto modulo 7 - Tecnicas avanzadas
Autora: Francisca Bittner

Proyecto de Machine Learning que analiza comentarios de usuarios en Google Play Store para clasificarlos como positivos, negativos o neutros.
El modelo se desarrolló con Python, TF-IDF y Linear SVC, y se desplegó como una API en Render.com.

#Dataset - comentarios de Google Play Store
Contiene más de 64.000 reseñas con columnas:
Translated_Review: comentario traducido
Sentiment: sentimiento (Positive, Negative, Neutral)

#Proceso
Limpieza de texto: eliminación de stopwords, signos y lematización.
Vectorización con TF-IDF: 7.000 características.
Modelo LinearSVC con ajuste de hiperparámetros mediante GridSearchCV.
Evaluación con accuracy y reporte de clasificación.

#Resultados
Alta precisión en clases positivas y negativas.
Ligera confusión en neutros (caso común en análisis de texto).
Buen rendimiento general en validación y test.

#Despliegue
API REST creada con Flask y hospedada en Render.com.
Link: https://proyecto-udd-m7.onrender.com
