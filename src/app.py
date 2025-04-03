# from utils import db_connect
# engine = db_connect()

import streamlit as st
import joblib
import pickle
import numpy as np

# Cargar el modelo entrenado
MODEL_PATH = "/workspaces/Random_forest/models/randomforest_classifier_n200_42.sav"
model = joblib.load(MODEL_PATH)

st.title("Predicción de Diabetes")

# Entradas del usuario
# Entradas del usuario
pregnancies = st.number_input("Número de embarazos", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucosa en sangre", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Presión arterial", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Grosor de piel (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulina (mu U/ml)", min_value=0, max_value=1000, value=30)
bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=100.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Edad", min_value=0, max_value=120, value=30)


# Crear un array con las entradas del usuario
input_data = np.array([[pregnancies, glucose, bmi, age, blood_pressure, skin_thickness, insulin, diabetes_pedigree]])

# Predicción
if st.button("Predecir"):
    prediction = model.predict(input_data)
    resultado = "Diabético" if prediction[0] == 1 else "No Diabético"
    st.write(f"Resultado: **{resultado}**")
