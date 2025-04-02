# from utils import db_connect
# engine = db_connect()

import streamlit as st
import pickle
import numpy as np

# Cargar el modelo entrenado
with open("modelo_random_forest.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Predicción de Diabetes")

# Entradas del usuario
pregnancies = st.number_input("Número de embarazos", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucosa en sangre", min_value=0, max_value=300, value=100)
bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=100.0, value=25.0)
age = st.number_input("Edad", min_value=0, max_value=120, value=30)

# Crear un array con las entradas del usuario
input_data = np.array([[pregnancies, glucose, bmi, age]])

# Predicción
if st.button("Predecir"):
    prediction = model.predict(input_data)
    resultado = "Diabético" if prediction[0] == 1 else "No Diabético"
    st.write(f"Resultado: **{resultado}**")
