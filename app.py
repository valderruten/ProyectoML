import streamlit as st
import numpy as np
import joblib

# Cargar el modelo y el escalador (igual que antes)
model = joblib.load('modelo_vino.joblib')
scaler = joblib.load('escalador_vino.joblib')

def predict_wine_variety(features):
    input_array = np.array(features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    variety_names = ['Variedad A', 'Variedad B', 'Variedad C']
    return variety_names[int(prediction[0])]

# Título de la app
st.title('Clasificador de Variedad de Vino')
st.markdown('Introduce las 13 características del vino para predecir su variedad.')

# Crear pestañas para organizar las entradas
tab1, tab2 = st.tabs(["Características Físico-químicas", "Componentes Fenólicos"])

user_inputs = []

with tab1:
    st.header("Características Físico-químicas")
    user_inputs.append(st.number_input('Alcohol', value=0.0))
    user_inputs.append(st.number_input('Malic acid', value=0.0))
    user_inputs.append(st.number_input('Ash', value=0.0))
    user_inputs.append(st.number_input('Alcalinity of ash', value=0.0))
    user_inputs.append(st.number_input('Magnesium', value=0.0))
    user_inputs.append(st.number_input('Color intensity', value=0.0))
    user_inputs.append(st.number_input('Hue', value=0.0))

with tab2:
    st.header("Componentes Fenólicos y Prolinas")
    user_inputs.append(st.number_input('Total phenols', value=0.0))
    user_inputs.append(st.number_input('Flavanoids', value=0.0))
    user_inputs.append(st.number_input('Nonflavanoid phenols', value=0.0))
    user_inputs.append(st.number_input('Proanthocyanins', value=0.0))
    user_inputs.append(st.number_input('OD280/OD315 of diluted wines', value=0.0))
    user_inputs.append(st.number_input('Proline', value=0.0))

# Botón y resultado
if st.button('Predecir Variedad'):
    predicted_variety = predict_wine_variety(user_inputs)
    st.success(f'La variedad de vino predicha es: **{predicted_variety}**')