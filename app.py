import streamlit as st
import numpy as np
import joblib

# --- Cargar el modelo y el escalador ---
try:
    model = joblib.load('modelo_vino.joblib')
    scaler = joblib.load('escalador_vino.joblib')
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo o del escalador.")
    st.error("Asegúrate de que 'modelo_vino.joblib' y 'escalador_vino.joblib' estén en la misma carpeta que este script.")
    st.stop()

# --- Función de predicción ---
def predict_wine_variety(features):
    input_array = np.array(features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    variety_names = ['Variedad A', 'Variedad B', 'Variedad C']
    return variety_names[int(prediction[0])]

# --- Interfaz de Usuario con Streamlit ---
st.set_page_config(page_title="Clasificador de Variedad de Vino", layout="centered")

st.title('🍷 Clasificador de Variedad de Vino')
st.markdown('Introduce las 13 características del vino para predecir su variedad.')
st.info('**Importante:** Por favor, utiliza un **punto (.)** para los valores decimales.')

# Crear pestañas para organizar las entradas
tab1, tab2 = st.tabs(["Características Físico-químicas", "Componentes Fenólicos y Prolinas"])

user_inputs_dict = {}

# Definir los nombres y valores predeterminados para los campos de entrada
# Esto hace que la interfaz se vea más profesional y sea más fácil de usar
input_params = {
    'Alcohol': 13.0,
    'Malic acid': 2.0,
    'Ash': 2.0,
    'Alcalinity of ash': 17.0,
    'Magnesium': 100.0,
    'Color intensity': 5.0,
    'Hue': 1.0,
    'Total phenols': 2.5,
    'Flavanoids': 2.0,
    'Nonflavanoid phenols': 0.2,
    'Proanthocyanins': 1.5,
    'OD280/OD315 of diluted wines': 2.5,
    'Proline': 800.0
}

with tab1:
    st.header("Características Físico-químicas")
    col1, col2 = st.columns(2)
    with col1:
        user_inputs_dict['Alcohol'] = st.number_input('Alcohol', value=input_params['Alcohol'], step=0.01, format="%.2f")
        user_inputs_dict['Malic acid'] = st.number_input('Malic acid', value=input_params['Malic acid'], step=0.01, format="%.2f")
        user_inputs_dict['Ash'] = st.number_input('Ash', value=input_params['Ash'], step=0.01, format="%.2f")
        user_inputs_dict['Alcalinity of ash'] = st.number_input('Alcalinity of ash', value=input_params['Alcalinity of ash'], step=0.01, format="%.2f")
    with col2:
        user_inputs_dict['Magnesium'] = st.number_input('Magnesium', value=input_params['Magnesium'], step=1.0, format="%d")
        user_inputs_dict['Color intensity'] = st.number_input('Color intensity', value=input_params['Color intensity'], step=0.01, format="%.2f")
        user_inputs_dict['Hue'] = st.number_input('Hue', value=input_params['Hue'], step=0.01, format="%.2f")

with tab2:
    st.header("Componentes Fenólicos y Prolinas")
    col1, col2 = st.columns(2)
    with col1:
        user_inputs_dict['Total phenols'] = st.number_input('Total phenols', value=input_params['Total phenols'], step=0.01, format="%.2f")
        user_inputs_dict['Flavanoids'] = st.number_input('Flavanoids', value=input_params['Flavanoids'], step=0.01, format="%.2f")
        user_inputs_dict['Nonflavanoid phenols'] = st.number_input('Nonflavanoid phenols', value=input_params['Nonflavanoid phenols'], step=0.01, format="%.2f")
    with col2:
        user_inputs_dict['Proanthocyanins'] = st.number_input('Proanthocyanins', value=input_params['Proanthocyanins'], step=0.01, format="%.2f")
        user_inputs_dict['OD280/OD315 of diluted wines'] = st.number_input('OD280/OD315 of diluted wines', value=input_params['OD280/OD315 of diluted wines'], step=0.01, format="%.2f")
        user_inputs_dict['Proline'] = st.number_input('Proline', value=input_params['Proline'], step=1.0, format="%d")

# Botón y resultado
if st.button('Predecir Variedad'):
    feature_names = list(input_params.keys())
    user_inputs = [user_inputs_dict[name] for name in feature_names]
    
    predicted_variety = predict_wine_variety(user_inputs)
    st.success(f'La variedad de vino predicha es: **{predicted_variety}**')