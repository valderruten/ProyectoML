import streamlit as st
import numpy as np
import joblib

# Cargar el modelo y el escalador
model = joblib.load('modelo_vino.joblib')
scaler = joblib.load('escalador_vino.joblib')

# Definir la función de predicción
# Nota: La lógica del modelo es la misma.
def predict_wine_variety(features):
    input_array = np.array(features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    
    variety_names = ['Variedad A', 'Variedad B', 'Variedad C']
    return variety_names[int(prediction[0])]

# --- Interfaz de Usuario con Streamlit ---
# Título y descripción de la aplicación
st.title('Clasificador de Variedad de Vino')
st.markdown('Introduce las 13 características del vino para predecir su variedad.')

# Crear una lista de nombres de características para los campos de entrada
feature_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
                 'Magnesium', 'Total phenols', 'Flavanoids', 
                 'Nonflavanoid phenols', 'Proanthocyanins', 
                 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Crear campos de entrada de números para el usuario
user_inputs = []
for name in feature_names:
    user_inputs.append(st.number_input(f'Valor de {name}:', value=0.0))

# Botón para hacer la predicción
if st.button('Predecir Variedad'):
    # Llama a la función de predicción con los valores del usuario
    predicted_variety = predict_wine_variety(user_inputs)
    
    # Muestra el resultado
    st.success(f'La variedad de vino predicha es: **{predicted_variety}**')