import gradio as gr
import numpy as np
import joblib

# Cargar el modelo y el escalador
model = joblib.load('modelo_vino.joblib')
scaler = joblib.load('escalador_vino.joblib')

# Definir la función de predicción
def predict_wine_variety(*args):
    # 'args' contendrá los 13 valores ingresados por el usuario
    input_array = np.array(args).reshape(1, -1)
    scaled_input = scaler.transform(input_array) 
    prediction = model.predict(scaled_input)

    variety_names = ['Variedad A', 'Variedad B', 'Variedad C']
    return variety_names[int(prediction[0])]

# Crear la interfaz de Gradio
inputs = [gr.Number(label=name) for name in ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 
                                           'Magnesium', 'Total_phenols', 'Flavanoids', 
                                           'Nonflavanoid_phenols', 'Proanthocyanins', 
                                           'Color_intensity', 'Hue', 'Od_280_315_of_diluted_wines', 'Proline']]

gr.Interface(fn=predict_wine_variety, 
             inputs=inputs, 
             outputs="text",
             title="Clasificador de Variedad de Vino",
             description="Introduce las 13 características del vino y obtén la variedad predicha.").launch()