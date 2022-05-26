import streamlit as st
import pandas as pd
from funciones import Switch


Title = """
    <h1><center>GUI de Mineria de Datos</center></h1>
    <h3><center>Aplicación diseñada con el proposito de utilizar algoritmos de Análisis de Datos</center></h3>
    <br>
"""
st.markdown(Title ,unsafe_allow_html=True) #informacion principal
st.text('Inserta archivo con extensión CSV')
data = st.file_uploader('Ingresa tu archivo', type =['csv']) #insercion de datos tipo csv de forma local

if data is not None:
    dataframe = pd.read_csv(data)
    Algoritmo = st.selectbox('Elige un algoritmo de implementacion',
                ('Muestra de Datos','Análisis Exploratorio de Datos', 'Selección de características',  
                'Clustering', 
                'Patrones Secuenciales', 
                'Pronóstico', 
                'Clasificación'))
    Switch(Algoritmo, dataframe)