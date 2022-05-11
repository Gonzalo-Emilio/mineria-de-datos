from this import d
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns  
from EDA import EDA        
from Seleccion_de_caracteristicas import  seleccion_de_caracteristicas          # Para la visualización de datos basado en matplotlib               

#simulacion swtich case
def Switch(Election,Data):
    if Election == 'Muestra de Datos':
        st.dataframe(Data)
    elif Election == 'Análisis Exploratorio de Datos':
        EDA(Data)
    elif Election == 'Selección de características':
        seleccion_de_caracteristicas(Data)
    elif Election == 'Métricas de Distancia':
        st.text('holaaa soy metri xd') 
    elif Election == 'Clustering':
        st.text('holaaa soy eda xd') 
    elif Election == 'Reglas de Asociación': 
        st.text('holaaa soy eda xd')
    elif Election == 'Pronóstico: Regresión lineal':
        st.text('holaaa soy eda xd') 
    elif Election == 'Clasificación: Regresión Logística':
        st.text('holaaa soy eda xd')



    

