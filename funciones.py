from this import d
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns  
from EDA import EDA       
from Seleccion_de_caracteristicas import  seleccion_de_caracteristicas          
from Clustering import clutering

#simulacion swtich case
def Switch(Election,Data):
    if Election == 'Muestra de Datos':
        st.header("Datos")
        st.dataframe(Data)
    elif Election == 'Análisis Exploratorio de Datos':
        st.header("Análisis Exploratorio de Datos")
        EDA(Data)
    elif Election == 'Selección de características':
        st.header("Selección de Características")
        seleccion_de_caracteristicas(Data)
    elif Election == 'Clustering':
        st.header("Clústers")
        #Data_Selection = seleccion_de_caracteristicas(Data)
        clutering(Data)
    elif Election == 'Reglas de Asociación': 
        st.text('holaaa soy eda xd')
    elif Election == 'Pronóstico: Regresión lineal':
        st.text('holaaa soy eda xd') 
    elif Election == 'Clasificación: Regresión Logística':
        st.text('holaaa soy eda xd')



    

