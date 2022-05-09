#analisis exploratorio de datos
from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns   

def EDA(Data):
    #Datos = pd.read_csv(Data)
    st.subheader('Estructura de los Datos e Identificación de Datos Faltantes')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Dimensión**")
        st.text('Renglones:\t'+ str(Data.shape[0]))
        st.text('Columnas:\t'+ str(Data.shape[1]))

    with col2:
        st.write("**tipos de Datos**")
        st.text(Data.dtypes)

    with col3:
        st.write("**Total de Valores Nulos**")
        st.text(Data.isnull().sum())

    #Estadistica
    st.subheader("Resumen estadístico de variables numéricas")
    st.dataframe(Data.describe())
    st.subheader("Detección de Datos Atípicos")
    Columna = st.text_input('Digita la Columna de análisis')
    #Histograma
    if Columna:
        col1, col2 = st.columns(2)

        with col1:
            Data[Columna].hist(figsize=(14,14), xrot=45)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        with col2:
            sns.boxplot(data = Data[Columna], orient = 'h')
            plt.show()    
            st.pyplot()
    
    st.subheader("Identificación de relaciones entre pares variables")
    correlation = Data.corr()
    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(correlation)
    sns.heatmap(correlation, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    