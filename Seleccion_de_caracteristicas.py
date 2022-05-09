#analisis exploratorio de datos
from xml.etree.ElementInclude import include
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

def seleccion(Data):
    st.subheader("Estandarización de Datos")
    st.write("Elección de borrado de Columnas de tipo Object")
    drop_column = borrado(Data)
    if drop_column is not None:
        sel = st.selectbox('Selecciona una forma de estandarización de datos',
        ('','Estandarización', 'Normalización'))
        if sel == 'Estandarización':
            st.subheader('Estandarización')
            Estandarizar = StandardScaler()
            Mestantadar = Estandarizar.fit_transform(drop_column)
            st.dataframe(Mestantadar)
        elif sel == 'Normalización':
            st.subheader('Normalización')
            Normalizar = MinMaxScaler() 


def borrado(Data):
    #datos = Data.drop(Data.select_types(include = 'object').columns)
    #st.table(datos)
    st.write('Es necesario eliminar las variables de tipo OBJECT, se pueden consultar los tipos de variables en la sección: Análisis Exploratorio de Datos -> Tipos de Datos')
    options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns
    )
    st.write('Nota: Es necesario eliminar las variables de tipo OBJECT, se pueden consultar los tipos de variables en la sección: Análisis Exploratorio de Datos -> Tipos de Datos')
    st.write('Columnas Eliminadas:', options)
    elimination = Data.drop(columns=options)
    st.dataframe(elimination)
    return elimination

#def seleccion(data, Estandar):
#    Mestantadar = Estandar.fit_transform(data)
#    st.dataframe(Mestantadar)
