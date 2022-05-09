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
    st.subheader("ACP")
    st.write("**Estandarización de Datos**")
    st.write("Elección de borrado de Columnas de tipo Object")
    drop_column = borrado(Data)
    if drop_column is not None:
        sel = st.selectbox('Selecciona una forma de estandarización de datos',
        ('','Estandarización', 'Normalización'))
        if sel == 'Estandarización':
            st.subheader('Estandarización')
            Estandarizar = StandardScaler()
            MEstantadar = Estandarizar.fit_transform(drop_column)
            st.dataframe(MEstantadar)
            pca = PCA(0.85)     # pca=PCA(n_components=None), pca=PCA(.85)
            pca.fit(MEstantadar)          # Se obtiene los componentes
            st.write("**Matriz Componentes**")
            st.dataframe(pca.components_)
            st.write('**Varianza**')
            Varianza = pca.explained_variance_ratio_
            st.write('Proporción de varianza:',Varianza.tolist())

        elif sel == 'Normalización':
            st.subheader('Normalización')
            Normalizar = MinMaxScaler() 
            MNormalizar = Normalizar.fit_transform(drop_column)
            st.dataframe(MNormalizar) 
            pca = PCA(0.85)     # pca=PCA(n_components=None), pca=PCA(.85)
            pca.fit(MNormalizar)          # Se obtiene los componentes
            st.write('**Matriz Componentes**')
            st.write(pca.components_)
            st.write('**Varianza**')
            Varianza = pca.explained_variance_ratio_
            st.write('Proporción de varianza:', Varianza.tolist())

def borrado(Data):
    options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns
    )
    st.write('Nota: Es necesario eliminar las variables de tipo OBJECT o DATE, se pueden consultar los tipos de datos en la sección: Análisis Exploratorio de Datos -> Tipos de Datos')
    col1, col2 = st.columns(2)
    with col1:      
        st.write('Columnas Eliminadas:', options)
        elimination = Data.drop(columns=options)
        
    with col2:
        st.write('Columnas Restantes:', elimination.columns.tolist())

    st.dataframe(elimination)
    return elimination



#def seleccion(Estandar):
    
