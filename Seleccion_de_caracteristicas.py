#analisis exploratorio de datos
from io import SEEK_CUR
from tkinter.tix import Select
from xml.etree.ElementInclude import include
from django.template import Origin
import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

def seleccion_de_caracteristicas(Data):
    st.subheader("ACP")
    st.write("**Estandarización de Datos**")
    st.write("Eliminación de Datos")
    st.write('Nota: Es necesario eliminar las variables de tipo OBJECT o DATE, se pueden consultar los tipos de datos en la sección: Análisis Exploratorio de Datos -> Tipos de Datos')
    drop_column = borrado(Data)
    na_cero = drop_column.fillna(0)
    if drop_column is not None:
        sel = st.selectbox('Selecciona una forma de estandarización de datos',
        ('','Estandarización', 'Normalización'))
        if sel == 'Estandarización':
            st.subheader('Estandarización')
            Estandarizar = StandardScaler()
            seleccion(Estandarizar, na_cero, Data)

        elif sel == 'Normalización':
            st.subheader('Normalización')
            Normalizar = MinMaxScaler() 
            seleccion(Normalizar, na_cero, Data)
            
            


def borrado(Data):
    options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns, key = 0
    )
    col1, col2 = st.columns(2)
    with col1:      
        st.write('Columnas Eliminadas:', options)
        elimination = Data.drop(columns=options)
        
    with col2:
        st.write('Columnas Restantes:', elimination.columns.tolist())

    st.dataframe(elimination)
    return elimination



def seleccion(Estandarizar, na_cero, Original):
    sum_varianza = 0 
    MEstandar = Estandarizar.fit_transform(na_cero)
    st.dataframe(MEstandar)
    pca = PCA(0.85)     # pca=PCA(n_components=None), pca=PCA(.85)
    pca.fit(MEstandar)          # Se obtiene los componentes
    st.write("**Matriz Componentes**")
    st.dataframe(pca.components_)
    st.write('**Varianza**')
    Varianza = pca.explained_variance_ratio_
    varianza2 = Varianza.tolist()
    st.write('Proporción de varianza:', varianza2)
    sum_varianza=st.number_input("Selecciona el numero de varianzas a sumar")
    if sum_varianza >= len(varianza2):
            st.write("Digita un valor en rango")
    else:
        suma = sum(Varianza[0:int(sum_varianza)+1])
        if suma >= 0.75 and suma <= 0.90:
            st.write("Suma de varianza:", suma, "Aceptable")
        else:
            st.write("Suma de varianza:", suma, "no es aceptable")
    
    st.write("**Gráfica de Varianza**")
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza acumulada')
    plt.grid()
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.write("**Proporción de Relevancia**")
    Data = pd.DataFrame(pca.components_, columns=na_cero.columns)
    st.dataframe(Data)
    st.write("**Identificación de Relevancia de Variables**")
    st.write("Seleciona las variables que se van a tener dependiendo de la relevancia, en un porcentaje seleccionado por el usuario por ejemplo '30%'")
    st.write("Ejemplo:")
    st.image('img/ejemplo_relevancia.png')
    options2 = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns, key = 1
    )
    st.dataframe(Data)
    st.write('Columnas Eliminadas:', options2)
    elimination = Original.drop(columns=options2)
    st.dataframe(elimination)