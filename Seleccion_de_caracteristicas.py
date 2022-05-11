#analisis exploratorio de datos
from io import SEEK_CUR
from tkinter.tix import Select
from xml.etree.ElementInclude import include
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
    st.write("Elección de borrado de Columnas de tipo Object")
    drop_column = borrado(Data)
    na_cero = drop_column.fillna(0)
    if drop_column is not None:
        sel = st.selectbox('Selecciona una forma de estandarización de datos',
        ('','Estandarización', 'Normalización'))
        if sel == 'Estandarización':
            st.subheader('Estandarización')
            Estandarizar = StandardScaler()
            seleccion(Estandarizar, na_cero)

        elif sel == 'Normalización':
            st.subheader('Normalización')
            Normalizar = MinMaxScaler() 
            seleccion(Normalizar, na_cero)
            
            


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



def seleccion(Estandarizar, na_cero):
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
    st.pyplot()

    st.write("**Proporción de Relevancia**")
    st.dataframe(pd.DataFrame(pca.components_, columns=na_cero.columns))
    st.write("**Identificación de Relevancia de Variables**")
    st.write()