from asyncio import PriorityQueue
from typing import List
import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori


def rda(Data):
    ##Datos = pd.read_csv(Data,header=None)
    options = st.multiselect(
        'Selecciona las columnas que serán eliminadas',
        Data.columns, key = 0
        )
    st.write("Si no identifica las variables a eliminar puede acceder previamente a la pestaña 'Selección de Características' donde encontrá dos formas de seleccionar que variables son necesarias para su análisis")
    col1, col2 = st.columns(2)
    with col1:      
        st.write('Columnas Eliminadas:', options)
        elimination = Data.drop(columns=options)      
    with col2:
        st.write('Columnas Restantes:', elimination.columns.tolist())
    if elimination is not None:
        st.subheader("Procesamiento de los datos")
        st.write("Exploración:Antes de ejecutar el algoritmo  es recomendable observar la distribución de la frecuencia de los elementos.")
        st.write("Se incluyen todas las transacciones en una sola lista")
        Transacciones = elimination.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 0
        st.write("Se agrupa los elementos")
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        st.dataframe(ListaM)

        Lista = elimination.stack().groupby(level=0).apply(list).tolist() #Se crea una lista de listas a partir del dataframe y se remueven los 'NAN'
        ReglasC1 = apriori(Lista, min_support=0.01,  min_confidence=0.3,  min_lift=2) #Aplicación del algoritmo

        st.write("Total de reglas encontradas")
        ResultadosC1 = list(ReglasC1)
        st.write(len(ResultadosC1)) 

        st.write("Descripción de las reglas asociadas a los datos")
        for item in ResultadosC1:
            Emparejar = item[0]
            items = [x for x in Emparejar]
            st.text("Regla: " + str(item[0]))
            st.text("Soporte: " + str(item[1]))
            st.text("Confianza: " + str(item[2][0][2]))
            st.text("Lift: " + str(item[2][0][3])) 
            st.write("-------------------------------------")
    














