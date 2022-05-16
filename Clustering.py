import streamlit as st
import pandas as pd
import numpy as np   


def clutering(Data):
    st.subheader("Clústers")
    st.write("Para la eliminación de variables se pueden seleccionar directamene, si no determina que variables eliminar, puede acceder a la sección 'Selección de Características")
    options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns, key = 0
    )
    st.write('Columnas Eliminadas:', options)
    Colum_drop = Data.drop(columns=options)
    st.dataframe(Colum_drop)
    option = st.selectbox(
     ' Elige un tipo de Selección de características',
     ('Clúster Jerárquico','Clúster Particional'))
