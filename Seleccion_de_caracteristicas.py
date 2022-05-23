import streamlit as st
import pandas as pd
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

def seleccion_de_caracteristicas(Data):
    option = st.selectbox(
     ' Elige un tipo de Selección de características',
     ('ACP','ACD'))
    if option == 'ACP':
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
                Data_caract = ACP(Estandarizar, na_cero, Data)
                return Data_caract
            elif sel == 'Normalización':
                st.subheader('Normalización')
                Normalizar = MinMaxScaler() 
                Data_caract = ACP(Normalizar, na_cero, Data)
                return Data_caract
    if option == 'ACD':
        st.subheader("ACD")
        Data_caract = ACD(Data)
        return Data_caract
            
            


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



def ACP(Estandarizar, na_cero, Original):
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
    st.write("Seleciona las variables que se van a tener dependiendo de la proporción de relevancia, en un porcentaje seleccionado por el usuario por ejemplo '30%'")
    st.write("Ejemplo:")
    st.image('img/ejemplo_relevancia.png')
    options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns, key = 1
    )
    st.write('Columnas Eliminadas:', options)
    st.dataframe(na_cero.drop(columns=options))
    return na_cero.drop(columns=options)


def ACD(Data):
    st.write("Se realiza una evaluación visual de los datos respecto a la tabla de datos, a partir de la correlación de Pearson el usuario puede realizar un inspección, en la que se buscará relaciones altas")
    st.write("De -1.0 a -0.70 y 0.70 a 1.0 se conocen como correlaciones fuertes o altas.")
    st.write("De -0.69 a -0.31 y 0.31 a 0.70 se conocen como correlaciones moderadas o medias.")
    st.write("De -0.30 a 0.0 y 0.0 a 0.30 se conocen como correlaciones débiles o bajas.")
    CorrData = Data.corr(method = 'pearson')
    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(CorrData)
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.show()
    st.pyplot()
    options = st.multiselect(
      'Selecciona las columnas que serán eliminadas',
      Data.columns, key = 2
    )
    st.write('Columnas Eliminadas:', options)
    
    st.dataframe(Data.drop(columns=options))
    return Data.drop(columns=options)
    