import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

def Arbol(Data):
  optionsx = st.multiselect(
    'Selecciona las columnas para las variables Predictoras',
    Data.columns, key = 0
  )
  st.write("Si no identifica las variables a seleccionar puede acceder previamente a la pestaña 'Selección de Características' donde encontrá dos formas de seleccionar que variables son necesarias para su análisis")
  if optionsx != []:
    X = np.array(Data[optionsx])
    optionsy = st.multiselect(
    'Selecciona la columna para la variable clase',
      Data.columns, key = 1)
    if optionsy != []:
      Y = np.array(Data[optionsy])
      X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                  test_size = 0.2, 
                                                                                  random_state = 0,
                                                                                  shuffle = True)
      option = st.selectbox(
      ' Elige un tipo de Árbol',
      ('','Árbol de decisión: Pronóstico','Árbol de decisión: Clasificación')
      )
      if option == 'Árbol de decisión: Pronóstico':
        Pronostico(optionsx, X_train, X_test, Y_train, Y_test)
      elif option == 'Árbol de decisión: Clasificación':
        Clasificacion(optionsx,X_train, X_test, Y_train, Y_test)

def Pronostico(optionsx,X_train, X_test, Y_train, Y_test):
    PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=0)
    PronosticoAD.fit(X_train, Y_train)
    Y_Pronostico = PronosticoAD.predict(X_test)
    st.write("**Parámetros del Modelo**")
    st.write("Bondad de Ajuste", r2_score(Y_test, Y_Pronostico))
    st.write('Criterio: \n', PronosticoAD.criterion)
    st.write('Importancia variables: \n', PronosticoAD.feature_importances_)
    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
    st.write("**Conformación del modelo de pronóstico**")
    plt.figure(figsize=(16,16))  
    plot_tree(PronosticoAD, feature_names = optionsx)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def Clasificacion(optionsx, X_train, X_test, Y_train, Y_test):
    ClasificacionAD = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=2)
    ClasificacionAD.fit(X_train, Y_train)
    Y_Clasificacion = ClasificacionAD.predict(X_test)
    st.write("**Reporte de la Clasificación**")
    st.write('Criterio: \n', ClasificacionAD.criterion)
    st.write('Importancia variables: \n', ClasificacionAD.feature_importances_)
    st.write("Bondad de Ajuste", ClasificacionAD.score(X_test, Y_test))
    st.text(classification_report(Y_test, Y_Clasificacion))
    st.write("**Matriz de Clasificación**")
    Matriz_Clasificacion = pd.crosstab(Y_test.ravel(), 
                                        Y_Clasificacion, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación']) 
    st.dataframe(Matriz_Clasificacion)
    st.write("**Gráfico del Bosque Generado**")
    plt.figure(figsize=(16,16))  
    plot_tree(ClasificacionAD, 
            feature_names = optionsx,
            class_names = Y_Clasificacion)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
