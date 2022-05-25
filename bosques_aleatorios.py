from click import option
import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection


def Bosques(Data):
    optionsx = st.multiselect(
      'Selecciona las columnas para las variables Predictoras',
      Data.columns, key = 0
    )
    st.write("Si no identifica las variables a eliminar puede acceder previamente a la pestaña 'Selección de Características' donde encontrá dos formas de seleccionar que variables son necesarias para su análisis")
    X = np.array(Data[optionsx])
    st.dataframe(pd.DataFrame(X))
    optionsy = st.multiselect(
      'Selecciona la columna para la variable clase',
      Data.columns, key = 1
    )

    Y = np.array(Data[optionsy])
    pd.DataFrame(Y)
    st.dataframe(pd.DataFrame(Y))
    if X is not None:
        if Y is not None:
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)
            ClasificacionBA = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=0)
            ClasificacionBA.fit(X_train, Y_train)
            Y_Clasificacion = ClasificacionBA.predict(X_validation)
            ClasificacionBA.score(X_validation, Y_validation)

