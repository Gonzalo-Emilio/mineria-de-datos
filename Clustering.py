from webbrowser import Elinks
import streamlit as st
import pandas as pd
import numpy as np   
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def clutering(Data):
    option = st.selectbox(
      ' Elige un tipo de Clúster',
      ('Clúster Jerárquico','Clúster Particional')
    )
    if option == 'Clúster Jerárquico':
      geraquico(Data)
    elif option == 'Clúster Particional':
      particional(Data)
     
     
        
def geraquico(Data):
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
  Estandarizar = StandardScaler()
  MEstandar = Estandarizar.fit_transform(elimination)
  st.dataframe(MEstandar)
  plt.figure(figsize=(10, 7))
  plt.title("Casos del árbol Gerárquico")
  plt.xlabel('Componentes')
  plt.ylabel('Distancia')
  option = st.selectbox(
     ' Elige un tipo de Metrica de Distancia',
     ('Euclidiana','Chebyshev', 'Manhattan'))

  if option == 'Euclidiana':
    Arbol = shc.dendrogram(shc.linkage(MEstandar, method='complete', metric='euclidean'))
    st.pyplot()
    number=st.number_input("Selecciona el número de Clústers respecto al gráfico")
    if number == 0:
      st.write("Selecciona un valor diferente de 0")
    else:
      MJerarquico = AgglomerativeClustering(n_clusters=int(number), linkage='complete', affinity='euclidean')
      MJerarquico.fit_predict(MEstandar)
      elimination.assign(Cluster_H = 0)
      elimination['Clúster_H'] = MJerarquico.labels_
      CentroidesH = elimination.groupby('Clúster_H').mean()
      st.dataframe(CentroidesH)
      plt.figure(figsize=(10, 7))
      plt.scatter(MEstandar[:,0], MEstandar[:,1], c=MJerarquico.labels_)
      plt.grid()
      plt.show() 
      st.pyplot()

  elif option == 'Chebyshev':
    Arbol = shc.dendrogram(shc.linkage(MEstandar, method='complete', metric='chebyshev'))
    st.pyplot()
    number=st.number_input("Selecciona el número de Clústers respecto al gráfico")
    if number == 0:
      st.write("Selecciona un valor diferente de 0")
    else:
      MJerarquico = AgglomerativeClustering(n_clusters=int(number), linkage='complete', affinity='chebyshev')
      MJerarquico.fit_predict(MEstandar)
      elimination.assign(Cluster_H = 0)
      elimination['Clúster_H'] = MJerarquico.labels_
      CentroidesH = elimination.groupby('Clúster_H').mean()
      st.dataframe(CentroidesH)
      plt.figure(figsize=(10, 7))
      plt.scatter(MEstandar[:,0], MEstandar[:,1], c=MJerarquico.labels_)
      plt.grid()
      plt.show() 
      st.pyplot()


  elif option == 'Manhattan':
    Arbol = shc.dendrogram(shc.linkage(MEstandar, method='complete', metric='cityblock'))
    st.pyplot()
    number=st.number_input("Selecciona el número de Clústers respecto al gráfico")
    if number == 0:
      st.write("Selecciona un valor diferente de 0")
    else:
      MJerarquico = AgglomerativeClustering(n_clusters=int(number), linkage='complete', affinity='cityblock')
      MJerarquico.fit_predict(MEstandar)
      elimination.assign(Cluster_H = 0)
      elimination['Clúster_H'] = MJerarquico.labels_
      CentroidesH = elimination.groupby('Clúster_H').mean()
      st.dataframe(CentroidesH)
      plt.figure(figsize=(10, 7))
      plt.scatter(MEstandar[:,0], MEstandar[:,1], c=MJerarquico.labels_)
      plt.grid()
      plt.show() 
      st.pyplot()

  
def particional(Data):
  st.write("DATOSSSSSS")

