# Page Name: MLflow
import streamlit as st
import time
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules.",
    page_icon=":straight_ruler:",
)

st.markdown("# CO2 MLflow")
st.sidebar.header("Affiner les choix d'hyperparamètres")

st.write("""Choisir et affiner un modèle est un défi pour tout data scientist.  
         MLflow nous aide à comparer les performances selon différents hyperparamètres, en identifiant ceux qui améliorent les scores.  
         Une méthode efficace consiste à lancer un random search large, analyser les résultats dans MLflow, puis affiner via un grid search ciblé.  
Voici ce qui a pu être fait sur k-NN par exemple:
         """)
image_path = "images/mlflow_knn_01.jpeg"
image = Image.open(image_path)
st.image(image)
st.write("""
ici, l’hyperparamètre à droite est la méthode chebyshev, que nous pouvons donc éliminer pour la suite
         """)
#image_path = "images/mlflow_knn_02.jpeg"
#image = Image.open(image_path)
#st.image(image)
#st.write("""
#ici, l’hyperparamètre correspondant au type de calcul de distance est décidément à favoriser
#""")

st.write("""

Exemple avec hyperparamètres numériques:
         """)
image_path = "images/mlflow_03.png"
image = Image.open(image_path)
st.image(image)
st.write("""Avec suffisamment d’explorations randomForestRegressor, et en se fixant comme contrainte de garder un min_sample_leaf > 1, on voit ci-dessus qu’on peut :  
* fixer min_sample_leaf == 3 car on a déjà de bons résultats avec cette valeur >1,  
* choisir min_sample_split == 2 ou == 6 (on décide donc 6 )  
* garder une max_depth de 20.
  
""")



