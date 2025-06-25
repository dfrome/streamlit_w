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

st.write("""L’un des grands enjeux pour le data scientist est de trouver le bon modèle, mais aussi d’en tirer le meilleur. D’ailleurs il faut s’approcher du meilleur de chaque modèle pour sélectionner le bon. 
Et parfois cela peut être obscur tant il y a d’ajustements possibles.
Pour nous aider, l’outil mlflow peut s’avérer précieux pour consigner les résultats de campagnes d’apprentissage avec des hyperparamètres différents.
Ceci permet de cibler les plages d’hyperparamètres favorisant les bons scores, et surtout d’éliminer des plages ou des types qui plombent les scores.
Ainsi, une bonne stratégie est de réaliser quelques random search avec une plage large de paramètres, et décortiquer les résultats sur mlflow pour procéder à un gridsearch ensuite.
Voici ce qui a pu être fait sur k-NN par exemple:
         """)
image_path = "images/mlflow_knn_01.jpeg"
image = Image.open(image_path)
st.image(image)
st.write("""
ici, l’hyperparamètre à droite est la méthode chebyshev, que nous pouvons donc éliminer pour la suite
         """)
image_path = "images/mlflow_knn_02.jpeg"
image = Image.open(image_path)
st.image(image)
st.write("""
ici, l’hyperparamètre correspondant au type de calcul de distance est décidément à favoriser
""")
st.write("""
(images pour ref)
         """)
st.write("""


Graph mlflow. En abscisse : Minkowski / Manhattan / Ericsson / Chebyshev. En ordonnée les MSE des runs.


Graph mlflow. En abscisse : distance / uniform. En ordonnée les MSE des runs.

Exemple avec hyperparamètres numériques:
         """)
image_path = "images/mlflow_03.png"
image = Image.open(image_path)
st.image(image)
st.write("""Avec suffisamment d’explorations randomForestRegressor, et en se fixant comme contrainte de garder un min_sample_leaf > 1, on voit ci-dessus qu’on peut :
fixer min_sample_leaf == 3 car on a déjà de bons résultats avec cette valeur >1,
choisir min_sample_split == 2 ou == 6 (on décidedonc 6 pour un modèle plus robuste)
garder une max_depth de 20 qui est déjà suffisante pour les bons résultats.
  
""")



