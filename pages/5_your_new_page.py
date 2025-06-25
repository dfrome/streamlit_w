# Page Name : page with a big image


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

# charge les chemins vers les fichiers de données : base_processed, base_raw, base_models...
from init_notebook import base_processed, base_raw, base_models

# charge des fonctions faites pour streamlit
from utils_streamlit_co2 import display_model_parameters

pd.set_option('future.no_silent_downcasting', True)

st.set_page_config(
    page_title="Projet Datascientest - page with big image.",
    page_icon=":dart:",
)

# pour icone classification on pourra prendre 	:placard:
# sinon il y a :balances: aussi
# ref: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

#st.markdown("# Estimations de l'émission de CO2")
#st.sidebar.header("Nous entraînons des modèles de régression à partir des données")
#st.write("Nous voulons estimer l'émission de CO2 en fonction des caractéristiques des véhicules\n  "
#"En science des données, on parle alors d'un sujet de regression: estimer une valeur numérique continue.")

image_path = "images/big_01.png"
image = Image.open(image_path)
st.image(image)