# Ceci est une alternative au 4_estimer_CO2.py
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
pd.set_option('future.no_silent_downcasting', True)

st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules.",
    page_icon=":dart:",
)

# pour icone classification on pourra prendre 	:placard:
# sinon il y a :balances: aussi
# ref: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.markdown("# Estimations de l'émission de CO2")
st.sidebar.header("Nous entraînons des modèles de régression à partir des données")
st.write("Nous voulons estimer l'émission de CO2 en fonction des caractéristiques des véhicules  "
"En science des données, on parle d'un sujet de regression: estimer une valeur numérique continue.")

            
def load_our_data():
    
    X_train_scaled = pd.read_csv(base_processed + 'X_train_scaled.csv')
    X_test_scaled = pd.read_csv(base_processed + 'X_test_scaled.csv')
    y_train = pd.read_csv(base_processed + 'y_train.csv')
    y_test = pd.read_csv(base_processed + 'y_test.csv')
    X_train_scaled = X_train_scaled.replace({False: 0, True: 1}).astype(float)
    X_test_scaled = X_test_scaled.replace({False: 0, True: 1}).astype(float)
    #
    y_column = "Ewltp (g/km)"
    y_train = y_train[y_column]
    y_test = y_test[y_column]
    return X_train_scaled, X_test_scaled, y_train, y_test

# Fonction pour entraîner un modèle
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Fonction pour afficher les résultats
def display_results(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### {model_name}")
    st.write(f"- Mean Squared Error (MSE) : {mse:.2f}")
    st.write(f"- Coefficient de Détermination (R²) : {r2:.2f}")

    # Afficher les paramètres spécifiques aux modèles
    if isinstance(model, LinearRegression):
        st.write("Paramètres du modèle :")
        st.write(f" - Dans le cadre d'une regression simple, on a une seule variable explicative: nous choisissons la cylindrée, 'ec (cm3)'")
        st.write(f" - Coefficient (pente) : {model.coef_[0]:.4f}")
        st.write(f" - Intercept (ordonnée à l'origine) : {model.intercept_:.4f}")

# Charger les données
X_train_scaled, X_test_scaled, y_train, y_test = load_our_data()

# Interface utilisateur pour sélectionner le modèle
model_options = {
    "Régression Linéaire": LinearRegression(),
    "Forêt Aléatoire": RandomForestRegressor(),
    "Support Vector Machine (SVM)": SVR()
}

model_choice = st.selectbox("Choisissez un modèle :", list(model_options.keys()))

# Sélectionner la variable explicative 'ec (cm3)'
X_train_ec = X_train_scaled[['ec (cm3)']]
X_test_ec = X_test_scaled[['ec (cm3)']]

# Entraîner le modèle sélectionné
selected_model = train_model(model_options[model_choice], X_train_ec, y_train)

# Afficher les résultats
display_results(model_choice, selected_model, X_test_ec, y_test)