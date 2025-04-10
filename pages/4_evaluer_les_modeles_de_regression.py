# Ceci est une alternative au 4_estimer_CO2.py

# Note à mes camarades de projet:
# pour ajouter ou modifier un modèle,
# éditez les fonctions handle_model_selection et display_results ("paramètres spécifiques au modèle")
# et ajoutez (ou modifiez) le modèle dans le dictionnaire model_options
# bonus si besoin : ajouter affichage de paramètres spécifiques au modèle dans display_model_parameters du fichier utils_streamlit_co2.py
# Ensuite il faut déposer ce fichier sur le github servant à streamlit (si c'est le github de Denis, lui demander)


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
def display_results(model_name, model, X_test, y_test, hyperparameters):
    # Effectuer les prédictions
    y_pred = model.predict(X_test)
    
    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Afficher les résultats principaux
    st.write(f"### {model_name}")
    st.write(f"- Mean Squared Error (MSE) : {mse:.2f}")
    st.write(f"- Coefficient de Détermination (R²) : {r2:.2f}")

    # Afficher les hyperparamètres
    st.write("**Hyperparamètres utilisés :**")
    if hyperparameters:
        for param, value in hyperparameters.items():
            st.write(f"- {param} : {value}")
    else:
        st.write("- Pas d'hyperparamètre appliqué")
    st.markdown("\n---\n")

    # Afficher les paramètres spécifiques au modèle
    #if isinstance(model, LinearRegression):
    #    st.write("**Paramètres du modèle :**")
    #    st.write(f" - Coefficient (pente) : {model.coef_[0]:.4f}")
    #    st.write(f" - Intercept (ordonnée à l'origine) : {model.intercept_:.4f}")
    display_model_parameters(model, X_test)


# Fonction appelée à chaque sélection de modèle
def handle_model_selection(model_name, model_class, X_train_scaled, X_test_scaled, y_train, y_test):

    # par défaut, on utilise toutes les variables explicatives
    X_train = X_train_scaled
    X_test = X_test_scaled

    # Initialisation des hyperparamètres par défaut
    hyperparameters = {}

    # Gestion des hyperparamètres via Streamlit
    if model_name == "Régression Linéaire simple":
        hyperparameters = {}
        X_train = X_train_scaled[['ec (cm3)']]
        X_test = X_test_scaled[['ec (cm3)']]

    elif model_name == "Forêt Aléatoire":
        n_estimators = st.slider("Nombre d'arbres (n_estimators)", 10, 200, 100)
        max_depth = st.slider("Profondeur maximale (max_depth)", 1, 20, 3)
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }

    elif model_name == "Support Vector Machine (SVM)":
        C = st.slider("Paramètre C", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Type de noyau (kernel)", ["linear", "rbf", "poly"])
        hyperparameters = {
            "C": C,
            "kernel": kernel
        }

    # Charger les hyperparamètres dans le modèle
    model = model_class(**hyperparameters)

    # Entraîner le modèle
    trained_model = train_model(model, X_train, y_train)

    # Afficher les résultats
    display_results(model_name, trained_model, X_test, y_test, hyperparameters)

# Charger les données
X_train_scaled, X_test_scaled, y_train, y_test = load_our_data()

# Dictionnaire des modèles disponibles
model_options = {
    "Régression Linéaire": LinearRegression,
    "Forêt Aléatoire": RandomForestRegressor,
    "Support Vector Machine (SVM)": SVR
}

# Interface utilisateur pour sélectionner un modèle
model_choice = st.selectbox("Choisissez un modèle :", list(model_options.keys()))

# Récupérer la classe du modèle sélectionné
selected_model_class = model_options[model_choice]

# Appeler la fonction pour gérer la sélection du modèle
handle_model_selection(
    model_name=model_choice,
    model_class=selected_model_class,
    X_train_scaled=X_train_scaled,
    X_test_scaled=X_test_scaled,
    y_train=y_train,
    y_test=y_test
)


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Recharger nos paramètres préférés")
