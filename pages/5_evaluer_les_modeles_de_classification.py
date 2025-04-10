# Page Name : Evaluer les modèles de classification
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# charge les chemins vers les fichiers de données : base_processed, base_raw, base_models...
from init_notebook import base_processed, base_raw, base_models

# charge des fonctions communes pour streamlit
from utils_streamlit_co2 import display_model_parameters, display_model_parameters_classification

# charge des fonctions communes pour le projet
from common_co2 import load_our_data_cat, display_norm_matrix

pd.set_option('future.no_silent_downcasting', True)

st.set_page_config(
    page_title="Projet Datascientest - Classification des véhicules.",
    page_icon=":placard:",
)

st.markdown("# Classification des caractéristiques des véhicules")
st.sidebar.header("Nous entraînons des modèles de classification à partir des données")
st.write("Nous voulons classifier les véhicules en fonction de leurs caractéristiques spécifiques. "
         "En science des données, on parle d'un sujet de classification: prédire une classe ou une catégorie.")


# Fonction pour entraîner un modèle
def train_model_classification(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# Fonction pour afficher les résultats
def display_results_classification(model_name, model, X_test, y_test, hyperparameters):
    # Effectuer les prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Afficher les résultats principaux
    st.write(f"### {model_name}")
    st.write(f"- Précision (Accuracy) : {accuracy:.2f}")
    st.write("**Rapport de classification :**")
    st.write(pd.DataFrame(report).transpose())

    # Afficher les hyperparamètres
    st.write("**Hyperparamètres utilisés :**")
    if hyperparameters:
        for param, value in hyperparameters.items():
            st.write(f"- {param} : {value}")
    else:
        st.write("- Pas d'hyperparamètre appliqué")
    st.markdown("\n---\n")

    # Afficher les paramètres spécifiques au modèle
    display_model_parameters_classification(model, X_test)



# Fonction appelée à chaque sélection de modèle
def handle_model_selection_classification(model_name, model_class, X_train_scaled, X_test_scaled, y_train, y_test):

    # on utilise toutes les variables explicatives
    X_train = X_train_scaled
    X_test = X_test_scaled

    # Initialisation des hyperparamètres par défaut
    hyperparameters = {}

    # Gestion des hyperparamètres via Streamlit
    # pour une selectbox, la 1ere valeur serait sélectionnée par défaut (à tester)
    if model_name == "Forêt Aléatoire":
        # Sélectionner le nombre d'arbres
        n_estimators = st.slider("Nombre d'arbres (n_estimators)", 10, 500, 100)
        
        # Sélectionner la profondeur maximale des arbres
        max_depth = st.slider("Profondeur maximale (max_depth)", 1, 50, 20)
        
        # Sélectionner le critère de qualité
        criterion = st.selectbox("Critère de division (criterion)", ["gini", "entropy"], index=0)
        
        # Nombre minimal d'échantillons pour diviser un nœud
        min_samples_split = st.slider("Nombre minimal d'échantillons pour diviser un nœud (min_samples_split)", 2, 20, 2)
        
        # Nombre minimal d'échantillons dans une feuille
        min_samples_leaf = st.slider("Nombre minimal d'échantillons dans une feuille (min_samples_leaf)", 1, 10, 2)
        
        # Fonctionnalités maximales pour une division
        max_features = st.selectbox("Méthode de sélection des fonctionnalités (max_features)", ["sqrt", "log2", None], index=0)
        
        # Activation ou désactivation du bootstrap
        bootstrap = st.checkbox("Utiliser le Bootstrap", value=True)
        
        # Stocker les hyperparamètres dans un dictionnaire
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap
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
    trained_model = train_model_classification(model, X_train, y_train)

    # Afficher les résultats
    display_results_classification(model_name, trained_model, X_test, y_test, hyperparameters)

# Charger les données
X_train_scaled, X_test_scaled, y_train, y_test = load_our_data_cat()

# Dictionnaire des modèles disponibles
model_options_classification = {
    "Forêt Aléatoire": RandomForestClassifier,
    "Support Vector Machine (SVM)": SVC
}

# Interface utilisateur pour sélectionner un modèle
model_choice = st.selectbox("Choisissez un modèle :", list(model_options_classification.keys()))

# Récupérer la classe du modèle sélectionné
selected_model_class = model_options_classification[model_choice]

# Appeler la fonction pour gérer la sélection du modèle
handle_model_selection_classification(
    model_name=model_choice,
    model_class=selected_model_class,
    X_train_scaled=X_train_scaled,
    X_test_scaled=X_test_scaled,
    y_train=y_train,
    y_test=y_test
)