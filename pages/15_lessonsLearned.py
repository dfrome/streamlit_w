# Page Name: Final

import streamlit as st
import time
import numpy as np
from PIL import Image

#st.set_page_config(page_title="Plotting Demo", page_icon=":checkered_flag:")
st.set_page_config(
    page_title="Prédiction d'émission de CO2 des véhicules.",
    page_icon=":checkered_flag:",
)

st.markdown("# Prédiction d'émission de CO2 des véhicules")
st.sidebar.header("Fin de notre projet")
#st.write(
#    """Nous explorons tout d'abord les données."""
#)


images = [
    {"path": "images/co2_digest.png", "caption": ""},
]

st.set_page_config(page_title="Résumé du Projet CO₂", layout="wide")

# Header
st.title("📊 Prédiction des Émissions de CO₂ selon les Caractéristiques des Véhicules")
st.subheader("Projet DataScientest — Septembre 2024 à Juin 2025")

# Section: Intro
st.markdown("### 🎯 Objectif du projet")
st.markdown("""
Prédire les émissions de CO₂ des véhicules à partir de leurs caractéristiques techniques pour aider les constructeurs à :
- Estimer l’impact environnemental d’un nouveau modèle
- Optimiser les paramètres de conception pour minimiser les émissions
""")
st.image("images/objectif.jpg", caption="Objectif du projet (à insérer)", use_column_width=True)

# Section: Données & Préparation
st.markdown("### 🔍 Données & Préparation")
st.markdown("""
- Base EEA de 2022 retenue (plus complète que 2023)
- Traitement rigoureux des valeurs manquantes, outliers, et doublons
- Création de nouvelles variables (e.g., innovations, types d’énergie)
- Scaling avec RobustScaler et MinMaxScaler
""")
st.image("images/data_cleaning.jpg", caption="Traitement des données", use_column_width=True)

# Section: Visualisations clés
st.markdown("### 📈 Visualisations clés")
st.markdown("""
- Répartition par carburant : essence majoritaire, mais très polluant
- Corrélations fortes : consommation carburant, cylindrée, puissance
- Insights métier : hybrides à forte autonomie → faibles émissions
""")
st.image("images/visualisation.jpg", caption="Exemples de visualisation", use_column_width=True)

# Section: Modèles de régression
st.markdown("### 🤖 Modèles de régression testés")
st.markdown("""
- Régression linéaire : insuffisante (hétéroscédasticité, non-normalité)
- SVR : bonne adaptation, mais temps de calcul élevé
- Random Forest : **Meilleur modèle**, R² = 0.985, MSE ≈ 49
""")
st.image("images/regression_models.jpg", caption="Comparaison des modèles de régression", use_column_width=True)

# Section: Modèles de classification
st.markdown("### 🏷️ Modèles de classification")
st.markdown("""
- Objectif : prédire l’étiquette énergétique des véhicules
- LightGBM & XGBoost : **Performances excellentes**, F1-score ≈ 0.90
- SHAP utilisé pour expliquer les décisions des modèles
""")
st.image("images/classification_models.jpg", caption="Modèles de classification", use_column_width=True)

# Section: Interprétabilité & Insights
st.markdown("### 🧠 Interprétabilité & Recommandations métier")
st.markdown("""
- Importance des variables : autonomie électrique, masse, cylindrée
- Recommandations techniques :
  - Essence/Diesel : réduire masse et puissance
  - Hybride : augmenter autonomie
  - Axes plus étroits → moindre émission
""")
st.image("images/shap_importance.jpg", caption="Interprétabilité SHAP", use_column_width=True)

# Section: Impacts & Reutilisation
st.markdown("### 🔄 Impacts industriels et réutilisation du modèle")
st.markdown("""
- Modèle utilisable via service web par les constructeurs
- Réentraînement annuel recommandé pour intégrer les innovations
- Application aux prévisions d'étiquetage énergétique et émission
""")
st.image("images/industry_use.jpg", caption="Impact industriel", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("📘 Fait par Polina Quignon, Vincent Guillemot, Denis Froment — avec l'accompagnement de Eliott Douieb.")