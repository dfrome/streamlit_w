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


# Section: Données & Préparation
st.markdown("### 🔍 Données & Exploration: bien comprendre nos données")
st.markdown("""
- Choix de la base : Nous avons favorisé la richesse plutôt que la jeunesse des données
- Visualisation des corrélations, des distributions marquées par le métier
            """)
st.image("images/data_cleaning.jpg", caption="Traitement des données", use_column_width=True)

# Section: Visualisations clés
st.markdown("### 📈 Feature engineering : une clé pour la qualité des entraînements")
st.markdown("""
- Traitement des corrélations fortes : élimination nécessaire de la consommation (cause vs effet), réduction de dimension pour les voies.
- Traitement rigoureux et respectueux du métier pour les erreurs, valeurs manquantes, outliers, et doublons
- Création de nouvelles variables (e.g., innovations, types d’énergie)
""")
st.image("images/visualisation.jpg", caption="Exemples de visualisation", use_column_width=True)

# Section: Modèles de régression
st.markdown("### Modèles de régression testés")
st.markdown("""
- Régression linéaire : insuffisante (hétéroscédasticité, non-normalité)
- SVR : bonne adaptation, mais temps de calcul élevé
- Random Forest : **Meilleur modèle**, R² = 0.985, MSE ≈ 49
""")
st.image("images/results_regression01.png", caption="Comparaison des modèles de régression", use_column_width=True)

# Section: Modèles de classification
st.markdown("### 🏷️ Modèles de classification")
st.markdown("""
- LightGBM & XGBoost : **Performances excellentes**, F1-score ≈ 0.90
- SHAP utilisé pour expliquer les décisions des modèles, feature importance
""")
st.image("images/classif_results_small.png", caption="Modèles de classification", use_column_width=True)

# Section: Interprétabilité & Insights
st.markdown("### 🧠 Interprétabilité & Recommandations métier")
st.markdown("""
- Importance des variables : autonomie électrique, masse, cylindrée
- Recommandations techniques :
  - Essence/Diesel : réduire masse et puissance
  - Hybride : augmenter autonomie
  - Interaction des divers paramètres: utiliser notre simulateur
""")
st.image("images/classif_shap.png", caption="Interprétabilité SHAP", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("📘 Fait par Polina Quignon, Vincent Guillemot, Denis Froment — avec la supervicion d'Eliott Douieb.")