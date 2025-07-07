# Page Name: Final

import streamlit as st
import time
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Prédiction d'émission de CO2 des véhicules.",
    page_icon=":checkered_flag:",
)

#st.markdown("# Prédiction d'émission de CO2 des véhicules")
st.sidebar.header("Conclusion de notre projet")


images = [
    {"path": "images/co2_digest.png", "caption": ""},
]

st.set_page_config(page_title="Leçons apprises avec projet CO₂", layout="wide")


# Section: Données & Préparation
st.markdown("### 🔍 Données & Exploration: bien comprendre nos données")
st.markdown("""
- Choix de la base : Privilégier la richesse des données pour mieux entraîner les modèles
- Visualiser les corrélations, les distributions propres au métier pour ne pas faire d'erreur
            """)

# Section: Visualisations clés
st.markdown("### 🔧 Feature engineering : une clé pour la qualité des entraînements")
st.markdown("""
- Traiter rigoureusement et en fonction du métier: erreurs, outliers, valeurs manquantes et doublons, corrélations fortes
""")

# Section: Modèles 
st.markdown("### 📈 Modèles de régression puis de classification")
st.markdown("""
- Le choix des hyperparamètres est crucial pour la performance des modèles
- Les données sont adaptées aux type GBM et Random Forest. Attention à la généralisation
- Besoin de SHAP pour expliquer les influences des variables explicatives
""")
#st.image("images/results_regression01.png", caption="Comparaison des modèles de régression", use_container_width =True)
#st.image("images/classif_results_small.png", caption="Modèles de classification", use_container_width =True)
col1, col2 = st.columns(2)

with col1:
    st.image("images/results_regression_small.png", caption="Modèles de régression", use_container_width=False)

with col2:
    st.image("images/classif_results_small.png", caption="Modèles de classification", use_container_width=False)


# Section: Interprétabilité & Insights
st.markdown("### Recommandations métier")
st.markdown("""
- Recommandations techniques :
  - Essence/Diesel : contenir la masse et la cylindrée pour réduire les émissions
  - Hybride : augmenter l'autonomie pour réduire les émissions
  - Interaction des divers paramètres: utiliser notre simulateur !
""")
st.image("images/classif_shap.png", caption="Interprétabilité SHAP", use_container_width =True)

# Footer
st.markdown("---")
st.markdown("🎬 Fait par Polina Quignon, Vincent Guillemot, Denis Froment — avec la supervision d'Eliott Douieb.")