# Page Name: Données

import streamlit as st
import time
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules - données.",
    page_icon="📈",
)

st.markdown("# CO2 Choix des données")
st.sidebar.header("Nous choisissons les données")
#st.write(
#    """Nous choisissons tout d'abord les données."""
#)

images = [
    {"path": "images/introduction.png", "caption": "Type de carburant des véhicules par pays, base des ventes."},
    {"path": "images/CO2ByFt.png", "caption": "Comparaison des émissions de CO2 par type de carburant."},
    {"path": "images/CO2_byEp.png", "caption": "Relation entre puissance et émissions de CO2."},
    {"path": "images/CO2ByFt.png", "caption": "Relation entre puissance et émissions de CO2."},
    {"path": "images/relations_01.png", "caption": "Focus sur des relations entre variables explicatives et cible."},
    {"path": "images/matrice_initiale.jpg", "caption": "Matrice de corrélation entre les valeurs numériques, base des modèles."},
    
]

# Affichage des images
for img in images:
    try:
        image = Image.open(img["path"])
        st.image(image, caption=img["caption"])
        st.write("")  # Pour espacer les images
    except FileNotFoundError:
        st.error(f"Image non trouvée : {img['path']}")

st.write("""Les donnés sont très complètes.  
    Toutefois, dans la prochaine phase, il faut corriger la qualité de ces données.  """)


