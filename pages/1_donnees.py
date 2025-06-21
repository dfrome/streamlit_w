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
]

# Affichage des images
for img in images:
    try:
        image = Image.open(img["path"])
        st.image(image, caption=img["caption"])
        st.write("")  # Pour espacer les images
    except FileNotFoundError:
        st.error(f"Image non trouvée : {img['path']}")

#st.write("""Les donnés sont très complètes.  
#    Toutefois, dans la prochaine phase, il faut corriger la qualité de ces données.  """)


