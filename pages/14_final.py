# Page Name: Final

import streamlit as st
import time
import numpy as np
from PIL import Image

#st.set_page_config(page_title="Plotting Demo", page_icon=":checkered_flag:")
st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules.",
    page_icon=":checkered_flag:",
)

st.markdown("# DataScientest - emission CO2 - Digest")
st.sidebar.header("Fin de notre projet")
#st.write(
#    """Nous explorons tout d'abord les données."""
#)


images = [
    {"path": "images/co2_digest.png", "caption": ""},
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


