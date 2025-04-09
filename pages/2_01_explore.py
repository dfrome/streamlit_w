import streamlit as st
import time
import numpy as np
from PIL import Image

#st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules.",
    page_icon="📈",
)

st.markdown("# CO2 Exploration des données")
st.sidebar.header("Nous explorons les données")
st.write(
    """Nous explorons tout d'abord les données."""
)

images = [
    {"path": "images/vehiclesByFtByCty_01.png", "caption": "Type de carburant des véhicules par pays, base des ventes."},
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



image_path = "images/CorrectionsEmpattement.jpg"
image = Image.open(image_path)
st.image(image, caption="Outliers d'empattements comparés à la masse.")

st.write("""Les donnés sont très complètes.  
    Il y a toutefois beaucoup d'erreurs manifestes de saisie.   
    Supprimons les données inutilisables, traitons les aberrations et interprêtons les codes.     """)

st.write("""
On ajoute des features en explosant les codes d'innovative technology:
         """)
image_path = "images/innov_tech.png"
image = Image.open(image_path)
st.image(image, caption="distribution selon types d'innovations embarquées.")



#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)

#for i in range(1, 11):
#    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#    status_text.text("%i%% Complete" % i)
#    chart.add_rows(new_rows)
#    progress_bar.progress(i)
#    last_rows = new_rows
#    time.sleep(0.05)

#progress_bar.empty()


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")

