# Page Name: Feature engineering
import streamlit as st
import time
import numpy as np
from PIL import Image

#st.set_page_config(page_title="Plotting Demo", page_icon=":wrench:")
st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules.",
    page_icon=":wrench:",
)

st.markdown("# CO2 Feature engineering")
st.sidebar.header("Nous corrigeons les données")

st.write("""Dans les données originales, il y a beaucoup d'erreurs manifestes de saisie.   
    Nous supprimons les données inutilisables (exemple hybrides sans autonomie),  
    Nous traitons les colonnes ayant des valeurs manquantes.  
    Nous corrigeons ou supprimpons les aberrations.  
""")

st.write("""Quelques particularités à noter: \n
         Pour les voies avant et arrière (At1 et At2) : le pairplot montre une relation très linéaire entre elles.  
Mais nous avons aussi voulu le confirmer statistiquement avec des tests de corrélation statistiques.
""")
image_path = "images/at1at2.jpg"
image = Image.open(image_path)
st.image(image, caption="\n")

image_path = "images/cylindrees.png"
image = Image.open(image_path)
st.image(image, caption="Outliers précieux du point de vue métier.")

image_path = "images/CorrectionsEmpattement.jpg"
image = Image.open(image_path)
st.image(image, caption="Outliers d'empattements comparés à la masse.")

st.write("""
On ajoute des features en explosant les codes d'innovative technology, exemple 'E9 29 37'=> IT29=1 et IT37=1:
         """)
image_path = "images/innov_tech.png"
image = Image.open(image_path)
st.image(image, caption="distribution selon types d'innovations embarquées.")

st.write("""
Au final après tous ces traitements, on harmonise l'échelle des valeurs numériques""")

image_path = "images/scaled.png"
image = Image.open(image_path)
st.image(image, caption="disctributionDistribution des variables explicatives numériques après scaling.")


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
#st.button("Re-run")

