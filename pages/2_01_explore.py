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

image_path = "images/vehiclesByFtByCty_01.png"
image = Image.open(image_path)
st.image(image, caption="Type de carburant des véhicules par pays.")
st.write("""
         """)

image_path = "images/CO2ByFt.png"
image = Image.open(image_path)
st.image(image, caption="Comparaison des émissiones de CO2 par type de carburant.")
st.write("""
         """)

image_path = "images/CorrectionsEmpattement.jpg"
image = Image.open(image_path)
st.image(image, caption="Outliers d'empattements comparés à la masse.")

st.write("""Les donnés sont très complètes.  
    Il y a toutefois beaucoup d'erreurs manifestes de saisie.   
    Supprimons les données inutilisables, traitons les aberrations et interprêtons les codes.     """)



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

