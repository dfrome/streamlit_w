import streamlit as st

st.set_page_config(
    page_title="Projet Datascientest - émission de CO2 des véhicules",
    page_icon=":earth_africa:",
)

st.write("# Emission de CO2 des véhicules  :earth_africa:")

st.sidebar.success("Sélectionnez un thème.")

st.markdown(
    """
    Cette page présente les relations entre caractéristiques de véhicules et leur émission de CO2
    Projet d'apprenants de l'organisme Datascientest
    **👈 Faites votre choix depuis le volet d'exploration** pour développer un thème.
      
    ### Fait par
    - Polina, Vincent, Denis
    - Formation continue
    - Métier: Data Scientist
    - Cohorte Septembre 2024
        
    ### Source des données
    - Source de données : année 2023 de [eea.europa.eu](http://co2cars.apps.eea.europa.eu/)
    - Organisme de formation: (http://www.datascientest.com)
"""
)