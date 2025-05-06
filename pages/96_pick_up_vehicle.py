import streamlit as st

# Page configuration
st.set_page_config(page_title="Update Variables", page_icon="🔄")

st.title("Mise à Jour des Variables 🔄")

# Ensure session state variables exist
if "x" not in st.session_state:
    st.session_state.x = 0
if "y" not in st.session_state:
    st.session_state.y = 0
if "z" not in st.session_state:
    st.session_state.z = 0

st.write(f"Valeurs actuelles dans la session : **x={st.session_state.x}, y={st.session_state.y}, z={st.session_state.z}**")

# Button to update session variables
if st.button("Mettre à jour x=7, y=8, z=9"):
    st.session_state.x = 7
    st.session_state.y = 8
    st.session_state.z = 9
    st.success("Les variables ont été mises à jour ! Les nouvelles valeurs seront visibles sur les autres pages.")

st.write("Allez sur d'autres pages pour voir les nouvelles valeurs ! 🚀")