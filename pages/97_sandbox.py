import streamlit as st

# Page configuration
st.set_page_config(page_title="Variable Calculator", page_icon="🔢")

# Initialize session state
if "x" not in st.session_state:
    st.session_state.x = 1
if "y" not in st.session_state:
    st.session_state.y = 2
if "z" not in st.session_state:
    st.session_state.z = 3

# Sidebar input fields
st.sidebar.header("Entrée des variables")
x = st.sidebar.number_input("Valeur de x", value=st.session_state.x, key="x")
y = st.sidebar.number_input("Valeur de y", value=st.session_state.y, key="y")
z = st.sidebar.number_input("Valeur de z", value=st.session_state.z, key="z")

# Calcul du résultat
result = 100 * x + 10 * y + z

# Main display
st.title("Calculateur de Résultat")
st.write(f"Résultat calculé: **{result}**")

# Button to reset values
if st.button("Réinitialiser à x=4, y=5, z=6"):
    st.session_state.x = 4
    st.session_state.y = 5
    st.session_state.z = 6
    st.experimental_rerun()  # Force update of the inputs
