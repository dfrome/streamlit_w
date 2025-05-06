import streamlit as st

# Page configuration
st.set_page_config(page_title="Variable Calculator", page_icon="🔢")

# Initialize session state variables if they don't exist yet.
if "x" not in st.session_state:
    st.session_state.x = 0
if "y" not in st.session_state:
    st.session_state.y = 0
if "z" not in st.session_state:
    st.session_state.z = 0
if "result" not in st.session_state:
    st.session_state.result = None
if "form_key" not in st.session_state:
    st.session_state.form_key = "form_1"
if "reset_clicked" not in st.session_state:
    st.session_state.reset_clicked = False  # Flag to show warning after reset

# Sidebar Form using the dynamic form key.
with st.sidebar.form(key=st.session_state.form_key):
    st.sidebar.header("Entrée des variables")

    # The default values come from session_state.
    x_input = st.number_input("Valeur de x", value=st.session_state.x, key="x_input")
    y_input = st.number_input("Valeur de y", value=st.session_state.y, key="y_input")
    z_input = st.number_input("Valeur de z", value=st.session_state.z, key="z_input")
    
    # The Submit button for the form.
    submitted = st.form_submit_button("Calculer")
    st.session_state.x = x_input
    st.session_state.y = y_input
    st.session_state.z = z_input
    
    if submitted:
        # Save the new values in session_state and compute the result.
        st.session_state.x = x_input
        st.session_state.y = y_input
        st.session_state.z = z_input
        st.session_state.result = 100 * x_input + 10 * y_input + z_input
        
        # Clear the reset flag since the user has (re)submitted the form.
        st.session_state.reset_clicked = False

# Main display area
st.title("Calculateur de Résultat")
if st.session_state.result is not None:
    st.write(f"Résultat calculé: **{st.session_state.result}**")
    
# If reset was clicked earlier, show a warning prompting the user to re-submit.
if st.session_state.reset_clicked:
    st.warning("Les valeurs ont été réinitialisées. Veuillez soumettre à nouveau le formulaire pour recalculer.")

# A button at the bottom of the main area to reset form values.
if st.button("Réinitialiser à x=4, y=5, z=6"):
    # Update session state with reset values.
    st.session_state.x = 4
    st.session_state.y = 5
    st.session_state.z = 6
    st.session_state.result = None
    
    # Set the reset_clicked flag to display the warning.
    st.session_state.reset_clicked = True
    
    # Change the form key to force reinitialization of form inputs
    st.session_state.form_key = f"form_{st.session_state.x}_{st.session_state.y}_{st.session_state.z}"
