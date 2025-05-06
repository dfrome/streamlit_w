import streamlit as st

# Page Configuration
st.set_page_config(page_title="Variable Calculator", page_icon="🔢")

# Initialize session state variables
if "x" not in st.session_state:
    st.session_state.x = 0
if "y" not in st.session_state:
    st.session_state.y = 4
if "z" not in st.session_state:
    st.session_state.z = 0
if "result" not in st.session_state:
    st.session_state.result = None  # No result initially
if "form_key" not in st.session_state:
    st.session_state.form_key = "form_1"  # Unique key for form refresh
if "reset_clicked" not in st.session_state:
    st.session_state.reset_clicked = False  # Tracks reset button usage

# Sidebar Form with Unique Key
with st.sidebar.form(key=st.session_state.form_key):
    st.sidebar.header("Entrée des variables")

    x = st.number_input("Valeur de x", value=st.session_state.x, key="x_input")
    y = st.number_input("Valeur de y", value=st.session_state.y, key="y_input")
    z = st.number_input("Valeur de z", value=st.session_state.z, key="z_input")

    submitted = st.form_submit_button("Calculer")

    if submitted:
        # Store values in session state
        st.session_state.x = x
        st.session_state.y = y
        st.session_state.z = z
        
        # Perform the calculation
        st.session_state.result = 100 * x + 10 * y + z
        
        # Reset the warning
        st.session_state.reset_clicked = False

# Main Display
st.title("Calculateur de Résultat")

if st.session_state.result is not None:
    st.write(f"Résultat calculé: **{st.session_state.result}**")

# Show warning if reset was clicked
if st.session_state.reset_clicked:
    st.warning("Les valeurs ont été réinitialisées. Veuillez soumettre à nouveau le formulaire pour recalculer.")

# Button to Reset Values
if st.button("Réinitialiser à x=4, y=5, z=6"):
    # Update session state values
    st.session_state.x = 4
    st.session_state.y = 5
    st.session_state.z = 6
    st.session_state.result = None  # Clear the previous result
    
    # Change form key to force refresh
    st.session_state.form_key = f"form_{st.session_state.x}_{st.session_state.y}_{st.session_state.z}"
    
    # Activate the warning for re-submission
    st.session_state.reset_clicked = True