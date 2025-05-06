import streamlit as st

# Page configuration
st.set_page_config(page_title="Variable Calculator", page_icon="🔢")

# Initialize a reset token (used to force widget reinitialization)
if "reset_token" not in st.session_state:
    st.session_state.reset_token = 0

# Optionally initialize x, y, z if they don't exist
if "x" not in st.session_state:
    st.session_state.x = 1
if "y" not in st.session_state:
    st.session_state.y = 2
if "z" not in st.session_state:
    st.session_state.z = 3

# Sidebar: input fields with unique keys that change when the reset token changes.
x = st.sidebar.number_input(
    "Valeur de x", value=st.session_state.x, key=f"x_{st.session_state.reset_token}"
)
y = st.sidebar.number_input(
    "Valeur de y", value=st.session_state.y, key=f"y_{st.session_state.reset_token}"
)
z = st.sidebar.number_input(
    "Valeur de z", value=st.session_state.z, key=f"z_{st.session_state.reset_token}"
)

# Save current values to session state
st.session_state.x = x
st.session_state.y = y
st.session_state.z = z

# Calculated result
result = 100 * st.session_state.x + 10 * st.session_state.y + st.session_state.z

# Main frame display of the result
st.title("Calculateur de Résultat")
st.write(f"Résultat calculé: **{result}**")

# Button in the main frame to reset values
if st.button("Réinitialiser à x=4, y=5, z=6"):
    # Update session state with new default values
    st.session_state.x = 4
    st.session_state.y = 5
    st.session_state.z = 6
    # Increment the reset token to force widgets to reinitialize with new defaults
    st.session_state.reset_token += 1
    st.experimental_rerun()  # Refresh the page so the new widget keys take effect