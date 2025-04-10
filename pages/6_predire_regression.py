import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

from init_notebook import *

# Load the saved scalers and the trained model
robust_scaler = joblib.load(base_models + 'robust_scaler.pkl')
min_max_scaler = joblib.load(base_models + 'min_max_scaler.pkl')
model = joblib.load(base_models + 'reg_linear_multiple.pkl')

# Define the columns and scaling categories
robust_cols = ["m (kg)", "W (mm)", "At1 (mm)", "ec (cm3)", "ep (KW)"]
min_max_cols = ["z (Wh/km)", "Electric range (km)"]
binary_cols = [
    "IT28", "IT29", "IT32", "IT33", "IT35", "IT37", "IT38", "IT39",
    "Ft_diesel/electric", "Ft_petrol", "Ft_petrol/electric",
    "Cr_M1G", "Cr_M1S", "Cr_N1G", "Fm_H", "Fm_M", "Fm_P"
]

# Combine all columns in the correct order
all_columns = robust_cols + min_max_cols + binary_cols

# Page title
st.title("Prédiction d'émission CO2")
st.write("Entrez les caractéristiques de votre véhicule.")

# User inputs for each feature
st.sidebar.header("Caractéristiques de véhicule")
user_inputs = {}
for col in all_columns:
    default_value = 0.0 if col in robust_cols + min_max_cols else 0  # Default 0 for binary columns
    user_inputs[col] = st.sidebar.number_input(f"{col}:", value=default_value)

# Convert user inputs into a DataFrame
vehicle_data = pd.DataFrame([user_inputs])[all_columns]

# Scale the appropriate columns
vehicle_data_scaled = vehicle_data.copy()
if robust_cols:
    vehicle_data_scaled[robust_cols] = robust_scaler.transform(vehicle_data[robust_cols])
if min_max_cols:
    vehicle_data_scaled[min_max_cols] = min_max_scaler.transform(vehicle_data[min_max_cols])
# Binary columns remain unchanged

# Make a prediction
prediction = model.predict(vehicle_data_scaled)[0]

# Display the prediction
st.subheader("Emission CO2 prédite")
st.write(f"Emission de CO2 prédite pour ce véhicule : **{prediction:.1f} g/km **")

# Debugging option (optional, can be removed)
st.write("### Debugging Information")
st.write("Données saisies:")
st.dataframe(vehicle_data)
st.write("Données mises à l'échelle:")
st.dataframe(vehicle_data_scaled)