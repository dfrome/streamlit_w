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

# Map visible feature names for user-friendly display
feature_name_mapping = {
    "m (kg)": "Masse (kg)",
    "W (mm)": "Empattement (mm)",
    "At1 (mm)": "voie (mm)",
    "ec (cm3)": "Cylindrée (cm3)",
    "ep (KW)": "Puissance en KW=1.36 * puiss CV",
    "z (Wh/km)": "Conso élec (Wh/km)",
    "Electric range (km)": "Autonomie électrique (km)",
    "IT28": "IT28",
    "IT29": "IT29",
    "IT32": "IT32",
    "IT33": "IT33",
    "IT35": "IT35",
    "IT37": "IT37",
    "IT38": "IT38",
    "IT39": "IT39",
    "Ft_diesel/electric": "Diesel+électrique",
    "Ft_petrol": "Essence",
    "Ft_petrol/electric": "Essence/électrique",
    "Cr_M1G": "Cr:transport de personnes M1G",
    "Cr_M1S": "Cr:transport de personnes M1S",
    "Cr_N1G": "Cr:transport de marchandise N1G",
    "Fm_H": "Fuel mode Hybride",
    "Fm_M": "Fuel mode Monofuel",
    "Fm_P": "Fuel mode Plug-in"
}
# Combine all columns in the correct order
all_columns = robust_cols + min_max_cols + binary_cols

# Page title
st.title("Prédiction d'émission CO2")
st.write("Entrez les caractéristiques de votre véhicule.")

# Sidebar for user input
st.sidebar.header("Caractéristiques du véhicule")
user_inputs = {}
for col in all_columns:
    if col in binary_cols:
        # Checkbox for binary columns
        user_inputs[col] = int(st.sidebar.checkbox(feature_name_mapping[col], value=False))
    else:
        # Number input for scaled columns
        default_value = 0.0
        user_inputs[col] = st.sidebar.number_input(feature_name_mapping[col], value=default_value)

# Convert user inputs into a DataFrame with consistent column names and order
vehicle_data = pd.DataFrame([user_inputs])[all_columns]

# Scale the appropriate columns
vehicle_data_scaled = vehicle_data.copy()
vehicle_data_scaled[robust_cols] = robust_scaler.transform(vehicle_data[robust_cols])
vehicle_data_scaled[min_max_cols] = min_max_scaler.transform(vehicle_data[min_max_cols])
# Binary columns remain unchanged

# Make a prediction
prediction = model.predict(vehicle_data_scaled)[0]

# Display the prediction
st.subheader("Valeur prédite")
st.write(f"La valeur prédite pour le véhicule entré est : **{prediction:.2f}**")

# Debugging section (optional)
st.write("### Informations pour debug")
st.write("Données entrées brut :")
st.dataframe(vehicle_data)
st.write("Données après mise à l'échelle :")
st.dataframe(vehicle_data_scaled)