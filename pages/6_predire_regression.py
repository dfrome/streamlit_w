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
all_columns = robust_cols + min_max_cols

# Page title
st.title("Prédiction d'émission CO2")
st.write("Entrez les caractéristiques de votre véhicule.")

# User inputs for each feature
st.sidebar.header("Caractéristiques de véhicule")
user_inputs = {}
for col in all_columns:
    user_inputs[col] = st.sidebar.number_input(f"{col}:", value=0.0)

# Convert user inputs into a DataFrame
vehicle_data = pd.DataFrame([user_inputs])

# Scale the input data
vehicle_data_scaled = vehicle_data.copy()
if robust_cols:
    vehicle_data_scaled[robust_cols] = robust_scaler.transform(vehicle_data[robust_cols])
if min_max_cols:
    vehicle_data_scaled[min_max_cols] = min_max_scaler.transform(vehicle_data[min_max_cols])

# Make a prediction
prediction = model.predict(vehicle_data_scaled)[0]

# Display the prediction
st.subheader("Emission CO2 prédite")
st.write(f"Emission de CO2 prédite pour ce véhicule : **{prediction:.2f} g/km **")

# Debugging option (optional, can be removed)
st.write("### Debugging Information")
st.write("Données saisies:")
st.dataframe(vehicle_data)
st.write("Données mises à l'échelle:")
st.dataframe(vehicle_data_scaled)