# Page Name : Prédire l'émission de CO2
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

from init_notebook import *

# Liste des modèles disponibles
models_dict = {
    "Régression Linéaire Multiple": "reg_linear_multiple.pkl",
    "k_NN": "knn_model_distance_manh_10.pkl",
    "random_forest": "reg_rf.pkl",
    # Ajouter d'autres modèles ici
    # reg_rf.pkl too big => store on google drive and download from there with gdown. To be checked before exam
}

# Boîte de sélection pour choisir un modèle
selected_model_name = st.sidebar.selectbox(
    "Sélectionnez un modèle :", list(models_dict.keys())
)

# Charger dynamiquement le modèle sélectionné
model_file = models_dict[selected_model_name]
model = joblib.load(base_models + model_file)

# Page title
st.title("Prédiction d'émission CO2")


# Affichage du modèle sélectionné
st.write("---")
st.write(f"Prédictions avec le modèle : **{selected_model_name}**")
st.write("---")


# Load the saved scalers and the trained model
robust_scaler = joblib.load(base_models + 'robust_scaler.pkl')
min_max_scaler = joblib.load(base_models + 'min_max_scaler.pkl')

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


# Valeurs initiales pour chaque colonne
default_values = {
    "m (kg)": 1350, "W (mm)": 2690, "At1 (mm)": 1510, "ec (cm3)": 1500, "ep (KW)": 77,
    "z (Wh/km)": 0, "Electric range (km)": 0,
    "IT28": 0, "IT29": 0, "IT32": 0, "IT33": 0, "IT35": 0, "IT37": 0, "IT38": 0, "IT39": 0,
    "Ft_diesel/electric": 1, "Ft_petrol": 0, "Ft_petrol/electric": 0,
    "Cr_M1G": 0, "Cr_M1S": 1, "Cr_N1G": 0, "Fm_H": 0, "Fm_M": 1, "Fm_P": 0
}

# Combine all columns in the correct order
all_columns = robust_cols + min_max_cols + binary_cols

# Sidebar pour les caractéristiques du véhicule
st.sidebar.header("Caractéristiques du véhicule")
user_inputs = {}
for col in all_columns:
    if col in binary_cols:
        # Checkbox pour les colonnes binaires
        user_inputs[col] = int(st.sidebar.checkbox(feature_name_mapping[col], value=bool(default_values[col])))
    else:
        # Input numérique pour les colonnes scalées
        user_inputs[col] = st.sidebar.number_input(feature_name_mapping[col], value=float(default_values[col]))

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
st.write(f"Le modèle entraîné prévoit une émission de CO2 de :")

# Appliquer des styles conditionnels en fonction de la valeur de la prédiction
if prediction > 200:
    color = "#FF0000"  # Rouge
elif prediction > 150:
    color = "#FFA500"  # Orange
else:
    color = "#4CAF50"  # Vert

st.markdown(
    f"<div style='text-align: center; font-size: 40px; font-weight: bold; color: {color}; margin: 20px 0;'>"
    f"{prediction:.1f} g/km</div>",
    unsafe_allow_html=True
)

# Vérifier les groupes pour les incohérences
fuel_types = ["Ft_petrol", "Ft_petrol/electric", "Ft_diesel/electric"]
selected_fuel_types = sum([user_inputs[f] for f in fuel_types])
if selected_fuel_types > 1:
    st.warning("⚠️ Plus d'une case cochée parmi 'Essence', 'Essence/électrique' et 'Diesel+électrique'. Combinaison peu réaliste.")

cr_types = ["Cr_M1G", "Cr_M1S", "Cr_N1G"]
selected_cr_types = sum([user_inputs[cr] for cr in cr_types])
if selected_cr_types > 1:
    st.warning("⚠️ Plus d'une case cochée parmi les types de transport (Cr...). Veuillez vérifier vos choix.")

fuel_modes = ["Fm_H", "Fm_M", "Fm_P"]
selected_fuel_modes = sum([user_inputs[fm] for fm in fuel_modes])
if selected_fuel_modes > 1:
    st.warning("⚠️ Plus d'une case cochée parmi les modes de carburant 'Hybride', 'Monofuel' et 'Plug-in'. Combinaison peu réaliste.")



# Debugging section (optional)
#st.write("### Informations pour debug")
#st.write("Données entrées brut :")
#st.dataframe(vehicle_data)
#st.write("Données après mise à l'échelle :")
#st.dataframe(vehicle_data_scaled)