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
st.sidebar.write("---")  # Separator line

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
    "z (Wh/km)": 22, "Electric range (km)": 50,
    "IT28": 0, "IT29": 0, "IT32": 0, "IT33": 0, "IT35": 0, "IT37": 0, "IT38": 0, "IT39": 0,
    "Ft_diesel/electric": 1, "Ft_petrol": 0, "Ft_petrol/electric": 0,
    "Cr_M1G": 0, "Cr_M1S": 1, "Cr_N1G": 0, "Fm_H": 0, "Fm_M": 1, "Fm_P": 0
}

# Combine all columns in the correct order
all_columns = robust_cols + min_max_cols + binary_cols


################################## 20250504 ####################################
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {col: default_values[col] for col in all_columns}

# mettre à jour session_state à chaque changement
def update_session_state():
    for col in all_columns:
        st.session_state.user_inputs[col] = st.session_state.val[col]

st.sidebar.header("Caractéristiques du véhicule")
for col in all_columns:
    if col in binary_cols:
        st.session_state.user_inputs[col]=st.sidebar.checkbox(
            feature_name_mapping[col],
            value=bool(st.session_state.user_inputs[col]),
            key=".val" + col,
            on_change=update_session_state
        )
    else:
        st.session_state.user_inputs[col]=st.sidebar.number_input(
            feature_name_mapping[col],
            value=int(st.session_state.user_inputs[col]),
            key=".val" + col,
            on_change=update_session_state
        )



# Image for preloading vehicle values
st.sidebar.image(base_images + "preload_vehicle_01.jpeg", caption="Cliquez pour charger ce véhicule")

# Button to preload values
    #    st.session_state.user_inputs.update({
if st.sidebar.button("Charger ce véhicule"):
    st.session_state.val.update({
        "m (kg)": 1293,
        "W (mm)": 2638,
        "At1 (mm)": 1558,
        "ec (cm3)": 999,
        "ep (KW)": 67,
        "z (Wh/km)": 0,
        "Electric range (km)": 0,
        "IT29": 1,
        "IT37": 1,
        "Ft_petrol": 1,
        "Ft_diesel/electric": 0,
        "Ft_petrol/electric": 0,
        "Cr_M1G": 0,
        "Cr_M1S": 0,
        "Cr_N1G": 0,
        "Fm_M": 1,
        "Fm_H": 0,
        "Fm_P": 0,
        "IT28": 0,
        "IT32": 0,
        "IT33": 0,
        "IT35": 0,
        "IT38": 0,
        "IT39": 0,
    })
    update_session_state()
    st.sidebar.success("Valeurs préchargées avec succès !")

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {col: default_values[col] for col in all_columns}
# Convert user inputs into a DataFrame with consistent column names
vehicle_data = pd.DataFrame([st.session_state.user_inputs])[all_columns]

################################## /20250504 ####################################


# Convert user inputs into a DataFrame with consistent column names and order
#vehicle_data = pd.DataFrame([user_inputs])[all_columns]

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
elif prediction > 120:
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
selected_fuel_types = sum([st.session_state.user_inputs[f] for f in fuel_types])
if selected_fuel_types > 1:
    st.warning("⚠️ Plus d'une case cochée parmi 'Essence', 'Essence/électrique' et 'Diesel+électrique'. Combinaison peu réaliste.")

cr_types = ["Cr_M1G", "Cr_M1S", "Cr_N1G"]
selected_cr_types = sum([st.session_state.user_inputs[cr] for cr in cr_types])
if selected_cr_types > 1:
    st.warning("⚠️ Plus d'une case cochée parmi les types de transport (Cr...). Veuillez vérifier vos choix.")

fuel_modes = ["Fm_H", "Fm_M", "Fm_P"]
selected_fuel_modes = sum([st.session_state.user_inputs[fm] for fm in fuel_modes])
if selected_fuel_modes > 1:
    st.warning("⚠️ Plus d'une case cochée parmi les modes de carburant 'Hybride', 'Monofuel' et 'Plug-in'. Combinaison peu réaliste.")

st.write(f"Cette émission donerait à ce véhicule l'étiquette :")

# afficher l'étiquette énergétique correspondante
co2_levels = {
    "A": (0, 100, "label_a.jpg", "Émission de 0 à 100 g/km"),
    "B": (101, 120, "label_b.jpg", "Émission de 101 à 120 g/km"),
    "C": (121, 140, "label_c.jpg", "Émission de 121 à 140 g/km"),
    "D": (141, 160, "label_d.jpg", "Émission de 141 à 160 g/km"),
    "E": (161, 200, "label_e.jpg", "Émission de 161 à 200 g/km"),
    "F": (201, 250, "label_f.jpg", "Émission de 201 à 250 g/km"),
    "G": (250, float('inf'), "label_g.jpg", "Émission supérieure à 250 g/km")
}
for level, (min_val, max_val, image_file, caption) in co2_levels.items():
    if min_val <= prediction <= max_val:
        st.image(base_images + image_file, caption=caption)
        break




# Debugging section (optional)
#st.write("### Informations pour debug")
#st.write("Données entrées brut :")
#st.dataframe(vehicle_data)
#st.write("Données après mise à l'échelle :")
#st.dataframe(vehicle_data_scaled)