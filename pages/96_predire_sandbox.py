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

# Charger les scalers enregistrés
robust_scaler = joblib.load(base_models + 'robust_scaler.pkl')
min_max_scaler = joblib.load(base_models + 'min_max_scaler.pkl')

# Définir les colonnes et les catégories de mise à l'échelle
robust_cols = ["m (kg)", "W (mm)", "At1 (mm)", "ec (cm3)", "ep (KW)"]
min_max_cols = ["z (Wh/km)", "Electric range (km)"]
binary_cols = [
    "IT28", "IT29", "IT32", "IT33", "IT35", "IT37", "IT38", "IT39",
    "Ft_diesel/electric", "Ft_petrol", "Ft_petrol/electric",
    "Cr_M1G", "Cr_M1S", "Cr_N1G", "Fm_H", "Fm_M", "Fm_P"
]

# Mapper les noms des fonctionnalités pour l'affichage
feature_name_mapping = {
    "m (kg)": "Masse (kg)",
    "W (mm)": "Empattement (mm)",
    "At1 (mm)": "Voie (mm)",
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

# Valeurs par défaut
default_values = {col: 0 for col in robust_cols + min_max_cols + binary_cols}
default_values.update({
    "m (kg)": 1350, "W (mm)": 2690, "At1 (mm)": 1510, "ec (cm3)": 1500, "ep (KW)": 77,
    "z (Wh/km)": 22, "Electric range (km)": 50, "Ft_diesel/electric": 1, "Ft_petrol": 0, "Ft_petrol/electric": 0,
    "Cr_M1G": 0, "Cr_M1S": 1, "Cr_N1G": 0, "Fm_H": 0, "Fm_M": 1, "Fm_P": 0
})

# Initialisation des valeurs dans `session_state`
if "val" not in st.session_state:
    st.session_state.val = {col: default_values[col] for col in default_values}
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {col: default_values[col] for col in default_values}

# Fonction pour synchroniser les valeurs avec session_state
def update_session_state():
    for col in default_values:
        st.session_state.user_inputs[col] = st.session_state.val[col]

# Interface utilisateur : Entrée des valeurs dans la barre latérale
st.sidebar.header("Caractéristiques du véhicule")
for col in default_values:
    if col in binary_cols:
        st.session_state.val[col] = st.sidebar.checkbox(
            feature_name_mapping.get(col, col),
            value=bool(st.session_state.val[col]),
            key="val_" + col,
            on_change=update_session_state
        )
    else:
        st.session_state.val[col] = st.sidebar.number_input(
            feature_name_mapping.get(col, col),
            value=int(st.session_state.val[col]),
            key="val_" + col,
            on_change=update_session_state
        )

# Image pour précharger les valeurs
st.sidebar.image(base_images + "preload_vehicle_01.jpeg", caption="Cliquez pour charger ce véhicule")

# Bouton pour précharger les valeurs
if st.sidebar.button("Charger ce véhicule"):
    st.session_state.val.update({
        "m (kg)": 1293, "W (mm)": 2638, "At1 (mm)": 1558, "ec (cm3)": 999, "ep (KW)": 67, "z (Wh/km)": 0, "Electric range (km)": 0,
        "IT29": 1, "IT37": 1, "Ft_petrol": 1, "Ft_diesel/electric": 0, "Ft_petrol/electric": 0,
        "Cr_M1G": 0, "Cr_M1S": 0, "Cr_N1G": 0, "Fm_M": 1, "Fm_H": 0, "Fm_P": 0, "IT28": 0, "IT32": 0,
        "IT33": 0, "IT35": 0, "IT38": 0, "IT39": 0,
    })
    update_session_state()
    st.sidebar.success("Valeurs préchargées avec succès !")

# Convertir les entrées utilisateur en DataFrame
vehicle_data = pd.DataFrame([{col: st.session_state.user_inputs.get(col, 0) for col in default_values}])

# Mise à l'échelle des colonnes
vehicle_data_scaled = vehicle_data.copy()
vehicle_data_scaled[robust_cols] = robust_scaler.transform(vehicle_data[robust_cols])
vehicle_data_scaled[min_max_cols] = min_max_scaler.transform(vehicle_data[min_max_cols])

# Vérifier si le modèle reçoit les bonnes caractéristiques avant la prédiction
missing_features = set(model.feature_names_in_) - set(vehicle_data_scaled.columns)
if missing_features:
    st.error(f"Les caractéristiques suivantes sont absentes: {missing_features}")
else:
    prediction = model.predict(vehicle_data_scaled)[0]

# Affichage de la prédiction avec couleur dynamique
st.subheader("Valeur prédite")
color = "#FF0000" if prediction > 200 else "#FFA500" if prediction > 120 else "#4CAF50"
st.markdown(f"<div style='text-align: center; font-size: 40px; font-weight: bold; color: {color}; margin: 20px 0;'>{prediction:.1f} g/km</div>", unsafe_allow_html=True)

# Vérifier les incohérences
fuel_types = ["Ft_petrol", "Ft_petrol/electric", "Ft_diesel/electric"]
selected_fuel_types = sum([st.session_state.user_inputs[f] for f in fuel_types])
if selected_fuel_types > 1:
    st.warning("⚠️ Plusieurs types de carburant sélectionnés. Veuillez vérifier vos choix.")

cr_types = ["Cr_M1G", "Cr_M1S", "Cr_N1G"]
selected_cr_types = sum([st.session_state.user_inputs[cr] for cr in cr_types])
if selected_cr_types > 1:
    st.warning("⚠️ Plusieurs catégories de transport sélectionnées. Veuillez vérifier vos choix.")

# Affichage de l'étiquette énergétique
st.write("Cette émission donnerait à ce véhicule l'étiquette :")
co2_levels = {"A": (0, 100), "B": (101, 120), "C": (121, 140), "D": (141, 160), "E": (161, 200), "F": (201, 250), "G": (251, float('inf'))}
for level, (min_val, max_val) in co2_levels.items():
    if min_val <= prediction <= max_val:
        st.image(base_images + f"label_{level.lower()}.jpg", caption=f"Émission : {min_val}-{max_val} g/km")
        break