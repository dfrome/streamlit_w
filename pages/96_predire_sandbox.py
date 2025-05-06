# Page Name : Prédire l'émission de CO2
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

from init_notebook import *  # Assurez-vous que ce module est bien accessible !

# Liste des modèles disponibles
models_dict = {
    "Régression Linéaire Multiple": "reg_linear_multiple.pkl",
    "k_NN": "knn_model_distance_manh_10.pkl",
    "random_forest": "reg_rf.pkl",
}

# Boîte de sélection pour choisir un modèle
selected_model_name = st.sidebar.selectbox("Sélectionnez un modèle :", list(models_dict.keys()))
st.sidebar.write("---")

# Charger dynamiquement le modèle sélectionné
model_file = models_dict[selected_model_name]
model = joblib.load(base_models + model_file)

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

# Valeurs par défaut
default_values = {
    "m (kg)": 1350, "W (mm)": 2690, "At1 (mm)": 1510, "ec (cm3)": 1500, "ep (KW)": 77,
    "z (Wh/km)": 22, "Electric range (km)": 50,
    "IT28": 0, "IT29": 0, "IT32": 0, "IT33": 0, "IT35": 0, "IT37": 0, "IT38": 0, "IT39": 0,
    "Ft_diesel/electric": 1, "Ft_petrol": 0, "Ft_petrol/electric": 0,
    "Cr_M1G": 0, "Cr_M1S": 1, "Cr_N1G": 0, "Fm_H": 0, "Fm_M": 1, "Fm_P": 0
}

# Initialiser `session_state` avec les valeurs par défaut si non définies
for col, val in default_values.items():
    if col not in st.session_state:
        st.session_state[col] = val

# Ajouter une clé de formulaire dynamique pour rafraîchir le formulaire lorsque les valeurs sont modifiées
if "form_key" not in st.session_state:
    st.session_state.form_key = "form_1"

# Sidebar form for user input with dynamic key
st.sidebar.header("Caractéristiques du véhicule")
with st.sidebar.form(key=st.session_state.form_key):
    for col in default_values:
        if col in binary_cols:
            st.session_state[col] = st.checkbox(col, value=bool(st.session_state[col]))
        else:
            st.session_state[col] = st.number_input(col, value=int(st.session_state[col]))

    submitted = st.form_submit_button("Mettre à jour")

# Bouton pour précharger des valeurs spécifiques et rafraîchir le formulaire
st.sidebar.image(base_images + "preload_vehicle_01.jpeg", caption="Cliquez pour charger ce véhicule")
if st.sidebar.button("Charger ce véhicule"):
    predefined_values = {
        "m (kg)": 1293, "W (mm)": 2638, "At1 (mm)": 1558, "ec (cm3)": 999, "ep (KW)": 67,
        "z (Wh/km)": 0, "Electric range (km)": 0, "IT29": 1, "IT37": 1, "Ft_petrol": 1,
        "Ft_diesel/electric": 0, "Ft_petrol/electric": 0, "Cr_M1G": 0, "Cr_M1S": 0, "Cr_N1G": 0,
        "Fm_M": 1, "Fm_H": 0, "Fm_P": 0, "IT28": 0, "IT32": 0, "IT33": 0, "IT35": 0,
        "IT38": 0, "IT39": 0,
    }
    st.session_state.update(predefined_values)
    
    # Changer la clé du formulaire pour forcer la mise à jour
    st.session_state.form_key = f"form_{st.session_state['m (kg)']}"
    

# Convertir les entrées utilisateur en DataFrame
vehicle_data = pd.DataFrame([{col: st.session_state[col] for col in default_values}])

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
st.markdown(
    f"<div style='text-align: center; font-size: 40px; font-weight: bold; color: {color}; margin: 20px 0;'>"
    f"{prediction:.1f} g/km</div>", unsafe_allow_html=True
)

# Vérifier les incohérences de groupes
fuel_types = ["Ft_petrol", "Ft_petrol/electric", "Ft_diesel/electric"]
selected_fuel_types = sum([st.session_state[f] for f in fuel_types])
if selected_fuel_types > 1:
    st.warning("⚠️ Plusieurs types de carburant sélectionnés. Veuillez vérifier vos choix.")

cr_types = ["Cr_M1G", "Cr_M1S", "Cr_N1G"]
selected_cr_types = sum([st.session_state[cr] for cr in cr_types])
if selected_cr_types > 1:
    st.warning("⚠️ Plusieurs catégories de transport sélectionnées. Veuillez vérifier vos choix.")

fuel_modes = ["Fm_H", "Fm_M", "Fm_P"]
selected_fuel_modes = sum([st.session_state[fm] for fm in fuel_modes])
if selected_fuel_modes > 1:
    st.warning("⚠️ Plusieurs modes de carburant sélectionnés. Veuillez vérifier vos choix.")

# Affichage de l'étiquette énergétique
st.write("Cette émission donnerait à ce véhicule l'étiquette :")
co2_levels = {
    "A": (0, 100), "B": (101, 120), "C": (121, 140), "D": (141, 160), "E": (161, 200),
    "F": (201, 250), "G": (251, float('inf'))
}
for level, (min_val, max_val) in co2_levels.items():
    if min_val <= prediction <= max_val:
        st.image(base_images + f"label_{level.lower()}.jpg", caption=f"Émission : {min_val}-{max_val} g/km")
        break