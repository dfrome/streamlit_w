import streamlit as st
import time
import numpy as np
from PIL import Image

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# charge les chemins vers les fichiers de données : base_processed, base_raw, base_models...
from init_notebook import base_processed, base_raw, base_models
pd.set_option('future.no_silent_downcasting', True)

st.set_page_config(
    page_title="OLD Projet Datascientest - émission de CO2 des véhicules.",
    page_icon=":dart:",
)

# pour icone classification on pourra prendre 	:placard:
# sinon il y a :balances: aussi
# ref: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.markdown("# OLD Estimations de l'émission de CO2")
st.sidebar.header("OLD Nous entraînons des modèles de régression à partir des données")
st.write("Nous voulons estimer l'émission de CO2 en fonction des caractéristiques des véhicules  "
"En science des données, on parle d'un sujet de regression: estimer une valeur numérique continue.")




              
def load_our_data():
    
    X_train_scaled = pd.read_csv(base_processed + 'X_train_scaled.csv')
    X_test_scaled = pd.read_csv(base_processed + 'X_test_scaled.csv')
    y_train = pd.read_csv(base_processed + 'y_train.csv')
    y_test = pd.read_csv(base_processed + 'y_test.csv')
    X_train_scaled = X_train_scaled.replace({False: 0, True: 1}).astype(float)
    X_test_scaled = X_test_scaled.replace({False: 0, True: 1}).astype(float)
    #
    y_column = "Ewltp (g/km)"
    y_train = y_train[y_column]
    y_test = y_test[y_column]
    return X_train_scaled, X_test_scaled, y_train, y_test


# %% [markdown]
# # Régression linéaire simple

# %%
X_train_scaled, X_test_scaled, y_train, y_test=load_our_data()

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
X_train_scaled, X_test_scaled, y_train, y_test = load_our_data()

# Sélectionner uniquement la variable explicative 'ec (cm3)'
X_train_ec = X_train_scaled[['ec (cm3)']]
X_test_ec = X_test_scaled[['ec (cm3)']]

# Initialiser et entraîner le modèle de régression linéaire
linear_model = LinearRegression()
linear_model.fit(X_train_ec, y_train)

# Prédire sur les données de test
y_pred = linear_model.predict(X_test_ec)

# Calculer les métriques de performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher les résultats
st.write(f"Mean Squared Error (MSE) sur les données de test : {mse:.2f}")
st.write(f"Coefficient de Détermination (R²) sur les données de test : {r2:.2f}")

# Afficher les paramètres du modèle
st.write("Paramètres du modèle :")
st.write(f" - Coefficient (pente) : {linear_model.coef_[0]:.4f}")
st.write(f" - Intercept (ordonnée à l'origine) : {linear_model.intercept_:.4f}")




#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)

#for i in range(1, 11):
#    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#    status_text.text("%i%% Complete" % i)
#    chart.add_rows(new_rows)
#    progress_bar.progress(i)
#    last_rows = new_rows
#    time.sleep(0.05)

#progress_bar.empty()


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
#st.button("Re-run")