# ce fichier contient des fonctions utilitaires pour les pages streamlit du projet "emissions CO2"
import streamlit as st


def display_model_parameters(model):
    """
    Affiche les paramètres spécifiques des modèles de régression et de classification.
    Compatible avec les modèles suivants :
    - Régression linéaire
    - Régression polynomiale
    - Ridge
    - Lasso
    - Elastic Net
    - SVR
    - Arbres de décision
    - Forêt aléatoire
    - k-NN
    """
    st.write("**Paramètres spécifiques du modèle :**")
    
    # Régression linéaire et modèles avec coefficients
    if hasattr(model, "coef_"):
        st.write(f"- Coefficients : {model.coef_}")
    if hasattr(model, "intercept_"):
        st.write(f"- Intercept : {model.intercept_}")
    
    # Modèles Ridge, Lasso, Elastic Net (attributs similaires)
    if hasattr(model, "alpha"):
        st.write(f"- Alpha (facteur de régularisation) : {model.alpha}")
    
    # Support Vector Regressor (SVR)
    if hasattr(model, "C"):
        st.write(f"- Paramètre C : {model.C}")
    if hasattr(model, "kernel"):
        st.write(f"- Noyau (kernel) : {model.kernel}")
    
    # Arbres de décision et Forêt aléatoire
    if hasattr(model, "max_depth"):
        st.write(f"- Profondeur maximale (max_depth) : {model.max_depth}")
    if hasattr(model, "n_estimators"):
        st.write(f"- Nombre d'arbres (n_estimators) : {model.n_estimators}")
    if hasattr(model, "feature_importances_"):
        st.write(f"- Importance des variables : {model.feature_importances_}")
    
    # Modèles k-NN (K-Nearest Neighbors)
    if hasattr(model, "n_neighbors"):
        st.write(f"- Nombre de voisins (n_neighbors) : {model.n_neighbors}")
    if hasattr(model, "weights"):
        st.write(f"- Type de pondération (weights) : {model.weights}")
    
    # Gestion générique des autres attributs
    # à l'utilisation , cela fournit bien trop de lignes !
    # attributes = [attr for attr in dir(model) if not attr.startswith("_") and not callable(getattr(model, attr))]
    #if attributes:
    #    st.write("**Autres attributs disponibles :**")
    #    for attr in attributes:
    #        st.write(f"- {attr} : {getattr(model, attr)}")
    #else:
    #    st.write("- Aucun paramètre spécifique détecté pour ce modèle.")

