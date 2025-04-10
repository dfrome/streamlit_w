# ce fichier contient des fonctions utilitaires pour les pages streamlit du projet "emissions CO2"
import streamlit as st


def display_model_parameters(model, X_test=None):
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
    if hasattr(model, "feature_importances_") and X_test is not None:
        st.write("**Graphique des importances des variables :**")
        # Créer un DataFrame associant noms des variables et importances
        importances = pd.DataFrame({
            "Variable": X_test.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        # Afficher le graphique
        st.bar_chart(importances.set_index("Variable"))
#    if hasattr(model, "feature_importances_"):
#        st.write(f"- Importance des variables : {model.feature_importances_}")
    
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

def display_model_parameters_classification(model):
    """
    Affiche les paramètres spécifiques du modèle de classification fourni.
    Compatible avec les modèles que nous avons retenu:
    Logistique, k-NN, Decision Tree, Random Forest,
    SVM, Naive Bayes, GBM (XGBoost, LightGBM, CatBoost), et réseaux de neurones.
    """
    st.write("**Paramètres spécifiques du modèle :**")
    
    # Régression Logistique
    if isinstance(model, LogisticRegression):
        st.write(f"- Coefficients : {model.coef_}")
        st.write(f"- Intercept : {model.intercept_}")
    
    # k-Nearest Neighbors
    elif isinstance(model, KNeighborsClassifier):
        st.write(f"- Nombre de voisins (n_neighbors) : {model.n_neighbors}")
        st.write(f"- Métrique utilisée : {model.metric}")
    
    # Decision Tree
    elif isinstance(model, DecisionTreeClassifier):
        st.write(f"- Critère de division : {model.criterion}")
        st.write(f"- Profondeur maximale (max_depth) : {model.max_depth}")
    
    # Random Forest
    elif isinstance(model, RandomForestClassifier):
        st.write(f"- Nombre d'arbres (n_estimators) : {model.n_estimators}")
        st.write(f"- Critère de division : {model.criterion}")
    
    # Support Vector Machine (SVM)
    elif isinstance(model, SVC):
        st.write(f"- Type de noyau (kernel) : {model.kernel}")
        st.write(f"- Paramètre C : {model.C}")
    
    # Naive Bayes
    elif isinstance(model, GaussianNB):
        st.write(f"- Variance des classes : {model.var_smoothing}")
    
    # GBM – XGBoost
    elif isinstance(model, XGBClassifier):
        st.write(f"- Nombre d'arbres : {model.n_estimators}")
        st.write(f"- Taux d'apprentissage (learning_rate) : {model.learning_rate}")
        st.write(f"- Profondeur maximale des arbres : {model.max_depth}")
    
    # GBM – LightGBM
    elif isinstance(model, LGBMClassifier):
        st.write(f"- Nombre d'itérations (n_estimators) : {model.n_estimators}")
        st.write(f"- Taux d'apprentissage (learning_rate) : {model.learning_rate}")
        st.write(f"- Nombre de feuilles (num_leaves) : {model.num_leaves}")
    
    # GBM – CatBoost
    elif isinstance(model, CatBoostClassifier):
        st.write(f"- Nombre d'itérations : {model.get_params()['iterations']}")
        st.write(f"- Taux d'apprentissage (learning_rate) : {model.get_params()['learning_rate']}")
    
    # Réseaux de neurones – couches denses
    elif isinstance(model, MLPClassifier):
        st.write(f"- Nombre de couches cachées : {model.hidden_layer_sizes}")
        st.write(f"- Taux d'apprentissage : {model.learning_rate}")
        st.write(f"- Activation utilisée : {model.activation}")
    
    else:
        st.write("(Nota: Modèle non reconnu ou pas encore pris en charge dans la fonction d'affichage des paramètres)")