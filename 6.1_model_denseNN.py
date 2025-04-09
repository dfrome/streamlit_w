# %% [markdown]
# <.ipynb>  ce fichier a été créé pour évaluer le run avec GPU sous linux
# #
# ##### Projet CO2 par Polina, Vincent, Denis
# 
# Ce notebook:
# entraine un modèle de classification pour prédiction par Réseau de neurones denses (Fully connected Neural Network), avec fonction d'activation ReLU.
# 
# Prend en entrée les fichiers:
#     (processed)/X_test_scaled.csv, X_train_scaled.csv, y_test_cat.csv, y_train_cat.csv : les données scalées et donc forcément préalablement séparées en jeux de train/test.
# 
# Fournit en sortie les fichiers:
# 
#     (models)/<nom_de_modele>.pkl
# 

# %% [markdown]
# # Initialisation de variables et fonctions

# %%
# charge les chemins vers les fichiers de données : base_processed, base_raw, base_models...
import init_notebook

random_state = 42
n_jobs = -1

# %%
# Les fonctions utiles à plusieurs modèles
from common_co2 import load_our_data_cat, display_norm_matrix, display_roc, adjust_with_penalty, check_differences

# %%
"""
!pip install tensorflow
"""

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# %%
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "16"  # nombre de cœurs physiques réels
"""

# %%
X_train_scaled, X_test_scaled, y_train, y_test=load_our_data_cat()

# %%


# %%
# dimensions for debugging

# paramétrisation des couches
layer_01_neurons = 32
layer_02_neurons = 4
activation_01 = 'relu'
activation_02 = 'relu'
activation_output = 'softmax'
dropout_rate = 0.2
# paramétrisation de l'optimiseur
optimizer = 'adam'
# my_loss = 'categorical_crossentropy' # seulement si y_train est one-hot
my_loss = 'sparse_categorical_crossentropy'  # seulement si y_train est une série
metrics = ['accuracy']
# paramétrisation de l'apprentissage
epochs = 3
batch_size = 32


# %%
# experiments

# paramétrisation des couches

activation_output = 'softmax'
dropout_rate = 0.1
# paramétrisation de l'optimiseur
optimizer = 'adam'
my_loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
# paramétrisation de l'apprentissage
epochs = 180
batch_size = 800


# %%
# Liste des couches avec leurs paramètres
layers_config = [
    {"neurons": 512, "activation": "relu"},
    {"neurons": 512, "activation": "relu"},
    {"neurons": 256, "activation": "relu"},
    #{"neurons": 256, "activation": "relu"},  # au final, 4 couvhes donnent la même résultat que 9 couches
    #{"neurons": 256, "activation": "relu"},
    #{"neurons": 256, "activation": "relu"},
    #{"neurons": 256, "activation": "relu"},
    #{"neurons": 256, "activation": "relu"},
    {"neurons": 128, "activation": "relu"}
]

# %%


# TODO : on n'est pas obligé de faire du one-hot encoding pour y_test et y_train
# mais si on veut ensuite appliquer des pénalités au modèle il faudra alors le faire, 
# et utiliser my_loss='categorical_crossentropy'
import time
start_time=time.time()

num_classes = len(set(y_train)) 

# Adjust class labels to start from 0
y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1


# Définir le modèle de réseau de neurones
model = Sequential()
# Première couche avec input_dim explicitement spécifié
model.add(Dense(layers_config[0]["neurons"], activation=layers_config[0]["activation"], input_dim=X_train_scaled.shape[1]))
model.add(Dropout(dropout_rate))

# Ajout des couches restantes dynamiquement
for layer in layers_config[1:]:
    model.add(Dense(layer["neurons"], activation=layer["activation"]))
    model.add(Dropout(dropout_rate))

# Couche de sortie
model.add(Dense(num_classes, activation=activation_output))

# Compiler le modèle
model.compile(optimizer=optimizer, loss=my_loss, metrics=metrics)

# Entraîner le modèle
history = model.fit(X_train_scaled, y_train_adjusted, 
                    validation_data=(X_test_scaled, y_test_adjusted), 
                    epochs=epochs, 
                    batch_size=batch_size)

end_time = time.time()
exec_time = end_time - start_time
print(f"Temps d'apprentissage = {exec_time:.2f} secondes")

# %%

# Évaluer les performances sur le jeu de test
from sklearn.metrics import f1_score, recall_score

# Evaluate the model to get loss and accuracy
loss, accuracy = model.evaluate(X_test_scaled, y_test_adjusted, verbose=0)

# Predict the classes for the test set
y_pred = model.predict(X_test_scaled)
y_pred_classes = y_pred.argmax(axis=1)  # Get class indices for predictions

# Compute F1-score and Recall
f1 = f1_score(y_test_adjusted, y_pred_classes, average='weighted')  # Weighted for class imbalance
recall = recall_score(y_test_adjusted, y_pred_classes, average='weighted')

# Display the results
print(f"Test Loss: {loss:.3f}")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Recall: {recall:.3f}")

# %% [markdown]
# # Représentations graphiques des résultats pour le rapport

# %%

# Visualiser les performances pendant l'entraînement
"""# <.ipynb> 
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Performance')
plt.show()
print(f"Temps d'apprentissage = {exec_time:.0f} secondes")
"""

# %%
name='Neural Network with dense layers'
hyperparams = f'activation={layers_config[0]["activation"]}, dropout_rate={dropout_rate}, optimizer={optimizer}, loss={my_loss}, epochs={epochs}, batch_size={batch_size}'

# %%
# y_pred_classes predicts the class -1, so the marix will have to take an corrected input
y_pred_class_corrected = y_pred_classes + 1
# <.ipynb> display_norm_matrix(name, y_pred_class_corrected, y_test_adjusted+1, hyperparams=hyperparams)
print(f"Test Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Recall: {recall:.3f}")

# %%
# display_roc(X_test_scaled, y_test, y_pred, model)

# %% [markdown]
# On ne ressent pas pour ce modèle le besoin d'égaliser les classes sur le dataset d'entraînement  

# %% [markdown]
# # Tentatives d'optimisation par randomizedSearchCV

# %% [markdown]
# Ceci ne s'appliquera pas pour le paramétrage d'un réseau de neurones

# %% [markdown]
# # Tentative d'optimisation par pénalité

# %% [markdown]
# à voir si le besoin se fait sentir au vu des résultats.  
# Ce sera seulement faisable si l'on a one_hot_encoded y_train et y_test

# %%
# paramètres pour ce chapitre:
threshold = 0.012 # par exemple 0.1 pour 10% de favorisation

# %%
""" # Prédire les classes sur les données de test
y_prob = model.predict_proba(X_test_scaled)

y_adjusted_pred= adjust_with_penalty(y_prob,threshold) """

# %%
""" from sklearn.metrics import accuracy_score, f1_score, recall_score

# Calculate new metrics
adjusted_accuracy = accuracy_score(y_test, y_adjusted_pred)
adjusted_f1 = f1_score(y_test, y_adjusted_pred, average='weighted')
adjusted_recall = recall_score(y_test, y_adjusted_pred, average='weighted')  # Include recall calculation

# Print the metrics
print(f"Adjusted Accuracy: {adjusted_accuracy:.4f}")
print(f"Adjusted F1-Score: {adjusted_f1:.4f}")
print(f"Adjusted Recall: {adjusted_recall:.4f}")
 """

# %%
""" 
differences, count_misadjustments, count_right_adjustments = check_differences(y_pred, y_adjusted_pred, y_test)
display(differences)
print("\nCount of misadjustments (real_3 to 2):      ", count_misadjustments)
print("Count of rightful adjustments (real_2 to 2):", count_right_adjustments) """

# %% [markdown]
# Ici, on remarque que dès les premières lignes affectées par la pénalité, on commence à classer des lignes de classe réelles 3 en classe 2, donc cela explique qu'on n'améliore pas la performance avec la pénalité.

# %% [markdown]
# Puis faire des représentations graphiques pour le rapport

# %% [markdown]
# # Interprêtabilité des résultats

# %%
# Résumé pour non experts :
# - L'architecture montre la structure du modèle (comme un plan de construction).
# - Le résumé offre une vue générale facile à comprendre (comme une présentation rapide).
# - Les poids appris sont les compétences acquises par le modèle après "l'entraînement" (comme les connaissances d'un élève).

# Affichage de l'architecture du modèle
print("Architecture du modèle (plan de construction) :")
model_config = model.get_config()
print(model_config)

print("\n---")

# Affichage du résumé du modèle
print("Résumé du modèle (présentation rapide) :")
model.summary()

print("\n---")

# Affichage des poids appris par le modèle
print("Poids appris par le modèle (compétences acquises après entraînement) :")
model_weights = model.get_weights()
for i, weight in enumerate(model_weights):
    print(f"Poids de la couche {i + 1}: {weight}")

# %% [markdown]
# Notice pour interprétation de la parie "paramètres", avec premiers exemples de valeurs pour illustrer :  
# 1. Total params: 2,903 (11.34 KB)
# C'est le nombre total de paramètres du modèle, qu'ils soient entraînables ou non. Un paramètre est une valeur ajustable (comme les poids ou les biais dans les couches) que le modèle utilise pour effectuer des prédictions.
# 
# La taille totale en mémoire (11.34 KB) montre l'espace nécessaire pour stocker tous les paramètres.
# 
# 2. Trainable params: 967 (3.78 KB)
# Les paramètres "entraînables" sont ceux que l'optimiseur modifie lors de l'entraînement pour minimiser la fonction de perte (loss). Ils incluent les poids et les biais des couches du modèle qui participent activement à l'apprentissage.
# 
# Ici, le modèle a 967 paramètres qui seront ajustés pendant l'entraînement.
# 
# La taille en mémoire (3.78 KB) montre l'espace nécessaire pour stocker ces paramètres spécifiquement.
# 
# 3. Non-trainable params: 0 (0.00 B)
# Ce sont des paramètres qui ne sont pas modifiés durant l'entraînement, par exemple, des paramètres fixes ou partagés entre plusieurs modèles. Dans notre cas, il n'y a aucun paramètre non-entraînable, ce qui signifie que tous les paramètres du modèle sont impliqués dans l'apprentissage.
# 
# 4. Optimizer params: 1,936 (7.57 KB)
# Ces paramètres sont spécifiques à l'optimiseur utilisé (ici: adam). Ils incluent des variables auxiliaires que l'optimiseur utilise, comme les moyennes mobiles ou les moments pour optimiser la descente de gradient.
# 
# Le nombre d'optimiseur paramètres (1,936) reflète les ressources supplémentaires nécessaires pour effectuer les calculs d'optimisation.
# 
# La taille en mémoire (7.57 KB) indique l'espace requis pour ces paramètres.

# %% [markdown]
# # Sauvegarde du dernier modèle utilisé

# %%
# Chemin pour enregistrer le modèle
model_path = base_models + 'rf_nn_dense_relu.h5'
"""
# Enregistrer le modèle entier
model.save(model_path)

print(f"Modèle de réseau de neurones enregistré dans {model_path}")
"""

""" pour le charger :
from tensorflow.keras.models import load_model
model = load_model(model_path)
 """


